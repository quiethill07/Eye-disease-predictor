import argparse
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from tqdm import tqdm

import archs
from dataset import InferenceDataset, discover_image_ids
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, type=str, help='experiment name used to load config/model from output_dir')
    parser.add_argument('--output_dir', default='outputs', type=str, help='experiment root dir')
    parser.add_argument('--model_path', default='', type=str, help='override model checkpoint path')
    parser.add_argument('--config_path', default='', type=str, help='override config file path')
    parser.add_argument('--image_dir', required=True, type=str, help='path to image folder')
    parser.add_argument('--img_exts', default='.jpg,.jpeg,.png,.bmp,.tif,.tiff', type=str, help='comma-separated image extensions')
    parser.add_argument('--recursive', default=True, type=str2bool, help='recursively discover images in image_dir')
    parser.add_argument('--batch_size', default=8, type=int, help='inference batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
    parser.add_argument('--save_dir', default='vessel_predictions', type=str, help='output prediction dir')
    parser.add_argument('--binary_subdir', default='binary_masks', type=str, help='subfolder for binary masks')
    parser.add_argument('--prob_subdir', default='prob_maps', type=str, help='subfolder for probability maps')
    parser.add_argument('--threshold', default=0.5, type=float, help='binary threshold')
    parser.add_argument('--save_binary', default=True, type=str2bool, help='save thresholded masks')
    parser.add_argument('--save_prob', default=True, type=str2bool, help='save probability maps')
    parser.add_argument('--restore_size', default=True, type=str2bool, help='resize outputs back to original image size')
    parser.add_argument('--id_prefix', default='', type=str, help='optional prefix for saved mask filenames')
    parser.add_argument('--manifest_name', default='manifest.csv', type=str, help='mapping file written in save_dir')
    return parser.parse_args()


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def resolve_paths(args):
    if args.config_path:
        config_path = args.config_path
    elif args.name is not None:
        config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    else:
        raise ValueError('Provide either --config_path or --name.')

    if args.model_path:
        model_path = args.model_path
    elif args.name is not None:
        model_path = os.path.join(args.output_dir, args.name, 'model.pth')
    else:
        raise ValueError('Provide either --model_path or --name.')

    return config_path, model_path


def torch_load_compat(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(model, ckpt_path):
    ckpt = torch_load_compat(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']

    if not isinstance(ckpt, dict):
        raise RuntimeError('Unsupported checkpoint format. Expected state_dict-like dict.')

    cleaned_ckpt = {}
    for key, value in ckpt.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned_ckpt[new_key] = value

    # strict=False here would silently leave layers randomly initialized when the
    # config and the checkpoint disagree (e.g. a different attention_mode), and
    # inference would emit plausible-looking garbage masks with no warning.
    incompatible = model.load_state_dict(cleaned_ckpt, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f'Checkpoint does not match the model built from the config.\n'
            f'  Missing keys (left randomly initialized): {missing}\n'
            f'  Unexpected keys (ignored): {unexpected}\n'
            f'Check that attention_mode / input_list / num_classes in the config '
            f'match the run that produced {ckpt_path}.'
        )


def sanitize_id(value):
    safe = []
    for c in str(value):
        if c.isalnum() or c in ['-', '_']:
            safe.append(c)
        else:
            safe.append('_')
    out = ''.join(safe).strip('_')
    return out if out else 'sample'


def discover_image_paths(image_dir, exts, recursive=True):
    paths = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(image_dir):
            full_path = os.path.join(image_dir, name)
            if not os.path.isfile(full_path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                paths.append(full_path)
    return sorted(paths)


def build_records_from_paths(paths):
    records = []
    counts = {}
    for p in paths:
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        raw_id = sanitize_id(stem)
        k = counts.get(raw_id, 0)
        counts[raw_id] = k + 1
        sample_id = raw_id if k == 0 else f'{raw_id}_{k}'
        records.append({
            'sample_id': sample_id,
            'img_id': stem,
            'img_path': p,
            'source': 'scan',
        })
    return records


def main():
    args = parse_args()
    config_path, model_path = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_yaml(config_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False),
        attention_mode=config.get('attention_mode', 'none'),
    )
    load_checkpoint(model, model_path)

    cudnn.benchmark = device.type == 'cuda'
    model = model.to(device)
    model.eval()

    exts = [x.strip().lower() for x in args.img_exts.split(',') if x.strip()]
    paths = discover_image_paths(args.image_dir, exts, recursive=args.recursive)
    if len(paths) == 0:
        # Keep a fallback for legacy behavior.
        expanded_ids = discover_image_ids(args.image_dir, exts)
        if len(expanded_ids) == 0:
            for ext in exts:
                for p in sorted(glob(os.path.join(args.image_dir, '*' + ext))):
                    paths.append(p)
        else:
            for img_id in expanded_ids:
                for ext in exts:
                    p = os.path.join(args.image_dir, img_id + ext)
                    if os.path.isfile(p):
                        paths.append(p)
                        break
    if len(paths) == 0:
        raise RuntimeError(f'No images found in {args.image_dir} with extensions: {exts}')

    records = build_records_from_paths(paths)

    # Optional prefix keeps experiment-wise uniqueness when needed.
    if args.id_prefix:
        for r in records:
            r['sample_id'] = f"{sanitize_id(args.id_prefix)}_{r['sample_id']}"

    img_ids = [r['sample_id'] for r in records]
    id_to_path = {r['sample_id']: r['img_path'] for r in records}

    first_ext = '.png'
    if len(paths) > 0:
        first_ext = os.path.splitext(paths[0])[1] or '.png'
    dataset = InferenceDataset(
        img_ids=img_ids,
        img_dir=args.image_dir,
        img_ext=first_ext,
        transform=A.Compose([
            A.Resize(config['input_h'], config['input_w']),
            A.Normalize(),
        ]),
        id_to_ext=None,
        id_prefix='',
        id_to_path=id_to_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    binary_dir = os.path.join(args.save_dir, args.binary_subdir)
    prob_dir = os.path.join(args.save_dir, args.prob_subdir)
    if args.save_binary:
        os.makedirs(binary_dir, exist_ok=True)
    if args.save_prob:
        os.makedirs(prob_dir, exist_ok=True)

    with torch.no_grad():
        for images, meta in tqdm(loader, total=len(loader)):
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            sample_ids = meta['sample_id'] if 'sample_id' in meta else meta['img_id']
            for i, (pred, sample_id) in enumerate(zip(probs, sample_ids)):
                pred_prob = pred[0]
                pred_bin = (pred_prob >= args.threshold).astype(np.uint8)

                src_path = meta['img_path'][i] if 'img_path' in meta else None

                if src_path is not None and args.restore_size:
                    src_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                    if src_img is not None:
                        h, w = src_img.shape[:2]
                        pred_prob = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
                        pred_bin = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)

                prob_u8 = np.clip(pred_prob * 255.0, 0, 255).astype(np.uint8)
                bin_u8 = (pred_bin * 255).astype(np.uint8)

                if args.save_prob:
                    cv2.imwrite(os.path.join(prob_dir, f'{sample_id}.png'), prob_u8)
                if args.save_binary:
                    cv2.imwrite(os.path.join(binary_dir, f'{sample_id}.png'), bin_u8)

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(args.save_dir, args.manifest_name)
    manifest.to_csv(manifest_path, index=False)
    print(f'Saved manifest: {manifest_path}')
    print(f'Inference completed. Output dir: {args.save_dir}')


if __name__ == '__main__':
    main()
