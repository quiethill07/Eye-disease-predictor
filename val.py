#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='evaluate val split from train or official test split')
    parser.add_argument('--val_ratio', default=-1.0, type=float,
                        help='validation ratio from train split; <0 uses the value recorded in config.yml')
    parser.add_argument('--disable_tqdm', default=False, type=str2bool, help='disable per-batch tqdm bar')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_load_compat(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    seed_torch()
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = device.type == 'cuda'

    # Must match the ratio used at training time, otherwise the "val" split is a
    # different random subset that overlaps the images the model was fit on.
    if args.val_ratio is not None and args.val_ratio >= 0:
        val_ratio = float(args.val_ratio)
    else:
        val_ratio = float(config.get('val_ratio', 0.2))
    print(f'val_ratio: {val_ratio}')

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False),
        attention_mode=config.get('attention_mode', 'none'),
    )

    model = model.to(device)

    dataset_name = config['dataset'].lower()
    img_ext = '.png'
    mask_ext = '.png'
    mask_in_class_subdir = True

    if dataset_name == 'fives':
        if args.split == 'val':
            original_dir = os.path.join(config['data_dir'], 'train', 'Original')
            gt_dir = os.path.join(config['data_dir'], 'train', 'Ground truth')
            all_ids = sorted(glob(os.path.join(original_dir, '*' + img_ext)))
            all_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_ids]
            _, val_img_ids = train_test_split(all_ids, test_size=val_ratio, random_state=config['dataseed'])
        else:
            original_dir = os.path.join(config['data_dir'], 'test', 'Original')
            gt_dir = os.path.join(config['data_dir'], 'test', 'Ground truth')
            val_img_ids = sorted(glob(os.path.join(original_dir, '*' + img_ext)))
            val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
        val_img_dir = original_dir
        val_mask_dir = gt_dir
        mask_in_class_subdir = False
        id_prefix = f'fives_{args.split}'
    else:
        # Data loading code
        img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        _, val_img_ids = train_test_split(img_ids, test_size=val_ratio, random_state=config['dataseed'])

        val_img_dir = os.path.join(config['data_dir'], config['dataset'], 'images')
        val_mask_dir = os.path.join(config['data_dir'], config['dataset'], 'masks')
        id_prefix = f'{dataset_name}_{args.split}'

    ckpt = torch_load_compat(f'{args.output_dir}/{args.name}/model.pth')

    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}

    # The old code swallowed the failure with a bare `except` and fell back to
    # strict=False, so a mismatched checkpoint produced metrics for a partly
    # random model. Report the mismatch and stop instead.
    incompatible = model.load_state_dict(ckpt, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'Checkpoint does not match the model built from config.yml.\n'
            f'  Missing keys (would stay randomly initialized): {list(incompatible.missing_keys)}\n'
            f'  Unexpected keys (ignored): {list(incompatible.unexpected_keys)}\n'
            f'Check that attention_mode / input_list / num_classes match the training run.'
        )

    model.eval()

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform,
        mask_in_class_subdir=mask_in_class_subdir,
        id_prefix=id_prefix)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()

    data_iter = val_loader
    if not args.disable_tqdm:
        data_iter = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f'{args.split.upper()}',
            leave=False,
            mininterval=0.5,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for input, target, meta in data_iter:
            non_blocking = device.type == 'cuda'
            input = input.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            # compute output
            output = model(input)

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            sample_ids = meta['sample_id'] if 'sample_id' in meta else meta['img_id']
            for pred, sample_id in zip(output, sample_ids):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np)
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(sample_id)))

    
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)



if __name__ == '__main__':
    main()
