import argparse
import os
import random
from collections import OrderedDict

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classifier_dataset import SegGuidedClassificationDataset, infer_num_classes, load_label_csv
from classifier_model import SegGuidedEfficientNetB0, DualInputSegGuidedEfficientNet
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='training experiment name')
    parser.add_argument('--output_dir', default='outputs_cls', type=str, help='classifier output root')
    parser.add_argument('--data_dir', default='', type=str, help='optional override for data_dir in saved config')
    parser.add_argument('--train_labels_csv', default='', type=str, help='optional override for train_labels_csv in saved config')
    parser.add_argument('--train_mask_dir', default='', type=str, help='optional override for train_mask_dir in saved config')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='evaluation split')
    parser.add_argument('--checkpoint', default='best_classifier.pth', type=str, help='checkpoint filename in run dir')
    parser.add_argument('--test_labels_csv', default='', type=str, help='required for split=test; CSV with img_id,label')
    parser.add_argument('--test_mask_dir', default='', type=str, help='required for split=test; predicted test masks dir')
    parser.add_argument('--disable_tqdm', default=False, type=str2bool)
    parser.add_argument('--tta', default=False, type=str2bool, help='use test time augmentation (horizontal flip)')
    return parser.parse_args()


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


def resolve_labels_csv(data_dir, explicit_path, split):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f'{split} labels CSV not found: {explicit_path}')
        return explicit_path

    preferred_name = f'{split}_labels.csv'
    candidates = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith('.csv'):
                continue
            lower_name = name.lower()
            full_path = os.path.join(root, name)
            score = 10
            if lower_name == preferred_name:
                score = 0
            elif split in lower_name and 'label' in lower_name:
                score = 1
            elif 'label' in lower_name:
                score = 5
            else:
                continue
            candidates.append((score, full_path))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f'Could not auto-discover {split} labels CSV under {data_dir}. '
            f'Pass --{split}_labels_csv explicitly.'
        )

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1]


def maybe_stratify(df):
    counts = df['label'].value_counts().to_dict()
    if len(counts) <= 1:
        return None
    if min(counts.values()) < 2:
        return None
    return df['label']


def compute_class_weights(df, power=1.0):
    """Compute class weights based on inverse frequency."""
    label_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = (total_samples / (num_classes * count)) ** power
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(loader, model, criterion, device, disable_tqdm=False, desc='Eval', use_tta=False):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    y_prob = []

    iterable = loader
    if not disable_tqdm:
        iterable = tqdm(loader, total=len(loader), desc=desc, leave=False, mininterval=0.5)

    with torch.no_grad():
        for images, masks, labels, _ in iterable:
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            if use_tta:
                # TTA: average predictions from multiple augmented versions
                # Original
                logits_list = [model(images, masks)]
                # Horizontal flip
                images_hflip = torch.flip(images, dims=[3])
                masks_hflip = torch.flip(masks, dims=[3])
                logits_list.append(model(images_hflip, masks_hflip))
                # Vertical flip
                images_vflip = torch.flip(images, dims=[2])
                masks_vflip = torch.flip(masks, dims=[2])
                logits_list.append(model(images_vflip, masks_vflip))
                # Average logits
                logits = torch.stack(logits_list).mean(dim=0)
            else:
                logits = model(images, masks)

            loss = criterion(logits, labels)
            losses.append(float(loss.item()))

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_prob = np.array(y_prob, dtype=np.float32)

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='macro') if len(set(y_true.tolist())) > 1 else acc
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) > 0 else 0.0
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(set(y_true.tolist())) > 1 else acc

    cm = confusion_matrix(y_true, y_pred)
    specificity_vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denom = tn + fp
        specificity_vals.append((tn / denom) if denom > 0 else 0.0)
    specificity = float(np.mean(specificity_vals)) if len(specificity_vals) > 0 else 0.0

    auc = np.nan
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
            auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        auc = np.nan

    # Per-class metrics
    classes = sorted(np.unique(y_true).tolist())
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    for cls in classes:
        per_class_f1[cls] = f1_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)
        per_class_precision[cls] = precision_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)
        per_class_recall[cls] = recall_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)

    return OrderedDict(
        loss=float(np.mean(losses)) if len(losses) > 0 else np.nan,
        acc=acc,
        f1=f1,
        precision=precision,
        recall=recall,
        sensitivity=recall,
        specificity=specificity,
        bal_acc=bal_acc,
        auc=auc,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def save_confusion_matrix_plot(y_true, y_pred, out_path):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def save_roc_plot(y_true, y_prob, out_path):
    if y_prob.ndim != 2 or y_prob.shape[0] != len(y_true):
        return
    num_classes = y_prob.shape[1]
    plt.figure(figsize=(6, 5))
    plotted = False

    if num_classes == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC AUC = {auc:.4f}')
            plotted = True
        except Exception:
            plotted = False
    else:
        classes = sorted(np.unique(y_true).tolist())
        for cls in classes:
            y_bin = (y_true == cls).astype(np.int32)
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls])
                cls_auc = roc_auc_score(y_bin, y_prob[:, cls])
                plt.plot(fpr, tpr, linewidth=1.5, alpha=0.7, label=f'Class {cls} AUC={cls_auc:.3f}')
                plotted = True
            except Exception:
                continue

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def print_eval_summary(split_name, metrics):
    print(f'\n=== {split_name.upper()} RESULTS ===')
    print(f'loss       : {metrics["loss"]:.4f}')
    print(f'acc        : {metrics["acc"]:.4f}')
    print(f'f1         : {metrics["f1"]:.4f}')
    print(f'precision  : {metrics["precision"]:.4f}')
    print(f'recall     : {metrics["recall"]:.4f}')
    print(f'specificity: {metrics["specificity"]:.4f}')
    print(f'bal_acc    : {metrics["bal_acc"]:.4f}')
    print(f'auc        : {metrics["auc"]:.4f}')

    # Print per-class metrics
    if 'per_class_f1' in metrics and metrics['per_class_f1']:
        print(f'\n--- Per-Class Metrics ---')
        for cls in sorted(metrics['per_class_f1'].keys()):
            print(f'Class {cls}: F1={metrics["per_class_f1"][cls]:.4f}, '
                  f'Precision={metrics["per_class_precision"][cls]:.4f}, '
                  f'Recall={metrics["per_class_recall"][cls]:.4f}')


def main():
    args = parse_args()
    run_dir = os.path.join(args.output_dir, args.name)
    cfg_path = os.path.join(run_dir, 'classifier_config.yml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f'Missing config file: {cfg_path}')
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed_torch(int(cfg.get('dataseed', 2981)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = device.type == 'cuda'

    data_dir = args.data_dir if args.data_dir else cfg['data_dir']
    cfg_train_labels = cfg.get('train_labels_csv', '')
    train_labels_csv = resolve_labels_csv(data_dir, args.train_labels_csv if args.train_labels_csv else cfg_train_labels, split='train')
    img_ext = cfg.get('img_ext', '.png')
    mask_ext = cfg.get('mask_ext', '.png')
    input_h = int(cfg.get('input_h', 256))
    input_w = int(cfg.get('input_w', 256))
    batch_size = int(cfg.get('batch_size', 16))
    num_workers = int(cfg.get('num_workers', 4))

    if args.split == 'val':
        train_df = load_label_csv(train_labels_csv)
        stratify = maybe_stratify(train_df)
        _, eval_df = train_test_split(
            train_df,
            test_size=float(cfg.get('val_ratio', 0.2)),
            random_state=int(cfg.get('dataseed', 2981)),
            stratify=stratify,
        )
        img_dir = os.path.join(data_dir, 'train', 'Original')
        mask_dir = args.train_mask_dir if args.train_mask_dir else cfg['train_mask_dir']
        split_prefixes = ['', 'fives_val_', 'fives_train_']
    else:
        if not args.test_labels_csv:
            test_labels_csv = resolve_labels_csv(data_dir, '', split='test')
        else:
            test_labels_csv = resolve_labels_csv(data_dir, args.test_labels_csv, split='test')
        if not args.test_mask_dir:
            raise ValueError('--test_mask_dir is required when --split test')
        eval_df = load_label_csv(test_labels_csv)
        img_dir = os.path.join(data_dir, 'test', 'Original')
        mask_dir = args.test_mask_dir
        split_prefixes = ['', 'fives_test_']

    eval_tf = A.Compose([
        A.Resize(input_h, input_w),
        A.Normalize(),
    ])

    dataset = SegGuidedClassificationDataset(
        records=eval_df,
        image_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        transform=eval_tf,
        split_prefixes=split_prefixes,
        image_root=data_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Prefer the value recorded at training time; the head size must match the
    # checkpoint, not whatever happens to be present in the current CSV.
    cfg_num_classes = cfg.get('num_classes', None)
    if cfg_num_classes:
        num_classes = int(cfg_num_classes)
    else:
        num_classes = infer_num_classes(load_label_csv(train_labels_csv))
        print(f'num_classes not found in config; inferred {num_classes} from {train_labels_csv}')

    # Model selection based on config
    model_type = cfg.get('model_type', 'single')
    dropout = cfg.get('dropout', 0.2)

    if model_type == 'dual_input':
        model = DualInputSegGuidedEfficientNet(num_classes=num_classes, pretrained=False, dropout=dropout).to(device)
        print(f'Loading DualInputSegGuidedEfficientNet with dropout={dropout}')
    else:
        model = SegGuidedEfficientNetB0(num_classes=num_classes, pretrained=False).to(device)
        print(f'Loading SegGuidedEfficientNetB0')

    ckpt_path = os.path.join(run_dir, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    model.load_state_dict(torch_load_compat(ckpt_path, map_location=device))

    # Loss function - use same configuration as training
    use_focal = cfg.get('use_focal', True)
    label_smoothing = cfg.get('label_smoothing', 0.03)
    use_class_weights = cfg.get('use_class_weights', True)
    class_weight_power = cfg.get('class_weight_power', 1.0)
    focal_gamma = cfg.get('focal_gamma', 2.0)

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(load_label_csv(train_labels_csv), power=class_weight_power).to(device)

    if use_focal:
        # Import FocalLoss from train_classifier module
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_classifier", os.path.join(os.path.dirname(__file__), 'train_classifier.py'))
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        FocalLoss = train_module.FocalLoss
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing, class_weights=class_weights).to(device)
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing).to(device)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    metrics = evaluate(
        loader=loader,
        model=model,
        criterion=criterion,
        device=device,
        disable_tqdm=args.disable_tqdm,
        desc=args.split.upper(),
        use_tta=args.tta,
    )
    print_eval_summary(args.split, metrics)

    metrics_row = {k: v for k, v in metrics.items() if k not in ['y_true', 'y_pred', 'y_prob']}
    pd.DataFrame([metrics_row]).to_csv(os.path.join(run_dir, f'cls_{args.split}_metrics.csv'), index=False)
    save_confusion_matrix_plot(metrics['y_true'], metrics['y_pred'], os.path.join(run_dir, f'cls_{args.split}_confusion_matrix.png'))
    save_roc_plot(metrics['y_true'], metrics['y_prob'], os.path.join(run_dir, f'cls_{args.split}_roc_curve.png'))
    print(f'\nSaved evaluation artifacts in: {run_dir}')


if __name__ == '__main__':
    main()
