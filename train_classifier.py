import argparse
import os
import random
from collections import OrderedDict

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from classifier_dataset import SegGuidedClassificationDataset, infer_num_classes, load_label_csv
from classifier_model import SegGuidedEfficientNetB0, DualInputSegGuidedEfficientNet
from utils import AverageMeter, str2bool


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.clone().detach())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='fives_effb0_seg_guided', type=str, help='experiment name')
    parser.add_argument('--output_dir', default='outputs_cls', type=str, help='output directory')
    parser.add_argument('--data_dir', required=True, type=str, help='FIVES root containing train/test folders')

    parser.add_argument('--train_labels_csv', default='', type=str,
                        help='CSV with train image labels (img_id,label). If empty, auto-discovered under data_dir.')
    parser.add_argument('--train_mask_dir', required=True, type=str, help='predicted segmentation masks for train/val images')
    parser.add_argument('--mask_ext', default='.png', type=str, help='mask extension')
    parser.add_argument('--img_ext', default='.png', type=str, help='image extension')

    parser.add_argument('--val_ratio', default=0.2, type=float, help='validation ratio from train split')
    parser.add_argument('--dataseed', default=2981, type=int, help='split seed')

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--input_h', default=512, type=int)
    parser.add_argument('--input_w', default=512, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--disable_tqdm', default=False, type=str2bool)
    parser.add_argument('--early_stopping', default=10, type=int)
    parser.add_argument('--resume_checkpoint', default='', type=str, help='path to full classifier checkpoint to resume from')
    parser.add_argument('--save_freq', default=5, type=int, help='save full checkpoint every N epochs')
    parser.add_argument('--save_last', default=True, type=str2bool, help='save full last checkpoint every epoch')
    parser.add_argument('--save_best_full', default=True, type=str2bool, help='save full best checkpoint')

    # New arguments for improved classifier
    parser.add_argument('--model_type', default='single', choices=['single', 'dual_input'], type=str, help='model architecture')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--use_focal', default=False, type=str2bool, help='use focal loss instead of cross entropy')
    parser.add_argument('--focal_gamma', default=1.5, type=float, help='focal loss gamma')
    parser.add_argument('--label_smoothing', default=0.05, type=float, help='label smoothing')
    parser.add_argument('--use_class_weights', default=False, type=str2bool, help='use class weights')
    parser.add_argument('--class_weight_power', default=1.0, type=float, help='class weight power for inverse frequency')
    parser.add_argument('--manual_class_weights', default='', type=str, help='manual class weight multipliers as JSON dict, e.g., \'{"0": 2.0}\' to double Class 0 weight')
    parser.add_argument('--use_mixup', default=False, type=str2bool, help='use mixup augmentation')
    parser.add_argument('--mixup_alpha', default=0.2, type=float, help='mixup alpha parameter')
    parser.add_argument('--clean_train_metrics', default=True, type=str2bool,
                        help='with mixup, score train metrics on unmixed inputs (one extra forward pass)')
    parser.add_argument('--num_classes', default=0, type=int,
                        help='number of classes; 0 = infer from the train labels CSV')
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


def run_epoch(loader, model, criterion, device, optimizer=None, disable_tqdm=False, desc='Train', return_outputs=False, use_mixup=False, mixup_alpha=0.2, clean_metrics=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    avg_loss = AverageMeter()
    y_true = []
    y_pred = []
    y_prob = []

    iterable = loader
    if not disable_tqdm:
        iterable = tqdm(loader, total=len(loader), desc=desc, leave=False, mininterval=0.5)

    with torch.set_grad_enabled(is_train):
        for images, masks, labels, _ in iterable:
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            if is_train and use_mixup:
                # Mixup augmentation
                batch_size = images.size(0)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(batch_size).to(device)
                mixed_images = lam * images + (1 - lam) * images[index]
                mixed_masks = lam * masks + (1 - lam) * masks[index]
                labels_a, labels_b = labels, labels[index]
                logits = model(mixed_images, mixed_masks)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images, masks)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss.update(loss.item(), images.size(0))

            # Metrics must come from UNMIXED inputs. Scoring predictions made on
            # blended images against the original labels understates train
            # accuracy badly (it was the cause of train<<val in the logs).
            # eval() during the extra pass keeps BatchNorm running stats and
            # dropout from being disturbed by this bookkeeping forward.
            if is_train and use_mixup and clean_metrics:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    metric_logits = model(images, masks)
                if was_training:
                    model.train()
            else:
                metric_logits = logits

            preds = torch.argmax(metric_logits, dim=1)
            probs = torch.softmax(metric_logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    y_prob_np = np.array(y_prob, dtype=np.float32)

    acc = accuracy_score(y_true_np, y_pred_np) if len(y_true_np) > 0 else 0.0
    f1 = f1_score(y_true_np, y_pred_np, average='macro') if len(set(y_true_np.tolist())) > 1 else acc
    precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0) if len(y_true_np) > 0 else 0.0
    recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0) if len(y_true_np) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np) if len(set(y_true_np.tolist())) > 1 else acc

    cm = confusion_matrix(y_true_np, y_pred_np)
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
        if y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2:
            auc = float(roc_auc_score(y_true_np, y_prob_np[:, 1]))
        elif y_prob_np.ndim == 2 and y_prob_np.shape[1] > 2:
            auc = float(roc_auc_score(y_true_np, y_prob_np, multi_class='ovr', average='macro'))
    except Exception:
        auc = np.nan

    # Per-class metrics
    classes = sorted(np.unique(y_true_np).tolist())
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    for cls in classes:
        per_class_f1[cls] = f1_score(y_true_np, y_pred_np, labels=[cls], average='macro', zero_division=0)
        per_class_precision[cls] = precision_score(y_true_np, y_pred_np, labels=[cls], average='macro', zero_division=0)
        per_class_recall[cls] = recall_score(y_true_np, y_pred_np, labels=[cls], average='macro', zero_division=0)

    results = OrderedDict(
        loss=avg_loss.avg,
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
    )
    if return_outputs:
        results['y_true'] = y_true_np
        results['y_pred'] = y_pred_np
        results['y_prob'] = y_prob_np
    return results


def maybe_stratify(df):
    counts = df['label'].value_counts().to_dict()
    if len(counts) <= 1:
        return None
    if min(counts.values()) < 2:
        return None
    return df['label']


def compute_class_weights(df, power=1.0, manual_weights=None):
    """
    Compute class weights based on inverse frequency, with optional manual overrides.
    manual_weights: dict mapping class index to weight multiplier (e.g., {0: 2.0} to double Class 0 weight)
    """
    label_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = (total_samples / (num_classes * count)) ** power
        # Apply manual weight multiplier if specified
        if manual_weights and i in manual_weights:
            weight *= manual_weights[i]
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


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


def make_checkpoint_payload(epoch, model, optimizer, scheduler, best_val_f1, config):
    return {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_f1': best_val_f1,
        'config': config,
    }


def load_resume_checkpoint(model, optimizer, scheduler, ckpt_path, map_location='cpu'):
    if not ckpt_path:
        return 0, -1.0
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Resume checkpoint not found: {ckpt_path}')

    ckpt = torch_load_compat(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise RuntimeError('Unsupported resume checkpoint format.')
    if 'model' not in ckpt:
        raise RuntimeError('Resume checkpoint missing model state.')

    incompatible = model.load_state_dict(ckpt['model'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f'WARNING: resume checkpoint does not fully match the model.\n'
            f'  Missing (randomly initialized): {list(incompatible.missing_keys)}\n'
            f'  Unexpected (ignored): {list(incompatible.unexpected_keys)}'
        )
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])

    start_epoch = int(ckpt.get('epoch', -1)) + 1
    best_val_f1 = float(ckpt.get('best_val_f1', -1.0))
    print(f'=> resumed classifier from: {ckpt_path}')
    print(f'=> start_epoch: {start_epoch}, best_val_f1: {best_val_f1:.4f}')
    return start_epoch, best_val_f1


def save_training_plots(history_df, run_dir):
    epochs = history_df['epoch'].values

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_df['train_loss'], label='train_loss', linewidth=2)
    plt.plot(epochs, history_df['val_loss'], label='val_loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'curve_loss.png'), dpi=220, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_df['train_acc'], label='train_acc', linewidth=2)
    plt.plot(epochs, history_df['val_acc'], label='val_acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Val Accuracy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'curve_accuracy.png'), dpi=220, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_df['train_f1'], label='train_f1', linewidth=2)
    plt.plot(epochs, history_df['val_f1'], label='val_f1', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.title('Train vs Val F1')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'curve_f1.png'), dpi=220, bbox_inches='tight')
    plt.close()


def save_confusion_matrix_plot(y_true, y_pred, run_dir):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Test)')
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
    plt.savefig(os.path.join(run_dir, 'confusion_matrix_test.png'), dpi=220, bbox_inches='tight')
    plt.close()


def save_roc_plot(y_true, y_prob, run_dir):
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
        aucs = []
        for cls in classes:
            y_bin = (y_true == cls).astype(np.int32)
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls])
                cls_auc = roc_auc_score(y_bin, y_prob[:, cls])
                plt.plot(fpr, tpr, linewidth=1.5, alpha=0.7, label=f'Class {cls} AUC={cls_auc:.3f}')
                aucs.append(cls_auc)
                plotted = True
            except Exception:
                continue
        if len(aucs) > 0:
            plt.plot([], [], ' ', label=f'Macro AUC={np.mean(aucs):.4f}')

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test)')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'roc_curve_test.png'), dpi=220, bbox_inches='tight')
    plt.close()


def format_split_line(split_name, metrics):
    return (
        f'{split_name:<5} | '
        f'loss: {metrics["loss"]:.4f}  '
        f'acc: {metrics["acc"]:.4f}  '
        f'f1: {metrics["f1"]:.4f}'
    )


def main():
    args = parse_args()
    seed_torch(args.dataseed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.join(args.output_dir, args.name), exist_ok=True)
    run_dir = os.path.join(args.output_dir, args.name)

    train_img_dir = os.path.join(args.data_dir, 'train', 'Original')
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(f'Could not find train images directory: {train_img_dir}')

    train_labels_csv = resolve_labels_csv(args.data_dir, args.train_labels_csv, split='train')
    train_df = load_label_csv(train_labels_csv)

    stratify = maybe_stratify(train_df)
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.dataseed,
        stratify=stratify,
    )

    val_split_df.to_csv(os.path.join(run_dir, 'val_split.csv'), index=False)
    train_split_df.to_csv(os.path.join(run_dir, 'train_split.csv'), index=False)

    num_classes = int(args.num_classes) if args.num_classes > 0 else infer_num_classes(train_df)
    print(f'num_classes: {num_classes}')

    run_cfg = vars(args).copy()
    run_cfg['train_labels_csv'] = train_labels_csv
    # Persist the resolved value so evaluation/ensembling never has to guess it.
    run_cfg['num_classes'] = num_classes
    with open(os.path.join(run_dir, 'classifier_config.yml'), 'w') as f:
        yaml.dump(run_cfg, f)

    train_tf = A.Compose([
        A.Resize(args.input_h, args.input_w),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(),
    ])
    eval_tf = A.Compose([
        A.Resize(args.input_h, args.input_w),
        A.Normalize(),
    ])

    train_dataset = SegGuidedClassificationDataset(
        records=train_split_df,
        image_dir=train_img_dir,
        mask_dir=args.train_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=train_tf,
        split_prefixes=['', 'fives_train_', 'fives_val_'],
        image_root=args.data_dir,
    )
    val_dataset = SegGuidedClassificationDataset(
        records=val_split_df,
        image_dir=train_img_dir,
        mask_dir=args.train_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=eval_tf,
        split_prefixes=['', 'fives_val_', 'fives_train_'],
        image_root=args.data_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    if len(train_dataset) < args.batch_size:
        print(f'Warning: train set size ({len(train_dataset)}) is smaller than batch_size ({args.batch_size}); using smaller final batch.')

    # Model selection
    if args.model_type == 'dual_input':
        model = DualInputSegGuidedEfficientNet(num_classes=num_classes, pretrained=args.pretrained, dropout=args.dropout).to(device)
        print(f'Using DualInputSegGuidedEfficientNet with dropout={args.dropout}')
    else:
        model = SegGuidedEfficientNetB0(num_classes=num_classes, pretrained=args.pretrained).to(device)
        print(f'Using SegGuidedEfficientNetB0')

    # Loss function with focal loss and class weights
    class_weights = None
    manual_weights = None
    if args.manual_class_weights:
        import json
        try:
            manual_weights = json.loads(args.manual_class_weights)
            manual_weights = {int(k): float(v) for k, v in manual_weights.items()}
            print(f'Manual class weights: {manual_weights}')
        except Exception as e:
            print(f'Error parsing manual_class_weights: {e}. Ignoring.')

    if args.use_class_weights:
        class_weights = compute_class_weights(train_split_df, power=args.class_weight_power, manual_weights=manual_weights).to(device)
        print(f'Final class weights: {class_weights.cpu().numpy()}')

    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing, class_weights=class_weights).to(device)
        print(f'Using FocalLoss with gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}')
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing).to(device)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
        print(f'Using CrossEntropyLoss with label_smoothing={args.label_smoothing}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    cudnn.benchmark = device.type == 'cuda'
    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = []
    start_epoch = 0

    if args.resume_checkpoint:
        start_epoch, best_val_f1 = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=args.resume_checkpoint,
            map_location=device,
        )

    for epoch in range(start_epoch, args.epochs):
        train_log = run_epoch(
            train_loader,
            model,
            criterion,
            device=device,
            optimizer=optimizer,
            disable_tqdm=args.disable_tqdm,
            desc=f'Train {epoch+1}/{args.epochs}',
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            clean_metrics=args.clean_train_metrics,
        )
        val_log = run_epoch(
            val_loader,
            model,
            criterion,
            device=device,
            optimizer=None,
            disable_tqdm=args.disable_tqdm,
            desc=f'Val {epoch+1}/{args.epochs}',
            use_mixup=False,  # Never use mixup for validation
        )
        scheduler.step()

        print(f'\nEpoch [{epoch+1:03d}/{args.epochs:03d}]')
        print(format_split_line('Train', train_log))
        print(format_split_line('Val', val_log))

        row = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_log['loss'],
            'train_acc': train_log['acc'],
            'train_f1': train_log['f1'],
            'train_precision': train_log['precision'],
            'train_recall': train_log['recall'],
            'train_specificity': train_log['specificity'],
            'train_bal_acc': train_log['bal_acc'],
            'train_auc': train_log['auc'],
            'val_loss': val_log['loss'],
            'val_acc': val_log['acc'],
            'val_f1': val_log['f1'],
            'val_precision': val_log['precision'],
            'val_recall': val_log['recall'],
            'val_specificity': val_log['specificity'],
            'val_bal_acc': val_log['bal_acc'],
            'val_auc': val_log['auc'],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'cls_log.csv'), index=False)

        if val_log['f1'] > best_val_f1:
            best_val_f1 = val_log['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_classifier.pth'))
            if args.save_best_full:
                best_full = make_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_f1=best_val_f1,
                    config=run_cfg,
                )
                torch.save(best_full, os.path.join(run_dir, 'best_classifier_checkpoint.pth'))
            print(f'=> saved best classifier | val_f1: {best_val_f1:.4f}')
        else:
            epochs_no_improve += 1

        torch.save(model.state_dict(), os.path.join(run_dir, 'last_classifier.pth'))

        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            periodic_ckpt = make_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_f1=best_val_f1,
                config=run_cfg,
            )
            torch.save(periodic_ckpt, os.path.join(run_dir, f'classifier_checkpoint_epoch_{epoch + 1}.pth'))

        if args.save_last:
            last_ckpt = make_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_f1=best_val_f1,
                config=run_cfg,
            )
            torch.save(last_ckpt, os.path.join(run_dir, 'last_classifier_checkpoint.pth'))

        if args.early_stopping >= 0 and epochs_no_improve >= args.early_stopping:
            print('=> early stopping triggered')
            break

    history_df = pd.DataFrame(history)
    if len(history_df) > 0:
        save_training_plots(history_df, run_dir)
    print(f'\nSaved training plots/logs in: {run_dir}')


if __name__ == '__main__':
    main()
