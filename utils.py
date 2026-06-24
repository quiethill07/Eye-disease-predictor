import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from dataset import INV_LABEL_MAP
from loss import ClassificationCriterion


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(logits.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return bce + (1.0 - dice.mean())


class DiceBCEFocalDeepSupervisionLoss(nn.Module):
    # NEW: sharper vessel loss with focal term and optional deep supervision.
    def __init__(self, smooth: float = 1.0, focal_gamma: float = 2.0, aux_weight: float = 0.3) -> None:
        super().__init__()
        self.smooth = smooth
        self.focal_gamma = focal_gamma
        self.aux_weight = aux_weight
        self.bce = nn.BCEWithLogitsLoss()

    def _single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(logits.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)).mean()
        focal_term = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-focal_term)
        focal = (((1 - pt) ** self.focal_gamma) * focal_term).mean()
        return 0.5 * bce + 0.3 * dice_loss + 0.2 * focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits: List[torch.Tensor] | None = None) -> torch.Tensor:
        loss = self._single_loss(logits, targets)
        if aux_logits:
            aux_loss = sum(self._single_loss(aux, targets) for aux in aux_logits) / len(aux_logits)
            loss = loss + self.aux_weight * aux_loss
        return loss


def classification_loss(
    main_logits: torch.Tensor,
    aux_logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "focal",
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    aux_weight: float = 0.3,
    confidence_penalty: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    criterion = ClassificationCriterion(
        loss_name=loss_name,
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        aux_weight=aux_weight,
        confidence_penalty=confidence_penalty,
    ).to(main_logits.device)
    return criterion(main_logits, aux_logits, targets, sample_weights=sample_weights)["loss"]


def _component_tortuosity(component: np.ndarray) -> float:
    if component.sum() < 2:
        return 1.0

    kernel = np.ones((3, 3), dtype=np.int32)
    neighbors = ndimage.convolve(component.astype(np.int32), kernel, mode="constant", cval=0)
    endpoints = np.argwhere(component & (neighbors == 2))
    points = np.argwhere(component)

    if len(points) < 2:
        return 1.0

    if len(endpoints) >= 2:
        start, end = endpoints[0], endpoints[-1]
    else:
        start, end = points[0], points[-1]

    arc_length = float(len(points))
    chord_length = float(np.linalg.norm(start - end))
    chord_length = max(chord_length, 1.0)
    return max(arc_length / chord_length, 1.0)


def extract_vessel_features_from_mask(mask: np.ndarray) -> np.ndarray:
    binary_mask = (mask > 0.5).astype(np.uint8)
    total_pixels = float(binary_mask.size)
    vessel_pixels = float(binary_mask.sum())
    density = vessel_pixels / max(total_pixels, 1.0)

    skeleton = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    skeleton_pixels = float(skeleton.sum())
    length = skeleton_pixels / max(total_pixels, 1.0)

    labeled, num_components = ndimage.label(skeleton)
    tortuosities = []
    for component_idx in range(1, num_components + 1):
        component = labeled == component_idx
        if component.sum() < 5:
            continue
        tortuosities.append(_component_tortuosity(component))

    tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0
    return np.array([density, length, tortuosity], dtype=np.float32)


def extract_vessel_features_batch(masks: torch.Tensor) -> torch.Tensor:
    features = []
    for mask in masks.detach().cpu().numpy():
        resized_mask = cv2.resize(mask[0], (128, 128), interpolation=cv2.INTER_NEAREST)
        features.append(extract_vessel_features_from_mask(resized_mask))
    return torch.tensor(np.stack(features), dtype=torch.float32)


def compute_feature_stats_from_dataframe(dataframe, mask_root: str = "") -> Dict[str, torch.Tensor]:
    features = []
    for _, row in dataframe.iterrows():
        mask_path = row["mask_path"]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(mask_root, mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask for vessel stats: {mask_path}")
        mask = (mask > 127).astype(np.float32)
        features.append(extract_vessel_features_from_mask(mask))

    feature_array = np.stack(features).astype(np.float32)
    mean = torch.tensor(feature_array.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(feature_array.std(axis=0) + 1e-6, dtype=torch.float32)
    return {"mean": mean, "std": std}


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = ((intersection + 1e-7) / (union + 1e-7)).mean().item()
    dice = ((2 * intersection + 1e-7) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-7)).mean().item()
    accuracy = (preds == targets).float().mean().item()
    return {"iou": iou, "dice": dice, "accuracy": accuracy}


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    truth = targets.detach().cpu().numpy()
    return {"accuracy": accuracy_score(truth, preds)}


def classification_metrics_detailed(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float | Dict[str, float]]:
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    truth = targets.detach().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        preds,
        labels=list(sorted(INV_LABEL_MAP.keys())),
        zero_division=0,
    )
    class_accuracy = {}
    class_recall = {}
    class_precision = {}
    class_f1 = {}
    for class_idx in sorted(INV_LABEL_MAP.keys()):
        mask = truth == class_idx
        class_name = INV_LABEL_MAP[class_idx]
        class_accuracy[class_name] = float((preds[mask] == truth[mask]).mean()) if mask.any() else 0.0
        class_precision[class_name] = float(precision[class_idx])
        class_recall[class_name] = float(recall[class_idx])
        class_f1[class_name] = float(f1[class_idx])
    return {
        "accuracy": accuracy_score(truth, preds),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(truth, preds, labels=list(sorted(INV_LABEL_MAP.keys()))).tolist(),
        "per_class_accuracy": class_accuracy,
        "per_class_precision": class_precision,
        "per_class_recall": class_recall,
        "per_class_f1": class_f1,
    }


def apply_class_calibration(
    logits: torch.Tensor,
    class_thresholds: List[float] | None = None,
    class_logit_biases: List[float] | None = None,
) -> torch.Tensor:
    calibrated = logits.clone()
    if class_logit_biases is not None:
        bias = torch.tensor(class_logit_biases, dtype=calibrated.dtype, device=calibrated.device)
        calibrated = calibrated + bias.view(1, -1)
    if class_thresholds is not None:
        thresholds = torch.tensor(class_thresholds, dtype=calibrated.dtype, device=calibrated.device).clamp_min(1e-6)
        probs = torch.softmax(calibrated, dim=1)
        calibrated = torch.log(probs / thresholds.view(1, -1))
    return calibrated


def build_classification_class_weights(dataframe, power: float = 1.0, normal_boost: float = 1.0) -> torch.Tensor:
    label_map = {"amd": 0, "dr": 1, "glaucoma": 2, "normal": 3}
    label_indices = dataframe["label"].map(lambda x: label_map[str(x).strip().lower()]).to_numpy()
    counts = np.bincount(label_indices, minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.power(weights, power)
    weights[3] *= normal_boost
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_history(history: Dict[str, List[float]], path: str) -> None:
    save_json(history, path)


def plot_training_curves(history: Dict[str, List[float]], output_dir: str, epoch: int, final: bool = False) -> None:
    ensure_dir(output_dir)
    epochs = range(1, len(history["train_total_loss"]) + 1)
    has_classification = any(value != 0.0 for value in history.get("train_cls_accuracy", [])) or any(
        value != 0.0 for value in history.get("val_cls_accuracy", [])
    )
    has_classification = has_classification or any(value != 0.0 for value in history.get("train_cls_loss", [])) or any(
        value != 0.0 for value in history.get("val_cls_loss", [])
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(epochs, history["train_total_loss"], label="Train")
    axes[0].plot(epochs, history["val_total_loss"], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], label="Train IoU")
    axes[1].plot(epochs, history["val_iou"], label="Val IoU")
    axes[1].plot(epochs, history["train_dice"], label="Train Dice")
    axes[1].plot(epochs, history["val_dice"], label="Val Dice")
    axes[1].set_title("Segmentation Metrics")
    axes[1].legend()

    if has_classification:
        if "train_cls_macro_f1" in history and "val_cls_macro_f1" in history:
            axes[2].plot(epochs, history["train_cls_macro_f1"], label="Train Macro-F1")
            axes[2].plot(epochs, history["val_cls_macro_f1"], label="Val Macro-F1")
            axes[2].plot(epochs, history["train_cls_accuracy"], label="Train Acc", alpha=0.5)
            axes[2].plot(epochs, history["val_cls_accuracy"], label="Val Acc", alpha=0.5)
            axes[2].set_title("Classification Macro-F1 / Accuracy")
        else:
            axes[2].plot(epochs, history["train_cls_accuracy"], label="Train Acc")
            axes[2].plot(epochs, history["val_cls_accuracy"], label="Val Acc")
            axes[2].set_title("Classification Accuracy")
    else:
        axes[2].plot(epochs, history["train_seg_accuracy"], label="Train Acc")
        axes[2].plot(epochs, history["val_seg_accuracy"], label="Val Acc")
        axes[2].set_title("Segmentation Accuracy")
    axes[2].legend()

    axes[3].plot(epochs, history["train_seg_loss"], label="Train Seg Loss")
    axes[3].plot(epochs, history["val_seg_loss"], label="Val Seg Loss")
    if has_classification:
        axes[3].plot(epochs, history["train_cls_loss"], label="Train Cls Loss")
        axes[3].plot(epochs, history["val_cls_loss"], label="Val Cls Loss")
        axes[3].set_title("Task Losses")
    else:
        axes[3].set_title("Segmentation Loss")
    axes[3].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)

    plt.tight_layout()
    suffix = "final" if final else f"epoch_{epoch:03d}"
    plt.savefig(os.path.join(output_dir, f"training_curves_{suffix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true: List[int], y_pred: List[int], output_path: str, title: str) -> None:
    labels = [INV_LABEL_MAP[idx].upper() for idx in sorted(INV_LABEL_MAP.keys())]
    matrix = confusion_matrix(y_true, y_pred, labels=list(sorted(INV_LABEL_MAP.keys())))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_np = image.detach().cpu().numpy()
    image_np = (image_np * std + mean).clip(0, 1)
    return np.transpose(image_np, (1, 2, 0))


def save_segmentation_visuals(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_logits: torch.Tensor,
    output_dir: str,
    prefix: str,
    max_items: int = 4,
) -> None:
    ensure_dir(output_dir)
    probs = torch.sigmoid(preds)
    pred_masks = (probs > 0.5).float()
    pred_labels = torch.argmax(class_logits, dim=1)

    for idx in range(min(max_items, images.size(0))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(denormalize_image(images[idx]))
        axes[0].set_title(
            f"Image\nTrue: {INV_LABEL_MAP[int(labels[idx])].upper()} | "
            f"Pred: {INV_LABEL_MAP[int(pred_labels[idx])].upper()}"
        )
        axes[1].imshow(masks[idx, 0].detach().cpu().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[2].imshow(pred_masks[idx, 0].detach().cpu().numpy(), cmap="gray")
        axes[2].set_title("Predicted Vessel Mask")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_sample_{idx}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def average_checkpoints_state_dict(checkpoint_paths: List[str], key: str = "model_state_dict") -> Dict[str, torch.Tensor]:
    # OPTIONAL: checkpoint averaging support for evaluation.
    averaged_state = None
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint[key]
        if averaged_state is None:
            averaged_state = {name: tensor.clone().float() for name, tensor in state_dict.items()}
        else:
            for name, tensor in state_dict.items():
                averaged_state[name] += tensor.float()
    assert averaged_state is not None
    num = float(len(checkpoint_paths))
    for name in averaged_state:
        averaged_state[name] /= num
    return averaged_state


def apply_segmentation_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    # OPTIONAL: lightweight TTA using horizontal and vertical flips.
    logits = []
    logits.append(model(images)["logits"])
    logits.append(torch.flip(model(torch.flip(images, dims=[3]))["logits"], dims=[3]))
    logits.append(torch.flip(model(torch.flip(images, dims=[2]))["logits"], dims=[2]))
    return torch.stack(logits, dim=0).mean(dim=0)


@dataclass
class TrainConfig:
    csv_path: str = ""
    image_root: str = ""
    mask_root: str = ""
    fives_root: str = ""
    output_dir: str = "outputs"
    image_size: int = 512
    batch_size: int = 4
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    backbone_name: str = "efficientnet_b0"
    scheduler_name: str = "cosine"
    use_pretrained: bool = True
    segmentation_base_channels: int = 32
    segmentation_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    classifier_dropout: float = 0.3
    classification_label_smoothing: float = 0.1
    plot_every: int = 25
    amp: bool = True
    val_size: float = 0.2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)
