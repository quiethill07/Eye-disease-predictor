import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders
from fusion_model import RetinaClassificationNet
from loss import ClassificationCriterion, HardExampleMemory
from model_segmentation import UKANSegmentationModel
from utils import (
    apply_class_calibration,
    build_classification_class_weights,
    classification_metrics_detailed,
    ensure_dir,
    extract_vessel_features_batch,
    plot_training_curves,
    save_confusion_matrix,
    save_history,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train retinal disease classification separately.")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--fives_root", type=str, default="")
    parser.add_argument("--segmentation_checkpoint", type=str, default="")
    parser.add_argument("--use_ground_truth_masks", action="store_true")
    parser.add_argument("--output_dir", type=str, default="classification_outputs")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone_name", type=str, default="efficientnet_b1")
    parser.add_argument("--scheduler_name", type=str, choices=["cosine", "plateau"], default="plateau")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--loss_name", type=str, choices=["ce", "focal"], default="focal")
    parser.add_argument("--classification_label_smoothing", type=float, default=0.03)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--aux_classification_loss_weight", type=float, default=0.25)
    parser.add_argument("--confidence_penalty", type=float, default=0.02)
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--normal_boost", type=float, default=1.35)
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--disable_weighted_sampling", action="store_true")
    parser.add_argument("--enable_hard_example_mining", action="store_true")
    parser.add_argument("--hard_example_growth", type=float, default=1.25)
    parser.add_argument("--hard_example_decay", type=float, default=0.96)
    parser.add_argument("--hard_example_max_weight", type=float, default=3.0)
    parser.add_argument("--plot_every", type=int, default=25)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--segmentation_base_channels", type=int, default=32)
    parser.add_argument("--segmentation_no_kan", action="store_true")
    parser.add_argument("--segmentation_attention_mode", type=str, choices=["none", "se_only", "cbam_only", "cbam_se"], default="none")
    parser.add_argument("--class_thresholds", nargs=4, type=float, default=None)
    parser.add_argument("--class_logit_biases", nargs=4, type=float, default=None)
    parser.add_argument("--normal_logit_bias", type=float, default=0.0)
    return parser.parse_args()


def build_scheduler(optimizer, args):
    if args.scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)
    return CosineAnnealingLR(optimizer, T_max=args.epochs)


def build_vessel_inputs(images, masks, segmentation_model, use_ground_truth_masks):
    if use_ground_truth_masks:
        vessel_mask = masks
    else:
        if segmentation_model is None:
            raise ValueError("Provide --segmentation_checkpoint or use --use_ground_truth_masks for classification training.")
        with torch.no_grad():
            seg_outputs = segmentation_model(images)
            vessel_mask = (torch.sigmoid(seg_outputs["logits"]) > 0.5).float()

    vessel_image = vessel_mask.repeat(1, 3, 1, 1) * images
    vessel_features = extract_vessel_features_batch(vessel_mask).to(images.device)
    return vessel_image, vessel_features, vessel_mask


def resolve_class_biases(args) -> List[float] | None:
    if args.class_logit_biases is None and args.normal_logit_bias == 0.0:
        return None
    biases = list(args.class_logit_biases) if args.class_logit_biases is not None else [0.0, 0.0, 0.0, 0.0]
    biases[3] += args.normal_logit_bias
    return biases


def run_epoch(
    model,
    segmentation_model,
    loader,
    optimizer,
    device,
    scaler,
    criterion: ClassificationCriterion,
    is_train: bool,
    args,
    hard_example_memory: HardExampleMemory | None,
):
    model.train(is_train)
    running_loss = 0.0
    collected_logits: List[torch.Tensor] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    class_biases = resolve_class_biases(args)

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sample_ids = batch["image_path"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        vessel_image, vessel_features, vessel_mask = build_vessel_inputs(images, masks, segmentation_model, args.use_ground_truth_masks)
        sample_weights = None
        if is_train and hard_example_memory is not None:
            sample_weights = hard_example_memory.get_weights(sample_ids, device)

        with torch.set_grad_enabled(is_train):
            with amp.autocast(device_type=device.type, enabled=(not args.disable_amp) and device.type == "cuda"):
                outputs = model(images, vessel_image, vessel_features, segmentation_mask=vessel_mask)
                cls_logits = outputs["classification_logits"]
                aux_logits = outputs["aux_classification_logits"]
                loss_outputs = criterion(cls_logits, aux_logits, labels, sample_weights=sample_weights)
                loss = loss_outputs["loss"]

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        calibrated_logits = apply_class_calibration(
            cls_logits.detach(),
            class_thresholds=args.class_thresholds,
            class_logit_biases=class_biases,
        )
        if is_train and hard_example_memory is not None:
            hard_example_memory.update(sample_ids, calibrated_logits, labels)

        preds = torch.argmax(calibrated_logits, dim=1)
        running_loss += loss.item()
        collected_logits.append(calibrated_logits.cpu())
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        metrics = classification_metrics_detailed(calibrated_logits, labels)
        progress.set_description(
            f"{'train' if is_train else 'val'} cls loss={loss.item():.4f} macro_f1={metrics['macro_f1']:.4f}"
        )

    num_batches = max(len(loader), 1)
    epoch_logits = torch.cat(collected_logits, dim=0) if collected_logits else torch.empty((0, 4))
    epoch_targets = torch.tensor(y_true, dtype=torch.long)
    epoch_metrics = classification_metrics_detailed(epoch_logits, epoch_targets) if len(y_true) > 0 else {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "confusion_matrix": [[0] * 4 for _ in range(4)],
        "per_class_accuracy": {},
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_f1": {},
    }
    epoch_metrics["loss"] = running_loss / num_batches
    return epoch_metrics, y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)

    artifact_dirs = {
        "root": args.output_dir,
        "checkpoints": os.path.join(args.output_dir, "checkpoints"),
        "plots": os.path.join(args.output_dir, "plots"),
        "logs": os.path.join(args.output_dir, "logs"),
    }
    for path in artifact_dirs.values():
        ensure_dir(path)
    save_json(vars(args), os.path.join(artifact_dirs["logs"], "train_config.json"))
    print(f"Using classification backbone: {args.backbone_name}")

    loaders = create_dataloaders(
        DatasetConfig(
            csv_path=args.csv_path,
            image_root=args.image_root,
            mask_root=args.mask_root,
            fives_root=args.fives_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
            seed=args.seed,
            weighted_sampling=not args.disable_weighted_sampling,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaClassificationNet(
        num_classes=4,
        classifier_backbone=args.backbone_name,
        pretrained=not args.no_pretrained,
        dropout=args.classifier_dropout,
    ).to(device)

    class_weights = None
    if not args.disable_class_weights:
        class_weights = build_classification_class_weights(
            loaders["train"].dataset.dataframe,
            power=args.class_weight_power,
            normal_boost=args.normal_boost,
        ).to(device)
        print(
            "Class weights | "
            f"AMD={class_weights[0].item():.4f} | DR={class_weights[1].item():.4f} | "
            f"GLAUCOMA={class_weights[2].item():.4f} | NORMAL={class_weights[3].item():.4f}"
        )
    else:
        print("Class weights disabled")

    criterion = ClassificationCriterion(
        loss_name=args.loss_name,
        gamma=args.focal_gamma,
        label_smoothing=args.classification_label_smoothing,
        class_weights=class_weights,
        aux_weight=args.aux_classification_loss_weight,
        confidence_penalty=args.confidence_penalty,
    ).to(device)
    print(f"Classification loss: {args.loss_name.upper()}")

    segmentation_model = None
    if not args.use_ground_truth_masks:
        segmentation_model = UKANSegmentationModel(
            base_channels=args.segmentation_base_channels,
            image_size=args.image_size,
            no_kan=args.segmentation_no_kan,
            attention_mode=args.segmentation_attention_mode,
        ).to(device)
        checkpoint = torch.load(args.segmentation_checkpoint, map_location=device)
        segmentation_model.load_state_dict(checkpoint["model_state_dict"])
        segmentation_model.eval()
        for param in segmentation_model.parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = amp.GradScaler(device=device.type, enabled=(not args.disable_amp) and device.type == "cuda")
    hard_example_memory = None
    if args.enable_hard_example_mining:
        hard_example_memory = HardExampleMemory(
            growth=args.hard_example_growth,
            decay=args.hard_example_decay,
            max_weight=args.hard_example_max_weight,
        )

    history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_seg_loss": [],
        "val_seg_loss": [],
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_seg_accuracy": [],
        "val_seg_accuracy": [],
        "train_cls_accuracy": [],
        "val_cls_accuracy": [],
        "train_cls_macro_f1": [],
        "val_cls_macro_f1": [],
    }

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}/{args.epochs} | Starting classification training...")
        train_metrics, train_y_true, train_y_pred = run_epoch(
            model,
            segmentation_model,
            loaders["train"],
            optimizer,
            device,
            scaler,
            criterion,
            True,
            args,
            hard_example_memory,
        )
        print(f"Epoch {epoch:03d}/{args.epochs} | Starting classification validation...")
        val_metrics, y_true, y_pred = run_epoch(
            model,
            segmentation_model,
            loaders["val"],
            optimizer,
            device,
            scaler,
            criterion,
            False,
            args,
            hard_example_memory=None,
        )

        if args.scheduler_name == "plateau":
            scheduler.step(val_metrics["macro_f1"])
        else:
            scheduler.step()

        history["train_total_loss"].append(train_metrics["loss"])
        history["val_total_loss"].append(val_metrics["loss"])
        history["train_seg_loss"].append(0.0)
        history["val_seg_loss"].append(0.0)
        history["train_cls_loss"].append(train_metrics["loss"])
        history["val_cls_loss"].append(val_metrics["loss"])
        history["train_iou"].append(0.0)
        history["val_iou"].append(0.0)
        history["train_dice"].append(0.0)
        history["val_dice"].append(0.0)
        history["train_seg_accuracy"].append(0.0)
        history["val_seg_accuracy"].append(0.0)
        history["train_cls_accuracy"].append(train_metrics["accuracy"])
        history["val_cls_accuracy"].append(val_metrics["accuracy"])
        history["train_cls_macro_f1"].append(train_metrics["macro_f1"])
        history["val_cls_macro_f1"].append(val_metrics["macro_f1"])

        improved = val_metrics["macro_f1"] > (best_macro_f1 + args.early_stopping_min_delta)
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "config": vars(args),
                },
                os.path.join(artifact_dirs["checkpoints"], "best_classification_model.pth"),
            )
            save_confusion_matrix(
                y_true,
                y_pred,
                os.path.join(artifact_dirs["plots"], "best_val_confusion_matrix.png"),
                title=f"Classification Validation Confusion Matrix (Epoch {epoch})",
            )
            print(f"New best classification model saved at epoch {epoch:03d} | Val Macro-F1={best_macro_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No macro-F1 improvement at epoch {epoch:03d} | "
                f"Current Macro-F1={val_metrics['macro_f1']:.4f} | Best Macro-F1={best_macro_f1:.4f} | "
                f"Patience {epochs_without_improvement}/{args.early_stopping_patience}"
            )

        save_json(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            },
            os.path.join(artifact_dirs["logs"], f"epoch_{epoch:03d}_metrics.json"),
        )

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            plot_training_curves(history, artifact_dirs["plots"], epoch, final=(epoch == args.epochs))
            save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Cls Loss {train_metrics['loss']:.4f} | Val Cls Loss {val_metrics['loss']:.4f} | "
            f"Train Acc {train_metrics['accuracy']:.4f} | Val Acc {val_metrics['accuracy']:.4f} | "
            f"Train Precision {train_metrics['macro_precision']:.4f} | Val Precision {val_metrics['macro_precision']:.4f} | "
            f"Train Recall {train_metrics['macro_recall']:.4f} | Val Recall {val_metrics['macro_recall']:.4f} | "
            f"Train F1 {train_metrics['macro_f1']:.4f} | Val F1 {val_metrics['macro_f1']:.4f} | "
            f"Best Macro-F1 {best_macro_f1:.4f}"
        )
        print(
            "Per-class Val Precision | "
            + " | ".join(
                f"{class_name.upper()}: {value:.4f}" for class_name, value in val_metrics["per_class_precision"].items()
            )
        )
        print(
            "Per-class Val Recall | "
            + " | ".join(
                f"{class_name.upper()}: {value:.4f}" for class_name, value in val_metrics["per_class_recall"].items()
            )
        )
        print(
            "Per-class Val F1 | "
            + " | ".join(
                f"{class_name.upper()}: {value:.4f}" for class_name, value in val_metrics["per_class_f1"].items()
            )
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            plot_training_curves(history, artifact_dirs["plots"], epoch, final=True)
            save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))
            print(f"Early stopping classification at epoch {epoch:03d} | Best epoch {best_epoch:03d}")
            break

    plot_training_curves(history, artifact_dirs["plots"], epoch, final=True)
    save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))
    save_json(
        {"best_epoch": best_epoch, "best_macro_f1": best_macro_f1},
        os.path.join(artifact_dirs["logs"], "best_summary.json"),
    )


if __name__ == "__main__":
    main()
