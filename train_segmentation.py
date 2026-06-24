import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders
from model_segmentation import UKANSegmentationModel
from utils import (
    DiceBCEFocalDeepSupervisionLoss,
    ensure_dir,
    plot_training_curves,
    save_history,
    save_json,
    save_segmentation_visuals,
    segmentation_metrics,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train retinal vessel segmentation only.")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--fives_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="segmentation_outputs")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_name", type=str, choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--segmentation_base_channels", type=int, default=32)
    parser.add_argument("--deep_supervision", action="store_true")
    parser.add_argument("--no_kan", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=["none", "se_only", "cbam_only", "cbam_se"], default="none")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--plot_every", type=int, default=25)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    return parser.parse_args()


def build_scheduler(optimizer, args):
    if args.scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    return CosineAnnealingLR(optimizer, T_max=args.epochs)


def run_epoch(model, loader, optimizer, criterion, device, scaler, is_train: bool, amp_enabled: bool):
    model.train(is_train)
    running = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "accuracy": 0.0}
    saved_batch = None

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                seg_logits = outputs["logits"]
                loss = criterion(seg_logits, masks, outputs.get("aux_logits"))

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        metrics = segmentation_metrics(seg_logits, masks)
        running["loss"] += loss.item()
        running["iou"] += metrics["iou"]
        running["dice"] += metrics["dice"]
        running["accuracy"] += metrics["accuracy"]
        saved_batch = (
            images.detach().cpu(),
            masks.detach().cpu(),
            seg_logits.detach().cpu(),
            torch.zeros(images.size(0), dtype=torch.long),
            torch.zeros(images.size(0), 4),
        )
        progress.set_description(
            f"{'train' if is_train else 'val'} seg loss={loss.item():.4f} iou={metrics['iou']:.4f} dice={metrics['dice']:.4f}"
        )

    num_batches = max(len(loader), 1)
    return {key: value / num_batches for key, value in running.items()}, saved_batch


def main():
    args = parse_args()
    set_seed(args.seed)

    artifact_dirs = {
        "root": args.output_dir,
        "checkpoints": os.path.join(args.output_dir, "checkpoints"),
        "plots": os.path.join(args.output_dir, "plots"),
        "logs": os.path.join(args.output_dir, "logs"),
        "visuals": os.path.join(args.output_dir, "visuals"),
    }
    for path in artifact_dirs.values():
        ensure_dir(path)
    save_json(vars(args), os.path.join(artifact_dirs["logs"], "train_config.json"))

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
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UKANSegmentationModel(
        base_channels=args.segmentation_base_channels,
        deep_supervision=args.deep_supervision,
        image_size=args.image_size,
        no_kan=args.no_kan,
        attention_mode=args.attention_mode,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = amp.GradScaler(device=device.type, enabled=(not args.disable_amp) and device.type == "cuda")
    criterion = DiceBCEFocalDeepSupervisionLoss(focal_gamma=args.focal_gamma)

    history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_seg_loss": [],
        "val_seg_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_seg_accuracy": [],
        "val_seg_accuracy": [],
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_cls_accuracy": [],
        "val_cls_accuracy": [],
    }

    best_iou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}/{args.epochs} | Starting segmentation training...")
        train_metrics, _ = run_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            scaler,
            is_train=True,
            amp_enabled=(not args.disable_amp) and device.type == "cuda",
        )

        print(f"Epoch {epoch:03d}/{args.epochs} | Starting segmentation validation...")
        val_metrics, val_visual_batch = run_epoch(
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
            scaler,
            is_train=False,
            amp_enabled=(not args.disable_amp) and device.type == "cuda",
        )

        if args.scheduler_name == "plateau":
            scheduler.step(val_metrics["iou"])
        else:
            scheduler.step()

        history["train_total_loss"].append(train_metrics["loss"])
        history["val_total_loss"].append(val_metrics["loss"])
        history["train_seg_loss"].append(train_metrics["loss"])
        history["val_seg_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_seg_accuracy"].append(train_metrics["accuracy"])
        history["val_seg_accuracy"].append(val_metrics["accuracy"])
        history["train_cls_loss"].append(0.0)
        history["val_cls_loss"].append(0.0)
        history["train_cls_accuracy"].append(0.0)
        history["val_cls_accuracy"].append(0.0)

        improved = val_metrics["iou"] > (best_iou + args.early_stopping_min_delta)
        if improved:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "config": vars(args),
                },
                os.path.join(artifact_dirs["checkpoints"], "best_segmentation_model.pth"),
            )
            if val_visual_batch is not None:
                save_segmentation_visuals(
                    *val_visual_batch,
                    output_dir=artifact_dirs["visuals"],
                    prefix=f"best_epoch_{epoch:03d}",
                )
            print(f"New best segmentation model saved at epoch {epoch:03d} | Val IoU={best_iou:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No IoU improvement at epoch {epoch:03d} | "
                f"Current IoU={val_metrics['iou']:.4f} | Best IoU={best_iou:.4f} | "
                f"Patience {epochs_without_improvement}/{args.early_stopping_patience}"
            )

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            plot_training_curves(history, artifact_dirs["plots"], epoch, final=(epoch == args.epochs))
            save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train IoU {train_metrics['iou']:.4f} | Val IoU {val_metrics['iou']:.4f} | "
            f"Train Dice {train_metrics['dice']:.4f} | Val Dice {val_metrics['dice']:.4f} | "
            f"Best IoU {best_iou:.4f}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            plot_training_curves(history, artifact_dirs["plots"], epoch, final=True)
            save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))
            print(f"Early stopping segmentation at epoch {epoch:03d} | Best epoch {best_epoch:03d}")
            break

    plot_training_curves(history, artifact_dirs["plots"], epoch, final=True)
    save_history(history, os.path.join(artifact_dirs["logs"], "history.json"))
    save_json({"best_epoch": best_epoch, "best_iou": best_iou}, os.path.join(artifact_dirs["logs"], "best_summary.json"))


if __name__ == "__main__":
    main()
