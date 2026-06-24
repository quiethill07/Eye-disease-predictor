import argparse
import os

import torch
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders
from model_segmentation import UKANSegmentationModel
from utils import (
    DiceBCEFocalDeepSupervisionLoss,
    apply_segmentation_tta,
    average_checkpoints_state_dict,
    ensure_dir,
    save_json,
    save_segmentation_visuals,
    segmentation_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retinal vessel segmentation model on the test set.")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--fives_root", type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="segmentation_test_outputs")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--extra_checkpoints", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "visuals"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UKANSegmentationModel(
        base_channels=config.get("segmentation_base_channels", 32),
        deep_supervision=config.get("deep_supervision", False),
        image_size=config.get("image_size", args.image_size),
        no_kan=config.get("no_kan", False),
        attention_mode=config.get("attention_mode", "none"),
    ).to(device)
    if args.extra_checkpoints:
        state_dict = average_checkpoints_state_dict([args.checkpoint] + args.extra_checkpoints)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
    loader = loaders["test"]

    criterion = DiceBCEFocalDeepSupervisionLoss(focal_gamma=config.get("focal_gamma", 2.0))
    running = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "accuracy": 0.0}
    sample_batch = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="test_seg", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            outputs = model(images)
            seg_logits = apply_segmentation_tta(model, images) if args.tta else outputs["logits"]
            loss = criterion(seg_logits, masks, outputs.get("aux_logits"))
            metrics = segmentation_metrics(seg_logits, masks)

            running["loss"] += loss.item()
            running["iou"] += metrics["iou"]
            running["dice"] += metrics["dice"]
            running["accuracy"] += metrics["accuracy"]

            sample_batch = (
                images.detach().cpu(),
                masks.detach().cpu(),
                seg_logits.detach().cpu(),
                torch.zeros(images.size(0), dtype=torch.long),
                torch.zeros(images.size(0), 4),
            )

    num_batches = max(len(loader), 1)
    metrics = {key: value / num_batches for key, value in running.items()}
    save_json(metrics, os.path.join(args.output_dir, "segmentation_test_metrics.json"))

    if sample_batch is not None:
        save_segmentation_visuals(
            *sample_batch,
            output_dir=os.path.join(args.output_dir, "visuals"),
            prefix="segmentation_test",
        )

    print(
        f"Segmentation Test Loss {metrics['loss']:.4f} | "
        f"IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f} | "
        f"Accuracy {metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
