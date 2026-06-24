import argparse
import os
from typing import List

import torch
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders
from fusion_model import RetinaClassificationNet
from loss import ClassificationCriterion
from model_segmentation import UKANSegmentationModel
from utils import (
    apply_class_calibration,
    average_checkpoints_state_dict,
    build_classification_class_weights,
    classification_metrics_detailed,
    ensure_dir,
    extract_vessel_features_batch,
    save_confusion_matrix,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retinal disease classification model on the test set.")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--fives_root", type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--segmentation_checkpoint", type=str, default="")
    parser.add_argument("--use_ground_truth_masks", action="store_true")
    parser.add_argument("--output_dir", type=str, default="classification_test_outputs")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--extra_checkpoints", nargs="*", default=[])
    parser.add_argument("--class_thresholds", nargs=4, type=float, default=None)
    parser.add_argument("--class_logit_biases", nargs=4, type=float, default=None)
    parser.add_argument("--normal_logit_bias", type=float, default=0.0)
    return parser.parse_args()


def build_vessel_inputs(images, masks, segmentation_model, use_ground_truth_masks):
    if use_ground_truth_masks:
        vessel_mask = masks
    else:
        with torch.no_grad():
            seg_outputs = segmentation_model(images)
            seg_logits = seg_outputs["logits"]
            vessel_mask = (torch.sigmoid(seg_logits) > 0.5).float()

    vessel_image = vessel_mask.repeat(1, 3, 1, 1) * images
    vessel_features = extract_vessel_features_batch(vessel_mask).to(images.device)
    return vessel_image, vessel_features, vessel_mask


def resolve_class_biases(args, config) -> List[float] | None:
    checkpoint_biases = config.get("class_logit_biases")
    normal_bias = config.get("normal_logit_bias", 0.0)
    cli_biases = args.class_logit_biases if args.class_logit_biases is not None else checkpoint_biases
    normal_bias += args.normal_logit_bias
    if cli_biases is None and normal_bias == 0.0:
        return None
    biases = list(cli_biases) if cli_biases is not None else [0.0, 0.0, 0.0, 0.0]
    biases[3] += normal_bias
    return biases


def forward_with_tta(model, images, vessel_image, vessel_features, segmentation_mask, use_tta: bool):
    outputs = model(images, vessel_image, vessel_features, segmentation_mask=segmentation_mask)
    logits = [outputs["classification_logits"]]
    if use_tta:
        flipped_images = torch.flip(images, dims=[3])
        flipped_vessel = torch.flip(vessel_image, dims=[3])
        flipped_mask = torch.flip(segmentation_mask, dims=[3])
        logits.append(model(flipped_images, flipped_vessel, vessel_features, segmentation_mask=flipped_mask)["classification_logits"])
    outputs["classification_logits"] = torch.stack(logits, dim=0).mean(dim=0)
    return outputs


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    class_biases = resolve_class_biases(args, config)
    class_thresholds = args.class_thresholds if args.class_thresholds is not None else config.get("class_thresholds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaClassificationNet(
        num_classes=4,
        classifier_backbone=config.get("backbone_name", "efficientnet_b1"),
        pretrained=False,
        dropout=config.get("classifier_dropout", 0.5),
    ).to(device)
    if args.extra_checkpoints:
        state_dict = average_checkpoints_state_dict([args.checkpoint] + args.extra_checkpoints)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    segmentation_model = None
    if not args.use_ground_truth_masks:
        if not args.segmentation_checkpoint:
            raise ValueError("Provide --segmentation_checkpoint or use --use_ground_truth_masks.")
        seg_checkpoint = torch.load(args.segmentation_checkpoint, map_location="cpu")
        seg_config = seg_checkpoint["config"]
        segmentation_model = UKANSegmentationModel(
            base_channels=seg_config.get("segmentation_base_channels", 32),
            deep_supervision=seg_config.get("deep_supervision", False),
            image_size=seg_config.get("image_size", args.image_size),
            no_kan=seg_config.get("segmentation_no_kan", seg_config.get("no_kan", False)),
            attention_mode=seg_config.get("segmentation_attention_mode", seg_config.get("attention_mode", "none")),
        ).to(device)
        segmentation_model.load_state_dict(seg_checkpoint["model_state_dict"])
        segmentation_model.eval()

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

    class_weights = build_classification_class_weights(
        loaders["train"].dataset.dataframe,
        power=config.get("class_weight_power", 1.0),
        normal_boost=config.get("normal_boost", 1.0),
    ).to(device)
    criterion = ClassificationCriterion(
        loss_name=config.get("loss_name", "focal"),
        gamma=config.get("focal_gamma", 2.0),
        label_smoothing=config.get("classification_label_smoothing", 0.0),
        class_weights=class_weights,
        aux_weight=config.get("aux_classification_loss_weight", 0.25),
        confidence_penalty=config.get("confidence_penalty", 0.0),
    ).to(device)

    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    collected_logits: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test_cls", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            vessel_image, vessel_features, vessel_mask = build_vessel_inputs(images, masks, segmentation_model, args.use_ground_truth_masks)
            outputs = forward_with_tta(model, images, vessel_image, vessel_features, vessel_mask, args.tta)
            calibrated_logits = apply_class_calibration(
                outputs["classification_logits"],
                class_thresholds=class_thresholds,
                class_logit_biases=class_biases,
            )
            loss = criterion(calibrated_logits, outputs["aux_classification_logits"], labels)["loss"]
            preds = torch.argmax(calibrated_logits, dim=1)

            running_loss += loss.item()
            collected_logits.append(calibrated_logits.cpu())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    num_batches = max(len(loader), 1)
    epoch_logits = torch.cat(collected_logits, dim=0)
    final_metrics = classification_metrics_detailed(epoch_logits, torch.tensor(y_true, dtype=torch.long))
    final_metrics["loss"] = running_loss / num_batches
    save_json(final_metrics, os.path.join(args.output_dir, "classification_test_metrics.json"))
    save_confusion_matrix(
        y_true,
        y_pred,
        os.path.join(args.output_dir, "classification_test_confusion_matrix.png"),
        title="Classification Test Confusion Matrix",
    )

    print(
        f"Classification Test Loss {final_metrics['loss']:.4f} | "
        f"Accuracy {final_metrics['accuracy']:.4f} | Precision {final_metrics['macro_precision']:.4f} | "
        f"Recall {final_metrics['macro_recall']:.4f} | F1 {final_metrics['macro_f1']:.4f}"
    )
    print(
        "Per-class Test Precision | "
        + " | ".join(
            f"{class_name.upper()}: {value:.4f}" for class_name, value in final_metrics["per_class_precision"].items()
        )
    )
    print(
        "Per-class Test Recall | "
        + " | ".join(
            f"{class_name.upper()}: {value:.4f}" for class_name, value in final_metrics["per_class_recall"].items()
        )
    )
    print(
        "Per-class Test F1 | "
        + " | ".join(
            f"{class_name.upper()}: {value:.4f}" for class_name, value in final_metrics["per_class_f1"].items()
        )
    )


if __name__ == "__main__":
    main()
