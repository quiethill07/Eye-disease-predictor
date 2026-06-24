import argparse
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import INV_LABEL_MAP
from fusion_model import VisionDeepRetinaNet
from utils import ensure_dir


def build_inference_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on new retinal images.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    parser.add_argument("--image_size", type=int, default=512)
    return parser.parse_args()


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = image.detach().cpu().numpy()
    image = (image * std + mean).clip(0, 1)
    return np.transpose(image, (1, 2, 0))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionDeepRetinaNet(
        num_classes=4,
        segmentation_base_channels=config["segmentation_base_channels"],
        classifier_backbone=config["backbone_name"],
        pretrained=False,
        dropout=config.get("classifier_dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_inference_transform(args.image_size)

    with torch.no_grad():
        for path in args.input_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            tensor = transformed["image"].unsqueeze(0).to(device)

            outputs = model(tensor)
            seg_mask = (torch.sigmoid(outputs["segmentation_logits"]) > 0.5).float()
            class_idx = int(torch.argmax(outputs["classification_logits"], dim=1).item())
            class_name = INV_LABEL_MAP[class_idx].upper()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(denormalize(tensor[0]))
            axes[0].set_title(f"Prediction: {class_name}")
            axes[1].imshow(seg_mask[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[1].set_title("Predicted Vessel Mask")
            for axis in axes:
                axis.axis("off")
            plt.tight_layout()

            file_name = os.path.splitext(os.path.basename(path))[0]
            plt.savefig(os.path.join(args.output_dir, f"{file_name}_prediction.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
