import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import warnings


class SegGuidedEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 classifier guided by segmentation masks.
    The mask acts as a spatial attention map over input RGB.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            except Exception as exc:
                warnings.warn(
                    f'Pretrained EfficientNet-B0 download/load failed ({exc}). '
                    f'Falling back to randomly initialized weights.'
                )
                self.backbone = efficientnet_b0(weights=None)
        else:
            self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Learnable strength for mask-guided spatial attention.
        self.mask_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, mask):
        if mask.ndim != 4 or mask.size(1) != 1:
            raise ValueError(f'Expected mask shape [B,1,H,W], got {tuple(mask.shape)}')

        mask = F.interpolate(mask, size=image.shape[-2:], mode='bilinear', align_corners=False)
        mask = torch.clamp(mask, 0.0, 1.0)
        guided_image = image * (1.0 + self.mask_scale * mask)
        logits = self.backbone(guided_image)
        return logits


class DualInputSegGuidedEfficientNet(nn.Module):
    """
    Dual-input EfficientNet classifier guided by segmentation masks.
    Uses a single backbone with two input views:
    1. Raw RGB image
    2. Vessel-enhanced image (RGB * mask)
    Features from both views are fused before classification.
    """

    def __init__(self, num_classes, pretrained=True, dropout=0.2):
        super().__init__()
        if pretrained:
            try:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            except Exception as exc:
                warnings.warn(
                    f'Pretrained EfficientNet-B0 download/load failed ({exc}). '
                    f'Falling back to randomly initialized weights.'
                )
                self.backbone = efficientnet_b0(weights=None)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Remove the classifier head to get features
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, image, mask):
        if mask.ndim != 4 or mask.size(1) != 1:
            raise ValueError(f'Expected mask shape [B,1,H,W], got {tuple(mask.shape)}')

        mask = F.interpolate(mask, size=image.shape[-2:], mode='bilinear', align_corners=False)
        mask = torch.clamp(mask, 0.0, 1.0)

        # Raw RGB view
        raw_features = self.backbone(image)

        # Vessel-enhanced view (RGB * mask)
        vessel_image = image * mask
        vessel_features = self.backbone(vessel_image)

        # Fuse features
        fused = torch.cat([raw_features, vessel_features], dim=1)
        logits = self.fusion(fused)
        return logits
