from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b1",
        pretrained: bool = True,
        in_chans: int = 3,
        drop_rate: float = 0.15,
        projection_dim: int = 1024,
    ) -> None:
        super().__init__()
        # NEW: use multi-scale intermediate feature maps instead of only final pooled output.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        backbone_dim = sum(self.backbone.feature_info.channels())
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.feature_dim = projection_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = self.backbone(x)
        pooled = [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in features]
        return self.projection(torch.cat(pooled, dim=1))


class TriBranchAttentionFusion(nn.Module):
    """
    Fuse raw image, vessel-enhanced image, and segmentation-mask structure.
    The raw branch is biased slightly higher to preserve NORMAL appearance cues.
    """

    def __init__(self, feature_dim: int, raw_bias: float = 0.45) -> None:
        super().__init__()
        initial = torch.tensor([raw_bias, 0.35, 0.20], dtype=torch.float32)
        self.logits = nn.Parameter(torch.log(initial))
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_features: torch.Tensor,
        vessel_features: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        combined = torch.cat([raw_features, vessel_features, mask_features], dim=1)
        dynamic = torch.softmax(self.refine(combined), dim=1)
        static = torch.softmax(self.logits, dim=0).unsqueeze(0).expand_as(dynamic)
        weights = 0.5 * dynamic + 0.5 * static
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        balanced_raw = raw_features * weights[:, :1]
        balanced_vessel = vessel_features * weights[:, 1:2]
        balanced_mask = mask_features * weights[:, 2:3]

        cross_gate = self.cross_gate(combined)
        gated = cross_gate * (balanced_raw + balanced_vessel + balanced_mask) / 3.0
        fused = torch.cat([balanced_raw, balanced_vessel, balanced_mask, gated], dim=1)
        return {
            "fused": fused,
            "raw_weight": weights[:, :1],
            "vessel_weight": weights[:, 1:2],
            "mask_weight": weights[:, 2:3],
        }


class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "efficientnet_b1",
        pretrained: bool = True,
        dropout: float = 0.25,
        projection_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.raw_branch = EfficientBranch(backbone_name, pretrained=pretrained, in_chans=3, drop_rate=dropout, projection_dim=projection_dim)
        self.vessel_branch = EfficientBranch(backbone_name, pretrained=pretrained, in_chans=3, drop_rate=dropout, projection_dim=projection_dim)
        self.mask_branch = EfficientBranch(backbone_name, pretrained=pretrained, in_chans=3, drop_rate=dropout, projection_dim=projection_dim)
        self.feature_fusion = TriBranchAttentionFusion(self.raw_branch.feature_dim)
        fusion_dim = self.raw_branch.feature_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.fusion_dim = fusion_dim

    def extract_features(self, raw_image: torch.Tensor, vessel_image: torch.Tensor, mask_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_features = self.raw_branch(raw_image)
        vessel_features = self.vessel_branch(vessel_image)
        mask_features = self.mask_branch(mask_image)
        return raw_features, vessel_features, mask_features

    def forward(self, raw_image: torch.Tensor, vessel_image: torch.Tensor, mask_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        raw_features, vessel_features, mask_features = self.extract_features(raw_image, vessel_image, mask_image)
        fusion_outputs = self.feature_fusion(raw_features, vessel_features, mask_features)
        fused = fusion_outputs["fused"]
        logits = self.classifier(fused)
        return logits, fused, fusion_outputs
