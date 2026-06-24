from typing import Dict

import torch
import torch.nn as nn

from model_classification import MultiModalClassifier
from model_segmentation import UKANSegmentationModel
from utils import extract_vessel_features_batch


class VesselFeatureNormalizer(nn.Module):
    def __init__(self, num_features: int = 3, momentum: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))
        self.momentum = momentum

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def update_stats(self, features: torch.Tensor) -> None:
        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.mean.lerp_(batch_mean.detach(), self.momentum)
        self.std.lerp_(batch_std.detach(), self.momentum)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.training and features.size(0) > 0:
            self.update_stats(features)
        return (features - self.mean) / self.std


class VesselFeatureTower(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CNNDominantFusion(nn.Module):
    """
    Fuse CNN and handcrafted vessel features while keeping the CNN branch dominant.
    The vessel branch starts with a low scale and must earn influence during training.
    """

    def __init__(self, cnn_dim: int, vessel_dim: int, dropout: float = 0.25, cnn_bias: float = 0.8) -> None:
        super().__init__()
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.vessel_norm = nn.LayerNorm(vessel_dim)
        self.vessel_align = nn.Linear(vessel_dim, cnn_dim)
        self.cross_gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(cnn_dim, cnn_dim),
            nn.Sigmoid(),
        )
        self.branch_attention = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim // 2),
            nn.GELU(),
            nn.Linear(cnn_dim // 2, 2),
        )
        self.vessel_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.output = nn.Sequential(
            nn.Linear(cnn_dim * 3, cnn_dim),
            nn.LayerNorm(cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cnn_bias = cnn_bias

    def forward(self, cnn_features: torch.Tensor, vessel_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        cnn = self.cnn_norm(cnn_features)
        vessel = self.vessel_norm(vessel_features)
        vessel_aligned = self.vessel_align(vessel)

        cross_gate = self.cross_gate(torch.cat([cnn, vessel_aligned], dim=1))
        gated_vessel = cross_gate * vessel_aligned * self.vessel_scale.clamp(0.05, 0.8)

        branch_logits = self.branch_attention(torch.cat([cnn, gated_vessel], dim=1))
        branch_weights = torch.softmax(branch_logits, dim=1)
        cnn_weight = branch_weights[:, :1] * self.cnn_bias + (1.0 - self.cnn_bias)
        vessel_weight = branch_weights[:, 1:2] * (1.0 - self.cnn_bias)

        dominant_cnn = cnn * cnn_weight
        restrained_vessel = gated_vessel * vessel_weight
        fused = self.output(torch.cat([dominant_cnn, restrained_vessel, dominant_cnn - restrained_vessel], dim=1))
        return {
            "fused_features": fused,
            "cnn_weight": cnn_weight,
            "vessel_weight": vessel_weight,
        }


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
            nn.Linear(192, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VisionDeepRetinaNet(nn.Module):
    """
    Joint model with:
    - U-KAN style vessel segmentation branch
    - Dual-branch classifier over raw images and vessel-enhanced images
    - Fusion head that combines CNN GAP features with handcrafted vessel biomarkers
    """

    def __init__(
        self,
        num_classes: int = 4,
        segmentation_base_channels: int = 32,
        classifier_backbone: str = "efficientnet_b1",
        pretrained: bool = True,
        dropout: float = 0.25,
        cnn_feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.segmentation_model = UKANSegmentationModel(base_channels=segmentation_base_channels)
        self.classification_model = MultiModalClassifier(
            num_classes=num_classes,
            backbone_name=classifier_backbone,
            pretrained=pretrained,
            dropout=dropout,
            projection_dim=cnn_feature_dim,
        )

        self.vessel_feature_normalizer = VesselFeatureNormalizer(num_features=3)
        self.vessel_feature_encoder = VesselFeatureTower(output_dim=128, dropout=dropout)
        self.learnable_fusion = CNNDominantFusion(
            cnn_dim=self.classification_model.fusion_dim,
            vessel_dim=128,
            dropout=dropout,
        )
        self.multi_modal_head = ClassificationHead(
            input_dim=self.classification_model.fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_outputs = self.segmentation_model(image)
        seg_logits = seg_outputs["logits"]
        vessel_prob = torch.sigmoid(seg_logits)
        vessel_rgb = vessel_prob.repeat(1, 3, 1, 1) * image
        mask_rgb = vessel_prob.repeat(1, 3, 1, 1)

        aux_class_logits, cnn_features, branch_outputs = self.classification_model(image, vessel_rgb, mask_rgb)
        vessel_feature_mask = (vessel_prob.detach() > 0.5).float()
        vessel_features = extract_vessel_features_batch(vessel_feature_mask).to(image.device)
        normalized_vessel_features = self.vessel_feature_normalizer(vessel_features)
        encoded_vessel_features = self.vessel_feature_encoder(normalized_vessel_features)
        fusion_outputs = self.learnable_fusion(cnn_features, encoded_vessel_features)
        final_logits = self.multi_modal_head(fusion_outputs["fused_features"])

        return {
            "segmentation_logits": seg_logits,
            "classification_logits": final_logits,
            "aux_classification_logits": aux_class_logits,
            "vessel_probability": vessel_prob,
            "cnn_features": cnn_features,
            "vessel_features": normalized_vessel_features,
            "raw_branch_weight": branch_outputs["raw_weight"],
            "mask_branch_weight": branch_outputs["mask_weight"],
            "cnn_fusion_weight": fusion_outputs["cnn_weight"],
            "vessel_fusion_weight": fusion_outputs["vessel_weight"],
        }


class RetinaClassificationNet(nn.Module):
    """
    Separate classification model that fuses:
    - CNN GAP features from raw fundus images
    - CNN GAP features from vessel-focused images
    - Handcrafted vessel biomarkers
    """

    def __init__(
        self,
        num_classes: int = 4,
        classifier_backbone: str = "efficientnet_b1",
        pretrained: bool = True,
        dropout: float = 0.25,
        cnn_feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.classification_model = MultiModalClassifier(
            num_classes=num_classes,
            backbone_name=classifier_backbone,
            pretrained=pretrained,
            dropout=dropout,
            projection_dim=cnn_feature_dim,
        )
        self.vessel_feature_normalizer = VesselFeatureNormalizer(num_features=3)
        self.vessel_feature_encoder = VesselFeatureTower(output_dim=128, dropout=dropout)
        self.learnable_fusion = CNNDominantFusion(
            cnn_dim=self.classification_model.fusion_dim,
            vessel_dim=128,
            dropout=dropout,
        )
        self.multi_modal_head = ClassificationHead(
            input_dim=self.classification_model.fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        raw_image: torch.Tensor,
        vessel_image: torch.Tensor,
        vessel_features: torch.Tensor,
        segmentation_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if segmentation_mask is None:
            segmentation_mask = vessel_image.mean(dim=1, keepdim=True)
        mask_rgb = segmentation_mask.repeat(1, 3, 1, 1)
        aux_class_logits, cnn_features, branch_outputs = self.classification_model(raw_image, vessel_image, mask_rgb)
        normalized_vessel_features = self.vessel_feature_normalizer(vessel_features)
        encoded_vessel_features = self.vessel_feature_encoder(normalized_vessel_features)
        fusion_outputs = self.learnable_fusion(cnn_features, encoded_vessel_features)
        final_logits = self.multi_modal_head(fusion_outputs["fused_features"])
        return {
            "classification_logits": final_logits,
            "aux_classification_logits": aux_class_logits,
            "cnn_features": cnn_features,
            "vessel_features": normalized_vessel_features,
            "raw_branch_weight": branch_outputs["raw_weight"],
            "vessel_branch_weight": branch_outputs["vessel_weight"],
            "mask_branch_weight": branch_outputs["mask_weight"],
            "cnn_fusion_weight": fusion_outputs["cnn_weight"],
            "vessel_fusion_weight": fusion_outputs["vessel_weight"],
        }
