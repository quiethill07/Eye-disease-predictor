from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

import archs


class UKANSegmentationModel(nn.Module):
    """V2-compatible wrapper around the stronger V1 UKAN segmentation model."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        deep_supervision: bool = False,
        image_size: int = 128,
        no_kan: bool = False,
        attention_mode: str = "none",
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.base_channels = base_channels
        self.image_size = image_size
        self.no_kan = no_kan
        self.attention_mode = attention_mode

        # V1 uses a fixed UKAN architecture; we keep the V2 class signature so
        # the newer training and classification pipelines can call it unchanged.
        self.backbone = archs.UKAN(
            num_classes=num_classes,
            input_channels=in_channels,
            deep_supervision=deep_supervision,
            img_size=image_size,
            no_kan=no_kan,
            attention_mode=attention_mode,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        logits = self.backbone(x)
        aux_logits = None

        # The V2 pipeline expects dict outputs even when the segmentation
        # backbone returns a plain tensor.
        if isinstance(logits, (list, tuple)):
            aux_logits = list(logits[:-1]) if len(logits) > 1 else None
            logits = logits[-1]

        return {
            "logits": logits,
            "features": logits,
            "aux_logits": aux_logits,
        }
