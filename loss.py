from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        return ((1.0 - pt) ** self.gamma) * ce


class ClassificationCriterion(nn.Module):
    def __init__(
        self,
        loss_name: str = "focal",
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        aux_weight: float = 0.3,
        confidence_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_name = loss_name.lower()
        self.aux_weight = aux_weight
        self.confidence_penalty = confidence_penalty
        if self.loss_name not in {"ce", "focal"}:
            raise ValueError(f"Unsupported loss_name '{loss_name}'. Expected 'ce' or 'focal'.")
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None
        self.focal = MultiClassFocalLoss(
            gamma=gamma,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
        self.label_smoothing = label_smoothing

    def _base_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "ce":
            return F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
        return self.focal(logits, targets)

    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        main_loss = self._base_loss(main_logits, targets)
        aux_loss = self._base_loss(aux_logits, targets)

        if sample_weights is not None:
            normalized_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
            main_loss = main_loss * normalized_weights
            aux_loss = aux_loss * normalized_weights

        total_loss = main_loss.mean() + self.aux_weight * aux_loss.mean()
        if self.confidence_penalty > 0.0:
            probs = torch.softmax(main_logits, dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
            total_loss = total_loss - self.confidence_penalty * entropy

        return {
            "loss": total_loss,
            "main_loss": main_loss.mean().detach(),
            "aux_loss": aux_loss.mean().detach(),
        }


class HardExampleMemory:
    """
    Keep a lightweight per-sample weight that increases after mistakes and
    slowly relaxes once a sample becomes easy again.
    """

    def __init__(self, growth: float = 1.25, decay: float = 0.96, max_weight: float = 3.0) -> None:
        self.growth = growth
        self.decay = decay
        self.max_weight = max_weight
        self.sample_weights: Dict[str, float] = {}

    def get_weights(self, sample_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        weights = [self.sample_weights.get(sample_id, 1.0) for sample_id in sample_ids]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def update(self, sample_ids: Iterable[str], logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits.detach(), dim=1).cpu().tolist()
        truth = targets.detach().cpu().tolist()
        for sample_id, pred, target in zip(sample_ids, preds, truth):
            current = self.sample_weights.get(sample_id, 1.0)
            if pred != target:
                self.sample_weights[sample_id] = min(self.max_weight, current * self.growth)
            else:
                self.sample_weights[sample_id] = max(1.0, current * self.decay)
