"""Multi-task losses for action recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combined loss for verb, noun, action prediction."""

    def __init__(
        self,
        verb_weight: float = 1.0,
        noun_weight: float = 1.0,
        action_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.verb_weight = verb_weight
        self.noun_weight = noun_weight
        self.action_weight = action_weight

        self.verb_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.noun_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.action_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = {
            "verb": self.verb_loss(preds["verb"], targets["verb_label"]),
            "noun": self.noun_loss(preds["noun"], targets["noun_label"]),
            "action": self.action_loss(preds["action"], targets["action_label"]),
        }

        total = (
            self.verb_weight * losses["verb"]
            + self.noun_weight * losses["noun"]
            + self.action_weight * losses["action"]
        )
        losses["total"] = total
        return losses


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
