"""Metrics for action recognition evaluation."""

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
)


def get_metrics(
    num_verb: int, num_noun: int, num_action: int, prefix: str = ""
) -> MetricCollection:
    """Get metric collection for verb/noun/action."""
    return MetricCollection(
        {
            f"{prefix}verb_top1": MulticlassAccuracy(num_classes=num_verb, top_k=1),
            f"{prefix}verb_top5": MulticlassAccuracy(num_classes=num_verb, top_k=5),
            f"{prefix}noun_top1": MulticlassAccuracy(num_classes=num_noun, top_k=1),
            f"{prefix}noun_top5": MulticlassAccuracy(num_classes=num_noun, top_k=5),
            f"{prefix}action_top1": MulticlassAccuracy(num_classes=num_action, top_k=1),
            f"{prefix}action_top5": MulticlassAccuracy(num_classes=num_action, top_k=5),
        }
    )


def compute_per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute per-class accuracy."""
    correct = torch.zeros(num_classes, device=preds.device)
    total = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            correct[c] = (preds[mask] == c).sum().float()
            total[c] = mask.sum().float()

    return correct / (total + 1e-8)
