import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Multi-task head for verb, noun, and action classification."""

    def __init__(
        self,
        embed_dim: int,
        num_verb_classes: int,
        num_noun_classes: int,
        num_action_classes: int,
        dropout: float = 0.5,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_verb_classes = num_verb_classes
        self.num_noun_classes = num_noun_classes
        self.num_action_classes = num_action_classes

        hidden_dim = hidden_dim or embed_dim

        self.shared = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.verb_head = nn.Linear(hidden_dim, num_verb_classes)
        self.noun_head = nn.Linear(hidden_dim, num_noun_classes)
        self.action_head = nn.Linear(hidden_dim, num_action_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {
            "verb": self.verb_head(shared),
            "noun": self.noun_head(shared),
            "action": self.action_head(shared),
        }


class SingleTaskHead(nn.Module):
    """Single-task head for finetuning on specific task."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.5,
        task: str = "action",
    ):
        super().__init__()
        self.task = task
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
