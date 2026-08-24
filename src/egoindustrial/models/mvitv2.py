"""MViTv2 model wrapper."""


import torch
import torch.nn as nn
from timm.models.mvit import mvit_base_16x4

from egoindustrial.models.head import MultiTaskHead
from egoindustrial.models.registry import register_model


@register_model("mvitv2")
class MViTv2(nn.Module):
    """MViTv2 for egocentric action recognition."""

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        num_action_classes: int = 3806,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = mvit_base_16x4(pretrained=pretrained)
        embed_dim = self.backbone.head.in_features

        self.backbone.head = nn.Identity()
        self.head = MultiTaskHead(
            embed_dim=embed_dim,
            num_verb_classes=num_verb_classes,
            num_noun_classes=num_noun_classes,
            num_action_classes=num_action_classes,
            dropout=dropout,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        features = self.backbone(x)
        return self.head(features)
