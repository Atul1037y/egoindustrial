"""SlowFast model wrapper."""


import torch
import torch.nn as nn

try:
    from pytorchvideo.models.slowfast import create_slowfast
    HAS_PYTORCHVIDEO = True
except ImportError:
    HAS_PYTORCHVIDEO = False
    create_slowfast = None

from egoindustrial.models.head import MultiTaskHead
from egoindustrial.models.registry import register_model


@register_model("slowfast")
class SlowFast(nn.Module):
    """SlowFast for egocentric action recognition."""

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        num_action_classes: int = 3806,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        slowfast_alpha: int = 4,
        slowfast_beta: float = 1/8,
    ):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        self.slowfast_beta = slowfast_beta

        if not HAS_PYTORCHVIDEO:
            raise ImportError(
                "SlowFast requires pytorchvideo. "
                "Install with: pip install pytorchvideo or use a different model."
            )

        self.backbone = create_slowfast(
            model_num_class=400,  # Kinetics pretrained
            pretrained=pretrained,
            slowfast_alpha=slowfast_alpha,
            slowfast_beta=slowfast_beta,
        )
        embed_dim = self.backbone.blocks[-1].proj.in_features

        self.backbone.blocks[-1].proj = nn.Identity()
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

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        # x: [slow_pathway, fast_pathway] each [B, C, T, H, W]
        features = self.backbone(x)
        return self.head(features)
