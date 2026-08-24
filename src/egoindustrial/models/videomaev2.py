"""VideoMAEv2 model wrapper - uses ViT from timm as backbone."""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    timm = None

from egoindustrial.models.head import MultiTaskHead
from egoindustrial.models.registry import register_model


@register_model("videomaev2")
class VideoMAEv2(nn.Module):
    """VideoMAEv2 for egocentric action recognition (uses ViT backbone)."""

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
        self.num_verb_classes = num_verb_classes
        self.num_noun_classes = num_noun_classes
        self.num_action_classes = num_action_classes

        if not HAS_TIMM:
            raise ImportError(
                "VideoMAEv2 requires timm. Install with: pip install timm"
            )

        # Use ViT-B/16 from timm as backbone (VideoMAE uses ViT architecture)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        embed_dim = self.backbone.embed_dim

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
        # x: [B, C, T, H, W] or [B, T, C, H, W]
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        # Process each frame through ViT
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, -1)
        # Temporal pooling (mean over time)
        features = features.mean(dim=1)
        return self.head(features)
