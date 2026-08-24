"""VideoMAEv2 model wrapper."""


import torch
import torch.nn as nn
from timm.models.videomae import vit_base_patch16_224

from egoindustrial.models.head import MultiTaskHead
from egoindustrial.models.registry import register_model


@register_model("videomaev2")
class VideoMAEv2(nn.Module):
    """VideoMAEv2 for egocentric action recognition."""

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

        # Load pretrained VideoMAE
        self.backbone = vit_base_patch16_224(pretrained=pretrained)
        embed_dim = self.backbone.embed_dim

        # Replace head with multi-task head
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
        # x: [B, C, T, H, W] or [B, T, C, H, W]
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        features = self.backbone.forward_features(x)
        # Use cls token
        cls_token = features[:, 0]
        return self.head(cls_token)
