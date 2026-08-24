"""InternVideo2 model wrapper."""

import torch
import torch.nn as nn

try:
    from transformers import AutoModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModel = None

from egoindustrial.models.head import MultiTaskHead
from egoindustrial.models.registry import register_model


@register_model("internvideo2")
class InternVideo2(nn.Module):
    """InternVideo2 for egocentric action recognition."""

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        num_action_classes: int = 3806,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        model_name: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "InternVideo2 requires transformers. "
                "Install with: pip install transformers or use a different model."
            )

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embed_dim = self.backbone.config.hidden_size

        # Replace classifier
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
        # x: [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head(features)
