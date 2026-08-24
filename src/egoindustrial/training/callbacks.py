"""Custom PyTorch Lightning callbacks."""

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class EMAModelCallback(Callback):
    """Exponential Moving Average of model weights."""

    def __init__(self, decay: float = 0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.use_ema_weights = use_ema_weights
        self.ema_weights = None

    def on_train_start(self, trainer, pl_module):
        self.ema_weights = {
            name: param.detach().clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.ema_weights:
                    self.ema_weights[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        if self.use_ema_weights and self.ema_weights:
            self.original_weights = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.ema_weights:
                    param.data.copy_(self.ema_weights[name])

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.use_ema_weights and hasattr(self, "original_weights"):
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.original_weights:
                    param.data.copy_(self.original_weights[name])


class GradientClippingCallback(Callback):
    """Gradient clipping with logging."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), self.max_norm, self.norm_type
        )
        if trainer.is_global_zero:
            pl_module.log("train/grad_norm", grad_norm, prog_bar=False)
