"""PyTorch Lightning module for egocentric action recognition."""

from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from egoindustrial.models import get_model
from egoindustrial.training.losses import MultiTaskLoss
from egoindustrial.training.metrics import get_metrics


class EgoIndustrialModule(LightningModule):
    """Lightning module for multi-dataset action recognition."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Model
        model_cfg = cfg.get("model", {})
        self.model = get_model(
            model_cfg.get("name", "videomaev2"),
            num_verb_classes=model_cfg.get("num_verb_classes", 97),
            num_noun_classes=model_cfg.get("num_noun_classes", 300),
            num_action_classes=model_cfg.get("num_action_classes", 3806),
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.5),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )

        # Loss
        loss_cfg = cfg.get("loss", {})
        self.loss_fn = MultiTaskLoss(
            verb_weight=loss_cfg.get("verb_weight", 1.0),
            noun_weight=loss_cfg.get("noun_weight", 1.0),
            action_weight=loss_cfg.get("action_weight", 1.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        )

        # Metrics
        self.train_metrics = get_metrics(
            model_cfg.get("num_verb_classes", 97),
            model_cfg.get("num_noun_classes", 300),
            model_cfg.get("num_action_classes", 3806),
            prefix="train_",
        )
        self.val_metrics = get_metrics(
            model_cfg.get("num_verb_classes", 97),
            model_cfg.get("num_noun_classes", 300),
            model_cfg.get("num_action_classes", 3806),
            prefix="val_",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds = self(batch["video"])
        losses = self.loss_fn(preds, batch)

        # Log losses
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, prog_bar=k == "total", sync_dist=True)

        # Update metrics
        self.train_metrics.update(
            preds["verb"], batch["verb_label"],
            preds["noun"], batch["noun_label"],
            preds["action"], batch["action_label"],
        )

        return losses["total"]

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        preds = self(batch["video"])
        losses = self.loss_fn(preds, batch)

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, prog_bar=k == "total", sync_dist=True)

        self.val_metrics.update(
            preds["verb"], batch["verb_label"],
            preds["noun"], batch["noun_label"],
            preds["action"], batch["action_label"],
        )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        opt_cfg = self.cfg.get("optimizer", {})
        lr = opt_cfg.get("lr", 1e-4)
        weight_decay = opt_cfg.get("weight_decay", 0.05)

        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        sched_cfg = self.cfg.get("scheduler", {})
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        max_epochs = sched_cfg.get("max_epochs", 50)

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
