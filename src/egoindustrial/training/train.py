"""Main training entry point."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from egoindustrial.data import build_dataloader, build_datasets
from egoindustrial.data.transforms import get_transforms
from egoindustrial.training.module import EgoIndustrialModule


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Seed
    seed_everything(cfg.seed, workers=True)

    # Data
    transforms = get_transforms(cfg.dataset.get("transforms", {}))
    datasets = build_datasets(cfg.dataset)
    train_loader = build_dataloader(
        datasets,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
        domain_probs=cfg.dataset.get("domain_probs"),
        drop_last=cfg.dataset.get("drop_last", True),
    )

    val_datasets = build_datasets({**cfg.dataset, "split": "val"})
    val_loader = build_dataloader(
        val_datasets,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Model
    model = EgoIndustrialModule(OmegaConf.to_container(cfg, resolve=True))

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename="epoch={epoch}-val_loss={val/loss_total:.4f}",
            monitor="val/loss_total",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss_total", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Logger
    logger = None
    if cfg.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            offline=cfg.wandb.mode == "offline",
        )

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if cfg.num_gpus > 0 else "cpu",
        devices=cfg.num_gpus,
        precision=cfg.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=cfg.train.get("gradient_clip", 1.0),
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        fast_dev_run=cfg.debug,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
