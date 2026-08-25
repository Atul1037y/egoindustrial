#!/usr/bin/env python3
"""
Model Distillation Script - Teacher-Student distillation for faster inference.

Usage:
    python scripts/distill.py \
        --teacher-checkpoint outputs/checkpoints/best.ckpt \
        --student-model videomae \
        --output-dir outputs/distilled \
        --epochs 20
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egoindustrial.data import build_dataloader, build_datasets
from egoindustrial.data.transforms import get_transforms
from egoindustrial.models import get_model
from egoindustrial.training.losses import MultiTaskLoss
from egoindustrial.training.module import EgoIndustrialModule


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard labels and soft teacher predictions."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        verb_weight: float = 1.0,
        noun_weight: float = 1.0,
        action_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.ce_loss = MultiTaskLoss(
            verb_weight=verb_weight,
            noun_weight=noun_weight,
            action_weight=action_weight,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        student_logits: dict[str, torch.Tensor],
        teacher_logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Hard label loss (student vs ground truth)
        hard_losses = self.ce_loss(student_logits, targets)

        # Soft label loss (student vs teacher) - KL divergence
        soft_losses = {}
        for task in ["verb", "noun", "action"]:
            student_logits_t = student_logits[task] / self.temperature
            teacher_logits_t = teacher_logits[task] / self.temperature

            soft_loss = F.kl_div(
                F.log_softmax(student_logits_t, dim=-1),
                F.softmax(teacher_logits_t, dim=-1),
                reduction="batchmean",
            ) * (self.temperature ** 2)
            soft_losses[task] = soft_loss

        # Combined loss
        total_loss = (
            (1 - self.alpha) * hard_losses["total"]
            + self.alpha * sum(soft_losses.values())
        )

        return {
            **hard_losses,
            "soft_verb": soft_losses["verb"],
            "soft_noun": soft_losses["noun"],
            "soft_action": soft_losses["action"],
            "distill_total": total_loss,
        }


def distill(
    teacher_ckpt: str,
    student_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    epochs: int = 20,
    lr: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "cuda",
    log_every: int = 10,
) -> dict:
    """Run knowledge distillation."""

    device = torch.device(device)

    # Load teacher
    teacher = EgoIndustrialModule.load_from_checkpoint(teacher_ckpt)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student
    student = get_model(
        student_cfg.get("name", "videomaev2"),
        num_verb_classes=student_cfg.get("num_verb_classes", 97),
        num_noun_classes=student_cfg.get("num_noun_classes", 300),
        num_action_classes=student_cfg.get("num_action_classes", 3806),
        pretrained=student_cfg.get("pretrained", True),
        dropout=student_cfg.get("dropout", 0.5),
        freeze_backbone=student_cfg.get("freeze_backbone", False),
    )
    student.to(device)
    student.train()

    # Loss and optimizer
    distill_loss = DistillationLoss(
        temperature=temperature,
        alpha=alpha,
        verb_weight=1.0,
        noun_weight=1.0,
        action_weight=1.0,
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Logging
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        student.train()
        train_losses = {"distill_total": 0.0, "hard_verb": 0.0, "hard_noun": 0.0, "hard_action": 0.0}

        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            video = batch["video"].to(device, non_blocking=True)
            targets = {
                "verb_label": batch["verb_label"].to(device, non_blocking=True),
                "noun_label": batch["noun_label"].to(device, non_blocking=True),
                "action_label": batch["action_label"].to(device, non_blocking=True),
            }

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(video)

            # Student forward
            student_logits = student(video)

            # Compute distillation loss
            losses = distill_loss(student_logits, teacher_logits, targets)
            loss = losses["distill_total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # Accumulate
            for k in train_losses:
                if k in losses:
                    train_losses[k] += losses[k].item()

            if batch_idx % log_every == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # Validation
        student.eval()
        val_losses = {"total": 0.0, "verb": 0.0, "noun": 0.0, "action": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device, non_blocking=True)
                targets = {
                    "verb_label": batch["verb_label"].to(device, non_blocking=True),
                    "noun_label": batch["noun_label"].to(device, non_blocking=True),
                    "action_label": batch["action_label"].to(device, non_blocking=True),
                }

                student_logits = student(video)
                losses = distill_loss(student_logits, teacher(video), targets)
                val_losses["total"] += losses["distill_total"].item()
                for k in ["verb", "noun", "action"]:
                    val_losses[k] += distill_loss.ce_loss(student_logits, batch)["total"].item()

        # Average
        for k in train_losses:
            train_losses[k] /= len(train_loader)
        for k in val_losses:
            val_losses[k] /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {train_losses['distill_total']:.4f} "
              f"Val Loss: {val_losses['total']:.4f}")

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": student_cfg,
            }, Path(output_dir) / "distilled_best.ckpt")
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Save last
        torch.save({
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses["total"],
            "config": student_cfg,
        }, Path(output_dir) / "distilled_last.ckpt")

        scheduler.step()

    return {
        "best_val_loss": best_val_loss,
        "student_ckpt": str(Path(output_dir) / "distilled_best.ckpt"),
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for EgoIndustrial")
    parser.add_argument("--teacher-checkpoint", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--student-model", default="videomaev2", choices=["videomaev2", "mvitv2", "slowfast"])
    parser.add_argument("--output-dir", default="outputs/distilled", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Distillation weight (0-1)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-root", default="/data", help="Data root directory")

    args = parser.parse_args()

    # Data
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "dataset": {
            "name": "mixture",
            "datasets": {
                "epic_kitchens": {"root": f"{args.data_root}/epic_kitchens", "split": "train", "clip_len": 16, "frame_stride": 2},
                "assembly101": {"root": f"{args.data_root}/assembly101", "split": "train", "clip_len": 16, "frame_stride": 2},
                "holoassist": {"root": f"{args.data_root}/holoassist", "split": "train", "clip_len": 16, "frame_stride": 2},
            },
            "domain_probs": [0.5, 0.3, 0.2],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "train": {"max_epochs": args.epochs},
    })

    transform_cfg = {"clip_len": 16, "crop_size": 224, "resize_size": 256, "is_train": True, "model_type": "videomae"}
    get_transforms(transform_cfg)

    datasets = build_datasets(cfg.dataset)
    train_loader = build_dataloader(datasets, batch_size=args.batch_size, num_workers=args.num_workers,
                                    shuffle=True, domain_probs=[0.5, 0.3, 0.2], drop_last=True)

    val_datasets = build_datasets({**cfg.dataset, "split": "val"})
    val_loader = build_dataloader(val_datasets, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=False, drop_last=False)

    # Student config
    student_cfg = {
        "name": args.student_model,
        "num_verb_classes": 97,
        "num_noun_classes": 300,
        "num_action_classes": 3806,
        "pretrained": True,
        "dropout": 0.5,
        "freeze_backbone": False,
    }

    # Run distillation
    result = distill(
        teacher_ckpt=args.teacher_checkpoint,
        student_cfg=student_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        device=args.device,
    )

    print("\nDistillation complete!")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Student checkpoint: {result['student_ckpt']}")


if __name__ == "__main__":
    main()
