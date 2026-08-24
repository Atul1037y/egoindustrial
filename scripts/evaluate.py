#!/usr/bin/env python3
"""Evaluation script for EgoIndustrial trained models."""

import argparse
import json

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoindustrial.data import build_dataloader, build_datasets
from egoindustrial.data.transforms import get_transforms
from egoindustrial.models import get_model
from egoindustrial.training.metrics import get_metrics


class Evaluator:
    """Evaluator for trained models."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        if config_path:
            self.cfg = OmegaConf.load(config_path)
        elif "hyper_parameters" in self.checkpoint:
            self.cfg = OmegaConf.create(self.checkpoint["hyper_parameters"])
        else:
            raise ValueError("No config found. Provide --config or ensure checkpoint has hyper_parameters.")

        # Build model
        model_cfg = self.cfg.get("model", {})
        self.model = get_model(
            model_cfg.get("name", "videomaev2"),
            num_verb_classes=model_cfg.get("num_verb_classes", 97),
            num_noun_classes=model_cfg.get("num_noun_classes", 300),
            num_action_classes=model_cfg.get("num_action_classes", 3806),
            pretrained=model_cfg.get("pretrained", False),
            dropout=model_cfg.get("dropout", 0.5),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )

        # Load weights
        state_dict = self.checkpoint.get("state_dict", self.checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Transforms
        transform_cfg = self.cfg.get("dataset", {}).get("transforms", {})
        transform_cfg["is_train"] = False
        self.transforms = get_transforms(transform_cfg)

        # Metrics
        self.metrics = get_metrics(
            model_cfg.get("num_verb_classes", 97),
            model_cfg.get("num_noun_classes", 300),
            model_cfg.get("num_action_classes", 3806),
        )

    def evaluate_dataloader(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate on a dataloader."""
        self.metrics.to(self.device)
        self.metrics.reset()

        all_preds = {"verb": [], "noun": [], "action": []}
        all_targets = {"verb": [], "noun": [], "action": []}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                video = batch["video"].to(self.device)
                targets = {
                    "verb": batch["verb_label"].to(self.device),
                    "noun": batch["noun_label"].to(self.device),
                    "action": batch["action_label"].to(self.device),
                }

                preds = self.model(video)

                # Update metrics
                self.metrics.update(
                    preds["verb"], targets["verb"],
                    preds["noun"], targets["noun"],
                    preds["action"], targets["action"],
                )

                # Store for per-class analysis
                for k in ["verb", "noun", "action"]:
                    all_preds[k].append(preds[k].argmax(dim=1).cpu())
                    all_targets[k].append(targets[k].cpu())

        # Compute metrics
        results = self.metrics.compute()
        self.metrics.reset()

        # Convert to CPU
        results = {k: float(v) for k, v in results.items()}

        return results

    def evaluate_dataset(self, dataset_name: str, split: str = "val") -> dict[str, float]:
        """Evaluate on a specific dataset."""
        # Build dataset config
        dataset_cfg = {
            "name": dataset_name,
            "root": self.cfg.paths.data_root / dataset_name,
            "split": split,
            "clip_len": self.cfg.dataset.clip_len,
            "frame_stride": self.cfg.dataset.frame_stride,
            "batch_size": self.cfg.dataset.batch_size,
            "num_workers": self.cfg.dataset.num_workers,
        }

        # Build dataset and dataloader
        datasets = [build_datasets({**dataset_cfg})]  # Simplified for single dataset
        dataloader = build_dataloader(
            datasets,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=False,
            drop_last=False,
        )

        return self.evaluate_dataloader(dataloader)

    def evaluate_all(self) -> dict[str, dict[str, float]]:
        """Evaluate on all validation datasets."""
        results = {}

        for dataset_name in ["epic_kitchens", "assembly101", "holoassist"]:
            print(f"\nEvaluating on {dataset_name}...")
            try:
                results[dataset_name] = self.evaluate_dataset(dataset_name, "val")
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}

        return results

    def get_predictions(self, dataloader: DataLoader) -> pd.DataFrame:
        """Get detailed predictions for analysis."""
        self.model.eval()

        rows = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions"):
                video = batch["video"].to(self.device)
                preds = self.model(video)

                for i in range(len(batch["video_id"])):
                    row = {
                        "video_id": batch["video_id"][i],
                        "start_frame": batch["start_frame"][i].item(),
                        "end_frame": batch["end_frame"][i].item(),
                        "verb_true": batch["verb_label"][i].item(),
                        "noun_true": batch["noun_label"][i].item(),
                        "action_true": batch["action_label"][i].item(),
                        "verb_pred": preds["verb"][i].argmax().item(),
                        "noun_pred": preds["noun"][i].argmax().item(),
                        "action_pred": preds["action"][i].argmax().item(),
                        "verb_conf": torch.softmax(preds["verb"][i], dim=-1).max().item(),
                        "noun_conf": torch.softmax(preds["noun"][i], dim=-1).max().item(),
                        "action_conf": torch.softmax(preds["action"][i], dim=-1).max().item(),
                    }
                    rows.append(row)

        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EgoIndustrial model")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", help="Path to config YAML")
    parser.add_argument("--data-root", default="/data", help="Data root directory")
    parser.add_argument("--dataset", choices=["epic_kitchens", "assembly101", "holoassist", "all"],
                        default="all", help="Dataset to evaluate on")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Data split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--predictions", help="Output CSV file for predictions")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")

    args = parser.parse_args()

    # Override data root in config
    import os
    os.environ["DATA_ROOT"] = args.data_root

    # Create evaluator
    evaluator = Evaluator(args.checkpoint, args.config, args.device)

    # Run evaluation
    if args.dataset == "all":
        results = evaluator.evaluate_all()
    else:
        results = {args.dataset: evaluator.evaluate_dataset(args.dataset, args.split)}

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Save predictions
    if args.predictions:
        # This would require building dataloader for predictions
        print("Predictions export not fully implemented yet.")


if __name__ == "__main__":
    main()
