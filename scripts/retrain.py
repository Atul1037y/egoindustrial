#!/usr/bin/env python3
"""
Automated Retraining Pipeline with Drift Detection.

Monitors model performance and data drift, triggers retraining when needed.

Usage:
    python scripts/retrain.py --config configs/retrain.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from prometheus_client import Counter, Gauge, Histogram, start_http_server

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egoindustrial.data import build_dataloader, build_datasets
from egoindustrial.data.transforms import get_transforms
from egoindustrial.inference.export_onnx import export_to_onnx
from egoindustrial.inference.tensorrt_engine import build_tensorrt_engine
from egoindustrial.training.module import EgoIndustrialModule

# Prometheus metrics
DRIFT_SCORE = Gauge("egoindustrial_drift_score", "Data drift score (0-1)")
PERFORMANCE_DROP = Gauge("egoindustrial_performance_drop", "Performance drop from baseline")
RETRAIN_TRIGGERED = Counter("egoindustrial_retrain_triggered_total", "Number of retraining triggers")
RETRAIN_DURATION = Histogram("egoindustrial_retrain_duration_seconds", "Retraining duration")
MODEL_ACCURACY = Gauge("egoindustrial_model_accuracy", "Model accuracy on validation set")


class DriftDetector:
    """Detects data and concept drift."""

    def __init__(self, reference_embeddings_path: str, threshold: float = 0.1):
        self.reference_embeddings = torch.load(reference_embeddings_path)
        self.threshold = threshold
        self.feature_extractor = None

    def extract_features(self, model, dataloader, device):
        """Extract embeddings from model."""
        model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                video = batch["video"].to(device)
                with torch.no_grad():
                    features = model.backbone.forward_features(video)
                    if hasattr(features, 'shape') and len(features.shape) > 2:
                        features = features[:, 0]  # CLS token
                    embeddings.append(features.cpu())
        return torch.cat(embeddings)

    def compute_drift(self, current_embeddings: torch.Tensor) -> float:
        """Compute MMD or cosine distance from reference."""

        # MMD with RBF kernel
        def mmd(x, y):
            xx = torch.cdist(x, x).pow(2)
            yy = torch.cdist(y, y).pow(2)
            xy = torch.cdist(x, y).pow(2)

            gamma = 1.0 / x.shape[1]
            kxx = torch.exp(-gamma * xx).mean()
            kyy = torch.exp(-gamma * yy).mean()
            kxy = torch.exp(-gamma * xy).mean()
            return kxx + kyy - 2 * kxy

        drift = mmd(current_embeddings, self.reference_embeddings).item()
        return drift

    def check_drift(self, model, dataloader, device) -> bool:
        """Check if drift exceeds threshold."""
        embeddings = self.extract_features(model, dataloader, device)
        drift = self.compute_drift(embeddings)
        DRIFT_SCORE.set(drift)
        return drift > self.threshold


class PerformanceMonitor:
    """Monitors model performance on validation data."""

    def __init__(self, baseline_metrics: dict, threshold: float = 0.02):
        self.baseline = baseline_metrics
        self.threshold = threshold

    def evaluate(self, model, dataloader, device) -> dict:
        """Evaluate model and return metrics."""
        model.eval()
        model.to(device)

        correct = {"verb": 0, "noun": 0, "action": 0}
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                video = batch["video"].to(device)
                {
                    "verb": batch["verb_label"].to(device),
                    "noun": batch["noun_label"].to(device),
                    "action": batch["action_label"].to(device),
                }

                with torch.no_grad():
                    preds = model(video)

                for task in ["verb", "noun", "action"]:
                    preds = batch[task].argmax(dim=1)
                    correct[task] += (preds == batch[f"{task}_label"]).sum().item()

                total += batch["video"].size(0)

        return {f"{k}_acc": v / len(dataloader.dataset) for k, v in correct.items()}

    def check_performance_drop(self, current_metrics: dict) -> bool:
        """Check if performance dropped below threshold."""
        drops = {}
        for task in ["verb", "noun", "action"]:
            baseline = self.baseline.get(f"{task}_acc", 1.0)
            current = current_metrics.get(f"{task}_acc", 0.0)
            drop = baseline - current
            drops[task] = drop
            PERFORMANCE_DROP.labels(task=task).set(drop)

        max_drop = max(drops.values())
        return max_drop > self.threshold


class AutomatedRetrainer:
    """Orchestrates automated retraining pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.output_dir = Path(config["paths"]["output_root"])
        self.drift_detector = DriftDetector(
            config["drift"]["reference_embeddings"],
            threshold=config["drift"]["threshold"]
        )
        self.perf_monitor = PerformanceMonitor(
            config["performance"]["baseline_metrics"],
            threshold=config["performance"]["threshold"]
        )
        self.last_retrain = 0
        self.retrain_cooldown = config.get("retrain_cooldown_hours", 24) * 3600

    def should_retrain(self, model, dataloader, device) -> tuple[bool, str]:
        """Check if retraining is needed."""
        now = time.time()
        if now - self.last_retrain < self.retrain_cooldown:
            return False, "Cooldown period active"

        # Check drift
        if self.drift_detector.check_drift(model, dataloader, device):
            return True, "Data drift detected"

        # Check performance
        metrics = self.perf_monitor.evaluate(model, dataloader, device)
        if self.perf_monitor.check_performance_drop(metrics):
            return True, "Performance dropped below threshold"

        return False, "No trigger"

    def retrain(self, config: dict) -> dict:
        """Execute full retraining pipeline."""
        start = time.time()
        RETRAIN_TRIGGERED.inc()

        try:
            # 1. Export current best model
            self.find_best_checkpoint()
            export_to_onnx(
                checkpoint_path=str(self.checkpoint_dir / "best.ckpt"),
                output_path=str(self.output_dir / "model.onnx"),
            )

            # 2. Build TensorRT engine
            build_tensorrt_engine(
                onnx_path="outputs/model.onnx",
                engine_path="outputs/model_int8.engine",
                precision="int8",
                max_batch_size=32,
            )

            # 3. Run distillation (optional - for smaller student)
            # result = distill(...)

            # 4. Deploy new model (if using Modal/GCP)
            # self.deploy_new_model()

            duration = time.time() - start
            RETRAIN_DURATION.observe(duration)
            self.last_retrain = time.time()

            logging.info(f"Retraining completed in {duration:.1f}s")
            return {"status": "success", "duration": duration}

        except Exception as e:
            logging.error(f"Retraining failed: {e}")
            return {"status": "failed", "error": str(e)}

    def find_best_checkpoint(self) -> Path:
        """Find best checkpoint by validation loss."""
        ckpts = list(self.checkpoint_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found")
        return min(ckpts, key=lambda p: float(p.stem.split("val_loss=")[1].split(".ckpt")[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to retrain config YAML")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Check interval (seconds)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    with open(args.config) as f:
        config = OmegaConf.load(f)

    # Start Prometheus metrics server
    start_http_server(9090)
    logging.info("Prometheus metrics server started on port 9090")

    # Load config
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Build dataloaders for monitoring
    transform_cfg = config["dataset"].get("transforms", {})
    transform_cfg["is_train"] = False
    get_transforms(transform_cfg)

    val_datasets = build_datasets({**config["dataset"], "split": "val"})
    val_loader = build_dataloader(
        val_datasets,
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    # Load current model
    retrainer = AutomatedRetrainer(config)

    if args.once:
        # One-time check
        model = EgoIndustrialModule.load_from_checkpoint(
            str(retrainer.find_best_checkpoint())
        )
        should_retrain, reason = retrainer.should_retrain(model, val_loader, "cuda")
        if should_retrain:
            logging.info(f"Retraining triggered: {reason}")
            result = retrainer.retrain(config)
            logging.info(f"Result: {result}")
        else:
            logging.info(f"No retraining needed: {reason}")
    else:
        # Continuous monitoring loop
        logging.info("Starting continuous monitoring...")
        while True:
            try:
                model = EgoIndustrialModule.load_from_checkpoint(
                    str(retrainer.find_best_checkpoint())
                )
                should_retrain, reason = retrainer.should_retrain(model, val_loader, "cuda")
                if should_retrain:
                    logging.info(f"Retraining triggered: {reason}")
                    result = retrainer.retrain(config)
                    logging.info(f"Result: {result}")
                else:
                    logging.info(f"Check passed: {reason}")

                time.sleep(config.get("check_interval", 3600))
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)


if __name__ == "__main__":
    main()
