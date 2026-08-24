"""INT8 calibration dataset generation for TensorRT."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from egoindustrial.data import build_datasets
from egoindustrial.data.transforms import get_transforms


class CalibrationDataset(Dataset):
    """Wrapper to yield calibration batches for TensorRT INT8."""

    def __init__(
        self,
        dataloader: DataLoader,
        max_samples: int = 500,
        input_names: list[str] = None,
    ):
        self.dataloader = dataloader
        self.max_samples = max_samples
        self.input_names = input_names or ["video"]
        self._data = []
        self._prepare_data()

    def _prepare_data(self):
        """Pre-load calibration data."""
        count = 0
        for batch in self.dataloader:
            if count >= self.max_samples:
                break

            # Handle different input formats
            if isinstance(batch["video"], list):
                # SlowFast: list of [slow, fast] pathways
                for i, pathway in enumerate(batch["video"]):
                    for b in range(pathway.shape[0]):
                        if count >= self.max_samples:
                            break
                        self._data.append({
                            self.input_names[i]: pathway[b:b+1].numpy().astype(np.float32)
                        })
                        count += 1
            else:
                # Standard: single video tensor [B, C, T, H, W]
                for b in range(batch["video"].shape[0]):
                    if count >= self.max_samples:
                        break
                    self._data.append({
                        self.input_names[0]: batch["video"][b:b+1].numpy().astype(np.float32)
                    })
                    count += 1

        print(f"Prepared {len(self._data)} calibration samples")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def get_calibration_dataloader(
    cfg: dict,
    split: str = "train",
    max_samples: int = 500,
    batch_size: int = 1,
) -> DataLoader:
    """Build calibration dataloader from config."""
    # Get transforms for validation (no augmentation)
    transform_cfg = cfg.get("transforms", {})
    transform_cfg["is_train"] = False
    _ = get_transforms(transform_cfg)

    # Build datasets
    datasets = build_datasets({**cfg, "split": split})

    # Build dataloader (no domain mixing for calibration)
    from egoindustrial.data.unified_dataloader import ConcatDatasetWithDomain
    concat_dataset = ConcatDatasetWithDomain(datasets)

    return DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for calibration
        pin_memory=False,
        drop_last=False,
    )


def generate_calibration_data(
    cfg: dict,
    output_dir: str = "calibration_data",
    max_samples: int = 500,
    split: str = "train",
) -> str:
    """Generate and save calibration data as NPZ files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataloader = get_calibration_dataloader(cfg, split, max_samples)

    calibration_dataset = CalibrationDataset(dataloader, max_samples)

    # Save as NPZ for TensorRT calibration
    for i, sample in enumerate(calibration_dataset):
        if isinstance(sample, dict) and len(sample) == 1:
            # Single input
            arr = list(sample.values())[0]
            np.savez_compressed(output_path / f"calib_{i:05d}.npz", input=arr)
        else:
            # Multiple inputs
            np.savez_compressed(output_path / f"calib_{i:05d}.npz", **sample)

    print(f"Saved {len(calibration_dataset)} calibration samples to {output_path}")
    return str(output_path)


class TensorRTCalibrationDataset:
    """TensorRT IInt8Calibrator compatible dataset iterator."""

    def __init__(self, calibration_dir: str, batch_size: int = 1):
        self.calibration_dir = Path(calibration_dir)
        self.files = sorted(self.calibration_dir.glob("calib_*.npz"))
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        self.index = 0
        return self

    def __next__(self) -> list[np.ndarray]:
        if self.index >= len(self.files):
            raise StopIteration

        data = np.load(self.files[self.index])
        self.index += 1

        # Return as list of arrays (one per input)
        if "input" in data:
            return [data["input"]]
        else:
            return [data[key] for key in sorted(data.keys())]

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Hydra config path")
    parser.add_argument("--output", type=str, default="calibration_data")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    generate_calibration_data(
        OmegaConf.to_container(cfg.dataset, resolve=True),
        args.output,
        args.max_samples,
        args.split,
    )
