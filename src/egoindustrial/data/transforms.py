"""Video transforms/augmentations for egocentric action recognition."""

import random
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class VideoTransforms:
    """Composable video transforms supporting spatial + temporal augmentations."""

    def __init__(
        self,
        clip_len: int = 16,
        crop_size: int = 224,
        resize_size: int = 256,
        is_train: bool = True,
        model_type: str = "videomae",  # videomae, slowfast, mvit
    ):
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.is_train = is_train
        self.model_type = model_type

        self.spatial_transform = self._build_spatial_transform()
        self.temporal_transform = self._build_temporal_transform()

    def _build_spatial_transform(self) -> A.Compose:
        if self.is_train:
            if self.model_type in ("videomae", "mvit"):
                return A.Compose(
                    [
                        A.SmallestMaxSize(max_size=self.resize_size, p=1.0),
                        A.RandomCrop(height=self.crop_size, width=self.crop_size, p=1.0),
                        A.HorizontalFlip(p=0.5),
                        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                        A.GaussNoise(var_limit=(10, 50), p=0.3),
                        A.GaussianBlur(blur_limit=3, p=0.3),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                        ToTensorV2(),
                    ],
                    additional_targets={f"image{i}": "image" for i in range(self.clip_len)},
                )
            elif self.model_type == "slowfast":
                return A.Compose(
                    [
                        A.SmallestMaxSize(max_size=self.resize_size, p=1.0),
                        A.RandomCrop(height=self.crop_size, width=self.crop_size, p=1.0),
                        A.HorizontalFlip(p=0.5),
                        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                        A.Normalize(
                            mean=[0.45, 0.45, 0.45],
                            std=[0.225, 0.225, 0.225],
                        ),
                        ToTensorV2(),
                    ],
                    additional_targets={f"image{i}": "image" for i in range(self.clip_len)},
                )
        else:
            return A.Compose(
                [
                    A.SmallestMaxSize(max_size=self.resize_size, p=1.0),
                    A.CenterCrop(height=self.crop_size, width=self.crop_size, p=1.0),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(self.clip_len)},
            )

    def _build_temporal_transform(self):
        if self.is_train:
            return TemporalRandomCrop(self.clip_len)
        return TemporalCenterCrop(self.clip_len)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply transforms to video frames [T, C, H, W]."""
        T, C, H, W = frames.shape

        # Convert to numpy for albumentations
        frames_np = frames.permute(0, 2, 3, 1).numpy()  # [T, H, W, C]

        # Apply temporal sampling
        frames_np = self.temporal_transform(frames_np)

        # Apply spatial transforms to each frame
        transformed = {}
        for i in range(frames_np.shape[0]):
            transformed[f"image{i}"] = frames_np[i]

        augmented = self.spatial_transform(**transformed)

        # Stack back to tensor [T, C, H, W]
        out_frames = torch.stack([augmented[f"image{i}"] for i in range(self.clip_len)])

        # Model-specific formatting
        if self.model_type == "slowfast":
            # SlowFast expects [C, T, H, W] with two pathways
            out_frames = out_frames.permute(1, 0, 2, 3)  # [C, T, H, W]
            fast_pathway = out_frames
            slow_pathway = torch.index_select(
                fast_pathway, 1, torch.linspace(0, self.clip_len - 1, self.clip_len // 4).long()
            )
            return [slow_pathway, fast_pathway]

        return out_frames.permute(1, 0, 2, 3)  # [C, T, H, W]


class TemporalRandomCrop:
    """Random temporal crop."""

    def __init__(self, clip_len: int):
        self.clip_len = clip_len

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T = frames.shape[0]
        if T <= self.clip_len:
            return frames
        start = random.randint(0, T - self.clip_len)
        return frames[start : start + self.clip_len]


class TemporalCenterCrop:
    """Center temporal crop."""

    def __init__(self, clip_len: int):
        self.clip_len = clip_len

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T = frames.shape[0]
        if T <= self.clip_len:
            return frames
        start = (T - self.clip_len) // 2
        return frames[start : start + self.clip_len]


def get_transforms(cfg: dict[str, Any]) -> VideoTransforms:
    """Factory function for Hydra config."""
    return VideoTransforms(
        clip_len=cfg.get("clip_len", 16),
        crop_size=cfg.get("crop_size", 224),
        resize_size=cfg.get("resize_size", 256),
        is_train=cfg.get("is_train", True),
        model_type=cfg.get("model_type", "videomae"),
    )
