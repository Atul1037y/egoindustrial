"""Abstract base dataset for egocentric action recognition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseEgocentricDataset(Dataset, ABC):
    """Base class for all egocentric datasets."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        clip_len: int = 16,
        frame_stride: int = 2,
        transforms=None,
    ):
        self.root = Path(root)
        self.split = split
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.transforms = transforms

        self.samples = self._load_annotations()
        self.label_maps = self._build_label_maps()

    @abstractmethod
    def _load_annotations(self) -> list[dict[str, Any]]:
        """Load dataset annotations. Returns list of sample dicts."""
        pass

    @abstractmethod
    def _build_label_maps(self) -> dict[str, dict[str, int]]:
        """Build label -> index mappings. Returns {'verb': {}, 'noun': {}, 'action': {}}."""
        pass

    @abstractmethod
    def _get_video_path(self, sample: dict[str, Any]) -> Path:
        """Get full video path for a sample."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        video_path = self._get_video_path(sample)

        frames = self._load_video(video_path)

        if self.transforms:
            frames = self.transforms(frames)

        return {
            "video": frames,
            "verb_label": sample.get("verb_label", -1),
            "noun_label": sample.get("noun_label", -1),
            "action_label": sample.get("action_label", -1),
            "video_id": sample.get("video_id", ""),
            "start_frame": sample.get("start_frame", 0),
            "end_frame": sample.get("end_frame", 0),
            "dataset": self.__class__.__name__,
        }

    def _load_video(self, path: Path) -> torch.Tensor:
        """Load video frames. Override for custom loading."""
        try:
            import torchvision.io as tv_io

            video, _, _ = tv_io.read_video(str(path), pts_unit="sec", output_format="TCHW")
            return self._sample_frames(video)
        except Exception:
            import cv2

            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                raise ValueError(f"No frames loaded from {path}")
            video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            return self._sample_frames(video)

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Uniform temporal sampling."""
        total_frames = video.shape[0]
        required_frames = self.clip_len * self.frame_stride

        if total_frames >= required_frames:
            start = (total_frames - required_frames) // 2
            indices = torch.arange(start, start + required_frames, self.frame_stride)
        else:
            indices = torch.linspace(0, total_frames - 1, self.clip_len).long()

        return video[indices]

    def get_label_names(self) -> dict[str, list[str]]:
        """Get index -> label name mappings."""
        return {
            "verb": {v: k for k, v in self.label_maps["verb"].items()},
            "noun": {v: k for k, v in self.label_maps["noun"].items()},
            "action": {v: k for k, v in self.label_maps["action"].items()},
        }