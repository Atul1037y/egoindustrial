"""Assembly101 dataset loader for industrial assembly tasks."""

from pathlib import Path
from typing import Any

import pandas as pd

from egoindustrial.data.base_dataset import BaseEgocentricDataset


class Assembly101Dataset(BaseEgocentricDataset):
    """Assembly101 dataset for industrial assembly action recognition.

    Expected structure:
    root/
    ├── annotations/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   ├── action_classes.csv
    │   ├── verb_classes.csv
    │   └── noun_classes.csv
    └── videos/
        ├── train/
        ├── val/
        └── test/
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        clip_len: int = 16,
        frame_stride: int = 2,
        transforms=None,
    ):
        super().__init__(root, split, clip_len, frame_stride, transforms)

    def _load_annotations(self) -> list[dict[str, Any]]:
        annot_dir = self.root / "annotations"
        split_file = annot_dir / f"{self.split}.csv"

        if not split_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {split_file}")

        df = pd.read_csv(split_file)

        samples = []
        for _, row in df.iterrows():
            sample = {
                "video_id": row["video_id"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "verb_label": int(row["verb_class"]),
                "noun_label": int(row["noun_class"]),
                "action_label": int(row["action_class"]),
                "narration": row.get("narration", ""),
            }
            samples.append(sample)

        return samples

    def _build_label_maps(self) -> dict[str, dict[str, int]]:
        annot_dir = self.root / "annotations"

        verb_df = pd.read_csv(annot_dir / "verb_classes.csv")
        noun_df = pd.read_csv(annot_dir / "noun_classes.csv")
        action_df = pd.read_csv(annot_dir / "action_classes.csv")

        verb_map = dict(zip(verb_df["verb"], verb_df["class_id"]))
        noun_map = dict(zip(noun_df["noun"], noun_df["class_id"]))
        action_map = dict(zip(action_df["action"], action_df["class_id"]))

        return {"verb": verb_map, "noun": noun_map, "action": action_map}

    def _get_video_path(self, sample: dict[str, Any]) -> Path:
        return self.root / "videos" / self.split / f"{sample['video_id']}.mp4"
