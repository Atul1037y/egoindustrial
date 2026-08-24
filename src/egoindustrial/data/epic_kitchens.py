"""EPIC-KITCHENS-100 dataset loader."""

from pathlib import Path
from typing import Any

import pandas as pd

from egoindustrial.data.base_dataset import BaseEgocentricDataset


class EpicKitchensDataset(BaseEgocentricDataset):
    """EPIC-KITCHENS-100 egocentric action recognition dataset.

    Expected structure:
    root/
    ├── annotations/
    │   ├── EPIC_100_train.csv
    │   ├── EPIC_100_val.csv
    │   ├── EPIC_100_test.csv
    │   ├── EPIC_100_noun_classes.csv
    │   └── EPIC_100_verb_classes.csv
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
        use_action_labels: bool = True,
    ):
        self.use_action_labels = use_action_labels
        super().__init__(root, split, clip_len, frame_stride, transforms)

    def _load_annotations(self) -> list[dict[str, Any]]:
        annot_dir = self.root / "annotations"
        split_file = annot_dir / f"EPIC_100_{self.split}.csv"

        if not split_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {split_file}")

        df = pd.read_csv(split_file)

        # Filter valid segments
        df = df[(df["stop_frame"] - df["start_frame"]) > 0].reset_index(drop=True)

        samples = []
        for _, row in df.iterrows():
            sample = {
                "video_id": row["video_id"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["stop_frame"]),
                "verb_label": int(row["verb_class"]),
                "noun_label": int(row["noun_class"]),
                "narration": row["narration"],
            }
            if self.use_action_labels and "action_class" in row:
                sample["action_label"] = int(row["action_class"])
            else:
                sample["action_label"] = sample["verb_label"] * 1000 + sample["noun_label"]
            samples.append(sample)

        return samples

    def _build_label_maps(self) -> dict[str, dict[str, int]]:
        annot_dir = self.root / "annotations"

        verb_df = pd.read_csv(annot_dir / "EPIC_100_verb_classes.csv")
        noun_df = pd.read_csv(annot_dir / "EPIC_100_noun_classes.csv")

        verb_map = dict(zip(verb_df["key"], verb_df["class_id"]))
        noun_map = dict(zip(noun_df["key"], noun_df["class_id"]))

        # Action map (verb_noun combination)
        action_map = {}
        for v_key, v_id in verb_map.items():
            for n_key, n_id in noun_map.items():
                action_map[f"{v_key}_{n_key}"] = v_id * 1000 + n_id

        return {"verb": verb_map, "noun": noun_map, "action": action_map}

    def _get_video_path(self, sample: dict[str, Any]) -> Path:
        video_id = sample["video_id"]
        participant_id = video_id.split("_")[0]  # P01, P02, etc.
        return self.root / "videos" / self.split / participant_id / f"{video_id}.MP4"
