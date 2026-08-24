"""Build training datasets from verified pseudo-labels."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def build_training_dataset(
    verified_csv: str,
    original_annotations: str | None = None,
    output_dir: str = "data/training",
    split_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, str]:
    """Build training dataset from verified pseudo-labels.

    Args:
        verified_csv: Path to human-verified pseudo-labels CSV
        original_annotations: Optional path to original labeled data
        output_dir: Output directory for new dataset
        split_ratio: Train/val split ratio
        seed: Random seed

    Returns:
        Dict with paths to train/val annotation files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load verified pseudo-labels
    pseudo_df = pd.read_csv(verified_csv)

    # Load original annotations if provided
    if original_annotations and Path(original_annotations).exists():
        orig_df = pd.read_csv(original_annotations)
        # Combine
        combined_df = pd.concat([orig_df, pseudo_df], ignore_index=True)
    else:
        combined_df = pseudo_df

    # Shuffle and split
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(combined_df) * split_ratio)

    train_df = combined_df.iloc[:split_idx]
    val_df = combined_df.iloc[split_idx:]

    # Save
    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Generate class mapping files
    _generate_class_maps(combined_df, output_path)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    return {
        "train": str(train_path),
        "val": str(val_path),
        "class_maps": str(output_path / "class_maps.yaml"),
    }


def _generate_class_maps(df: pd.DataFrame, output_path: Path) -> None:
    """Generate class mapping YAML files."""
    # Verb classes
    verb_map = {}
    for _, row in df.iterrows():
        if "verb_class" in row and "verb_label" in row:
            verb_map[row["verb_class"]] = int(row["verb_label"])

    # Noun classes
    noun_map = {}
    for _, row in df.iterrows():
        if "noun_class" in row and "noun_label" in row:
            noun_map[row["noun_class"]] = int(row["noun_label"])

    # Action classes
    action_map = {}
    for _, row in df.iterrows():
        if "action_class" in row and "action_label" in row:
            action_map[row["action_class"]] = int(row["action_label"])

    # Save
    class_maps = {
        "verbs": verb_map,
        "nouns": noun_map,
        "actions": action_map,
    }

    with open(output_path / "class_maps.yaml", "w") as f:
        yaml.dump(class_maps, f)


def merge_datasets(
    dataset_configs: list[dict[str, Any]],
    output_dir: str = "data/merged",
) -> str:
    """Merge multiple dataset annotations into unified format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for cfg in dataset_configs:
        df = pd.read_csv(cfg["annotations"])
        df["dataset"] = cfg.get("name", "unknown")
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(output_path / "merged.csv", index=False)

    _generate_class_maps(merged, output_path)

    return str(output_path / "merged.csv")
