"""Export verified pseudo-labels."""

from pathlib import Path

import pandas as pd
from egoindustrial.weak_supervision.ui.state import SessionState


def export_verified_labels(state: SessionState, output_path: str | None = None) -> str:
    """Export verified and corrected pseudo-labels."""
    df = state.get_verified_dataframe()

    if df.empty:
        raise ValueError("No verified labels to export")

    if output_path is None:
        output_path = "verified_pseudo_labels.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    return str(output_path)


def export_for_training(
    state: SessionState,
    output_dir: str = "data/weak_supervision",
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, str]:
    """Export in training-ready format with train/val split."""
    df = state.get_verified_dataframe()

    if df.empty:
        raise ValueError("No verified labels to export")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # Save in standard format
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)

    # Generate class maps
    _generate_class_maps(df, output_path)

    return {
        "train": str(output_path / "train.csv"),
        "val": str(output_path / "val.csv"),
    }


def _generate_class_maps(df: pd.DataFrame, output_path: Path) -> None:
    import yaml

    verb_map = {}
    noun_map = {}
    action_map = {}

    for _, row in df.iterrows():
        if "verb_class" in row and "verb_label" in row:
            verb_map[row["verb_class"]] = int(row["verb_label"])
        if "noun_class" in row and "noun_label" in row:
            noun_map[row["noun_class"]] = int(row["noun_label"])
        if "action_class" in row and "action_label" in row:
            action_map[row["action_class"]] = int(row["action_label"])

    class_maps = {"verbs": verb_map, "nouns": noun_map, "actions": action_map}

    with open(output_path / "class_maps.yaml", "w") as f:
        yaml.dump(class_maps, f)


def merge_with_original(
    verified_csv: str,
    original_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Merge verified pseudo-labels with original annotations."""
    verified = pd.read_csv(verified_csv)
    original = pd.read_csv(original_csv)

    # Ensure same columns
    common_cols = list(set(verified.columns) & set(original.columns))
    verified = verified[common_cols]
    original = original[common_cols]

    merged = pd.concat([original, verified], ignore_index=True)
    merged.to_csv(output_csv, index=False)

    return merged
