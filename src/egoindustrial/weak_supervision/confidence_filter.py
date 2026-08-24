"""Confidence filtering and quality assessment for pseudo-labels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FilterConfig:
    """Configuration for confidence filtering."""
    verb_conf_threshold: float = 0.9
    noun_conf_threshold: float = 0.9
    action_conf_threshold: float = 0.8
    verb_entropy_threshold: float = 0.5
    noun_entropy_threshold: float = 0.5
    min_duration_frames: int = 8
    max_duration_frames: int = 100
    iou_threshold: float = 0.5  # For NMS


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy."""
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))


def filter_by_confidence(
    df: pd.DataFrame,
    config: FilterConfig,
) -> pd.DataFrame:
    """Filter pseudo-labels by confidence thresholds."""
    mask = (
        (df["verb_confidence"] >= config.verb_conf_threshold) &
        (df["noun_confidence"] >= config.noun_conf_threshold) &
        (df["action_confidence"] >= config.action_conf_threshold)
    )
    return df[mask].copy()


def filter_by_entropy(
    df: pd.DataFrame,
    config: FilterConfig,
    verb_probs_col: str = "verb_probs",
    noun_probs_col: str = "noun_probs",
) -> pd.DataFrame:
    """Filter by entropy (requires probability distributions)."""
    # If we only have confidence, skip entropy filtering
    if verb_probs_col not in df.columns:
        return df

    verb_entropy = df[verb_probs_col].apply(lambda x: compute_entropy(np.array(x)))
    noun_entropy = df[noun_probs_col].apply(lambda x: compute_entropy(np.array(x)))

    mask = (
        (verb_entropy <= config.verb_entropy_threshold) &
        (noun_entropy <= config.noun_entropy_threshold)
    )
    return df[mask].copy()


def filter_by_duration(
    df: pd.DataFrame,
    config: FilterConfig,
) -> pd.DataFrame:
    """Filter by segment duration."""
    df = df.copy()
    df["duration"] = df["end_frame"] - df["start_frame"]
    mask = (
        (df["duration"] >= config.min_duration_frames) &
        (df["duration"] <= config.max_duration_frames)
    )
    return df[mask]


def non_maximum_suppression(
    df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Remove overlapping pseudo-labels (NMS)."""
    if len(df) == 0:
        return df

    df = df.sort_values("action_confidence", ascending=False).reset_index(drop=True)
    keep = []

    for i, row in df.iterrows():
        overlap = False
        for j in keep:
            kept_row = df.loc[j]
            # Check temporal IoU
            start_i, end_i = row["start_frame"], row["end_frame"]
            start_j, end_j = kept_row["start_frame"], kept_row["end_frame"]

            intersection = max(0, min(end_i, end_j) - max(start_i, start_j))
            union = (end_i - start_i) + (end_j - start_j) - intersection
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                overlap = True
                break

        if not overlap:
            keep.append(i)

    return df.loc[keep].reset_index(drop=True)


def balance_classes(
    df: pd.DataFrame,
    max_per_class: int = 1000,
    class_col: str = "action_label",
) -> pd.DataFrame:
    """Balance classes by subsampling frequent classes."""
    if len(df) == 0:
        return df

    counts = df[class_col].value_counts()
    sampled = []

    for cls, count in counts.items():
        cls_df = df[df[class_col] == cls]
        if count > max_per_class:
            cls_df = cls_df.sample(n=max_per_class, random_state=42)
        sampled.append(cls_df)

    return pd.concat(sampled, ignore_index=True)


def apply_all_filters(
    df: pd.DataFrame,
    config: FilterConfig,
    apply_nms: bool = True,
    balance: bool = True,
    max_per_class: int = 1000,
) -> pd.DataFrame:
    """Apply all filtering steps."""
    print(f"Initial labels: {len(df)}")

    df = filter_by_confidence(df, config)
    print(f"After confidence filter: {len(df)}")

    df = filter_by_entropy(df, config)
    print(f"After entropy filter: {len(df)}")

    df = filter_by_duration(df, config)
    print(f"After duration filter: {len(df)}")

    if apply_nms:
        df = non_maximum_suppression(df, config.iou_threshold)
        print(f"After NMS: {len(df)}")

    if balance:
        df = balance_classes(df, max_per_class)
        print(f"After class balancing: {len(df)}")

    return df


def load_pseudo_labels(csv_path: str) -> pd.DataFrame:
    """Load pseudo-labels from CSV."""
    return pd.read_csv(csv_path)


def save_filtered_labels(df: pd.DataFrame, output_path: str) -> None:
    """Save filtered pseudo-labels."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} filtered labels to {output_path}")
