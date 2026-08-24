"""EgoIndustrial weak supervision module."""

from egoindustrial.weak_supervision.confidence_filter import (
    FilterConfig,
    apply_all_filters,
    balance_classes,
    filter_by_confidence,
    filter_by_duration,
    filter_by_entropy,
    load_pseudo_labels,
    non_maximum_suppression,
    save_filtered_labels,
)
from egoindustrial.weak_supervision.dataset_builder import build_training_dataset, merge_datasets
from egoindustrial.weak_supervision.pseudo_labeler import (
    PseudoLabeler,
    create_pseudo_labeler_from_checkpoint,
)

__all__ = [
    "PseudoLabeler",
    "create_pseudo_labeler_from_checkpoint",
    "FilterConfig",
    "filter_by_confidence",
    "filter_by_entropy",
    "filter_by_duration",
    "non_maximum_suppression",
    "balance_classes",
    "apply_all_filters",
    "load_pseudo_labels",
    "save_filtered_labels",
    "build_training_dataset",
    "merge_datasets",
]
