"""EgoIndustrial: Scalable Egocentric Action Recognition Pipeline."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from egoindustrial.data import (
    Assembly101Dataset,
    ConcatDatasetWithDomain,
    EpicKitchensDataset,
    HoloAssistDataset,
)
from egoindustrial.inference import TensorRTEngine, create_app, run_server
from egoindustrial.models import (
    InternVideo2,
    MViTv2,
    SlowFast,
    VideoMAEv2,
    get_model,
)
from egoindustrial.training import EgoIndustrialModule

__all__ = [
    "EpicKitchensDataset",
    "Assembly101Dataset",
    "HoloAssistDataset",
    "ConcatDatasetWithDomain",
    "VideoMAEv2",
    "MViTv2",
    "SlowFast",
    "InternVideo2",
    "get_model",
    "EgoIndustrialModule",
    "TensorRTEngine",
    "create_app",
    "run_server",
]
