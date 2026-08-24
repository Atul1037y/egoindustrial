"""EgoIndustrial: Scalable Egocentric Action Recognition Pipeline."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from egoindustrial.data import (
    Assembly101Dataset,
    EpicKitchensDataset,
    HoloAssistDataset,
    UnifiedDataLoader,
)
from egoindustrial.inference import FastAPIServer, TensorRTEngine
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
    "UnifiedDataLoader",
    "VideoMAEv2",
    "MViTv2",
    "SlowFast",
    "InternVideo2",
    "get_model",
    "EgoIndustrialModule",
    "TensorRTEngine",
    "FastAPIServer",]
