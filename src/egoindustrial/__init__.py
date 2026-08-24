"""EgoIndustrial: Scalable Egocentric Action Recognition Pipeline."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from egoindustrial.data import (
    EpicKitchensDataset,
    Assembly101Dataset,
    HoloAssistDataset,
    UnifiedDataLoader,
)
from egoindustrial.models import (
    VideoMAEv2,
    MViTv2,
    SlowFast,
    InternVideo2,
    get_model,
)
from egoindustrial.training import EgoIndustrialModule
from egoindustrial.inference import TensorRTEngine, FastAPIServer

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