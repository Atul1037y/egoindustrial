"""EgoIndustrial models module."""

from egoindustrial.models.head import MultiTaskHead, SingleTaskHead
from egoindustrial.models.internvideo2 import InternVideo2
from egoindustrial.models.mvitv2 import MViTv2
from egoindustrial.models.registry import get_model, list_models, register_model
from egoindustrial.models.slowfast import SlowFast
from egoindustrial.models.videomaev2 import VideoMAEv2

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "VideoMAEv2",
    "MViTv2",
    "SlowFast",
    "InternVideo2",
    "MultiTaskHead",
    "SingleTaskHead",
]
