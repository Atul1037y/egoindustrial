"""EgoIndustrial inference module."""

from egoindustrial.inference.export_onnx import export_to_onnx, validate_onnx

# TensorRT components (optional)
try:
    from egoindustrial.inference.tensorrt_engine import (
        CalibrationDataset,
        Int8Calibrator,
        TensorRTEngine,
        benchmark_engine,
        build_tensorrt_engine,
    )
except ImportError:
    TensorRTEngine = None
    CalibrationDataset = None
    Int8Calibrator = None
    build_tensorrt_engine = None
    benchmark_engine = None

from egoindustrial.inference.models import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    enrich_response,
)
from egoindustrial.inference.server import (
    DynamicBatcher,
    create_app,
    run_server,
)

__all__ = [
    "export_to_onnx",
    "validate_onnx",
    "TensorRTEngine",
    "CalibrationDataset",
    "Int8Calibrator",
    "build_tensorrt_engine",
    "benchmark_engine",
    "DynamicBatcher",
    "create_app",
    "run_server",
    "InferenceRequest",
    "InferenceResponse",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "HealthResponse",
    "ModelInfo",
    "enrich_response",
]
