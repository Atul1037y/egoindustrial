"""TensorRT engine builder with INT8 calibration support."""

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import pycuda.driver as cuda
    import tensorrt as trt

try:
    import pycuda.driver as cuda
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    cuda = None
    trt = None

if HAS_TENSORRT:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
else:
    TRT_LOGGER = None


class TensorRTEngine:
    """TensorRT engine wrapper for inference."""

    def __init__(self, engine_path: str):
        if not HAS_TENSORRT:
            raise ImportError(
                "TensorRTEngine requires tensorrt and pycuda. "
                "Install with: pip install tensorrt pycuda"
            )
        self.engine_path = engine_path
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self._allocate_buffers()

    def _load_engine(self) -> "trt.ICudaEngine":
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def infer(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Run inference with numpy inputs."""
        # Copy inputs to device
        for i, inp in enumerate(inputs):
            np.copyto(self.inputs[i]["host"], inp.ravel())
            cuda.memcpy_htod_async(self.inputs[i]["device"], self.inputs[i]["host"], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        return [
            out["host"].reshape(self.engine.get_binding_shape(b))
            for b, out in zip(
                [b for b in self.engine if not self.engine.binding_is_input(b)], self.outputs
            )
        ]

    def infer_torch(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """Run inference with torch tensors."""
        np_inputs = [i.cpu().numpy().astype(np.float32) for i in inputs]
        np_outputs = self.infer(*np_inputs)
        return [torch.from_numpy(o) for o in np_outputs]


if HAS_TENSORRT:

    class CalibrationDataset:
        """Calibration dataset for INT8 quantization."""

        def __init__(self, dataloader, max_batches: int = 100):
            self.dataloader = dataloader
            self.max_batches = max_batches
            self.batch_idx = 0

        def __iter__(self):
            self.batch_idx = 0
            return self

        def __next__(self):
            if self.batch_idx >= self.max_batches:
                raise StopIteration
            try:
                batch = next(self.dataloader)
            except StopIteration:
                raise StopIteration
            self.batch_idx += 1

            # Return inputs as list of numpy arrays
            if isinstance(batch["video"], list):
                return [v.numpy().astype(np.float32) for v in batch["video"]]
            return [batch["video"].numpy().astype(np.float32)]

        def __len__(self):
            return min(self.max_batches, len(self.dataloader))

    class Int8Calibrator(trt.IInt8EntropyCalibrator2):
        """INT8 entropy calibrator."""

        def __init__(self, calibration_dataset: CalibrationDataset, cache_file: str = ""):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.dataset = calibration_dataset
            self.cache_file = cache_file
            self.device_input = None
            self.iterator = iter(calibration_dataset)

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: list[str], p_str: Any = None) -> list[int]:
            try:
                batch = next(self.iterator)
                if not isinstance(batch, list):
                    batch = [batch]

                if self.device_input is None:
                    self.device_input = cuda.mem_alloc(batch[0].nbytes)

                cuda.memcpy_htod(self.device_input, batch[0])
                return [int(self.device_input)]
            except StopIteration:
                return None

        def read_calibration_cache(self) -> bytes | None:
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes):
            if self.cache_file:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",  # fp32, fp16, int8
    max_batch_size: int = 32,
    max_workspace_size: int = 1 << 30,  # 1GB
    calibration_dataloader=None,
    calibration_cache: str = "calibration.cache",
    input_shapes: dict | None = None,
) -> str:
    """Build TensorRT engine from ONNX.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output engine path
        precision: fp32, fp16, or int8
        max_batch_size: Maximum batch size for dynamic shapes
        max_workspace_size: Max workspace in bytes
        calibration_dataloader: DataLoader for INT8 calibration
        calibration_cache: Cache file for calibration
        input_shapes: Dict of input_name -> (min, opt, max) shapes

    Returns:
        Path to built engine
    """
    if not HAS_TENSORRT:
        raise ImportError("build_tensorrt_engine requires tensorrt and pycuda")

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # Precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_dataloader is None:
            raise ValueError("INT8 requires calibration_dataloader")
        config.int8_calibrator = Int8Calibrator(
            CalibrationDataset(calibration_dataloader), calibration_cache
        )

    # Parse ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    # Dynamic shapes
    if input_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # Build engine
    print(f"Building TensorRT engine ({precision})...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build engine")

    # Save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"Engine saved to {engine_path}")
    return engine_path


def benchmark_engine(
    engine_path: str,
    input_shapes: list[tuple],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Benchmark TensorRT engine latency and throughput."""
    if not HAS_TENSORRT:
        raise ImportError("benchmark_engine requires tensorrt and pycuda")

    engine = TensorRTEngine(engine_path)

    # Generate dummy inputs
    dummy_inputs = [np.random.randn(*s).astype(np.float32) for s in input_shapes]

    # Warmup
    for _ in range(num_warmup):
        engine.infer(*dummy_inputs)

    # Benchmark
    import time

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        engine.infer(*dummy_inputs)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": 1000 / times.mean(),
        "batch_size": input_shapes[0][0] if input_shapes else 1,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=str, help="ONNX model path")
    parser.add_argument("engine", type=str, help="Output engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--workspace", type=int, default=1 << 30)
    args = parser.parse_args()

    # Example input shapes for VideoMAE: [B, C, T, H, W]
    input_shapes = {
        "video": (
            [1, 3, 16, 224, 224],
            [args.max_batch, 3, 16, 224, 224],
            [args.max_batch, 3, 16, 224, 224],
        )
    }

    build_tensorrt_engine(
        args.onnx,
        args.engine,
        precision=args.precision,
        max_batch_size=args.max_batch,
        max_workspace_size=args.workspace,
        input_shapes=input_shapes,
    )
