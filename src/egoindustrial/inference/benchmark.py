"""Latency and throughput benchmarking for ONNX/TensorRT/PyTorch models."""

import json
import time
from typing import Any

import numpy as np
import torch


class BenchmarkRunner:
    """Unified benchmarking for PyTorch, ONNX Runtime, and TensorRT."""

    def __init__(
        self,
        model: Any,
        input_shapes: list[tuple],
        device: str = "cuda",
        warmup: int = 10,
        runs: int = 100,
    ):
        self.model = model
        self.input_shapes = input_shapes
        self.device = device
        self.warmup = warmup
        self.runs = runs

    def _generate_inputs(self) -> list[np.ndarray]:
        return [np.random.randn(*s).astype(np.float32) for s in self.input_shapes]

    def _to_device(self, inputs: list[np.ndarray]) -> list[torch.Tensor]:
        return [torch.from_numpy(x).to(self.device) for x in inputs]

    def benchmark_pytorch(self) -> dict[str, float]:
        """Benchmark PyTorch model."""
        self.model.eval()
        self.model.to(self.device)

        inputs_np = self._generate_inputs()
        inputs_torch = self._to_device(inputs_np)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = self.model(*inputs_torch) if len(inputs_torch) > 1 else self.model(inputs_torch[0])
                torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.runs):
                start = time.perf_counter()
                _ = self.model(*inputs_torch) if len(inputs_torch) > 1 else self.model(inputs_torch[0])
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        return self._compute_stats(times)

    def benchmark_onnx(self, onnx_path: str, providers: list[str] = None) -> dict[str, float]:
        """Benchmark ONNX Runtime."""
        import onnxruntime as ort

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)

        inputs_np = self._generate_inputs()
        input_names = [inp.name for inp in session.get_inputs()]
        feed_dict = {name: inputs_np[i] for i, name in enumerate(input_names)}

        # Warmup
        for _ in range(self.warmup):
            session.run(None, feed_dict)

        # Benchmark
        times = []
        for _ in range(self.runs):
            start = time.perf_counter()
            session.run(None, feed_dict)
            times.append((time.perf_counter() - start) * 1000)

        return self._compute_stats(times)

    def benchmark_tensorrt(self, engine_path: str) -> dict[str, float]:
        """Benchmark TensorRT engine."""
        from egoindustrial.inference.tensorrt_engine import TensorRTEngine

        engine = TensorRTEngine(engine_path)
        inputs_np = self._generate_inputs()

        # Warmup
        for _ in range(self.warmup):
            engine.infer(*inputs_np)

        # Benchmark
        times = []
        for _ in range(self.runs):
            start = time.perf_counter()
            engine.infer(*inputs_np)
            times.append((time.perf_counter() - start) * 1000)

        return self._compute_stats(times)

    def _compute_stats(self, times: list[float]) -> dict[str, float]:
        times = np.array(times)
        batch_size = self.input_shapes[0][0]

        return {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": batch_size * 1000 / times.mean(),
            "throughput_fps": batch_size * 1000 / times.mean(),
            "batch_size": batch_size,
        }


def run_full_benchmark(
    pytorch_model: torch.nn.Module | None = None,
    onnx_path: str | None = None,
    engine_path: str | None = None,
    input_shapes: list[tuple] = None,
    device: str = "cuda",
    warmup: int = 10,
    runs: int = 100,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run comprehensive benchmark across all available backends."""
    if input_shapes is None:
        input_shapes = [(1, 3, 16, 224, 224)]

    runner = BenchmarkRunner(
        model=pytorch_model,
        input_shapes=input_shapes,
        device=device,
        warmup=warmup,
        runs=runs,
    )

    results = {
        "input_shapes": input_shapes,
        "device": device,
        "warmup": warmup,
        "runs": runs,
    }

    if pytorch_model is not None:
        print("Benchmarking PyTorch...")
        results["pytorch"] = runner.benchmark_pytorch()
        print(f"  PyTorch: {results['pytorch']['mean_ms']:.2f} ms, {results['pytorch']['fps']:.1f} FPS")

    if onnx_path:
        print("Benchmarking ONNX Runtime...")
        results["onnx"] = runner.benchmark_onnx(onnx_path)
        print(f"  ONNX: {results['onnx']['mean_ms']:.2f} ms, {results['onnx']['fps']:.1f} FPS")

    if engine_path:
        print("Benchmarking TensorRT...")
        results["tensorrt"] = runner.benchmark_tensorrt(engine_path)
        print(f"  TensorRT: {results['tensorrt']['mean_ms']:.2f} ms, {results['tensorrt']['fps']:.1f} FPS")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_json}")

    return results


def profile_memory(model: torch.nn.Module, input_shapes: list[tuple]) -> dict[str, float]:
    """Profile GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    inputs = [torch.randn(*s).cuda() for s in input_shapes]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    model.cuda()

    with torch.no_grad():
        _ = model(*inputs) if len(inputs) > 1 else model(inputs[0])

    allocated = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9

    return {
        "peak_allocated_gb": allocated,
        "peak_reserved_gb": reserved,
    }


def compare_accuracy(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_shapes: list[tuple],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare numerical accuracy between PyTorch and ONNX."""
    import onnxruntime as ort

    pytorch_model.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    results = {}
    for i in range(10):  # Test 10 random inputs
        torch_inputs = [torch.randn(*s) for s in input_shapes]
        ort_inputs = {session.get_inputs()[j].name: t.numpy() for j, t in enumerate(torch_inputs)}

        with torch.no_grad():
            torch_out = pytorch_model(*torch_inputs) if len(torch_inputs) > 1 else pytorch_model(torch_inputs[0])

        ort_outs = session.run(None, ort_inputs)

        if isinstance(torch_out, dict):
            torch_out = [torch_out["verb"], torch_out["noun"], torch_out["action"]]
        elif not isinstance(torch_out, (list, tuple)):
            torch_out = [torch_out]

        for j, (t, o) in enumerate(zip(torch_out, ort_outs)):
            t_np = t.numpy() if isinstance(t, torch.Tensor) else t
            max_diff = np.abs(t_np - o).max()
            mean_diff = np.abs(t_np - o).mean()
            close = np.allclose(t_np, o, rtol=rtol, atol=atol)

            key = f"output_{j}_run_{i}"
            results[key] = {"max_diff": float(max_diff), "mean_diff": float(mean_diff), "close": bool(close)}

    all_close = all(r["close"] for r in results.values())
    results["summary"] = {"all_close": all_close}
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-ckpt", type=str, help="PyTorch checkpoint")
    parser.add_argument("--onnx", type=str, help="ONNX model path")
    parser.add_argument("--engine", type=str, help="TensorRT engine path")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 16, 224, 224])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    input_shapes = [tuple(args.input_shape)]

    # Load PyTorch model if provided
    pytorch_model = None
    if args.pytorch_ckpt:
        from egoindustrial.training.module import EgoIndustrialModule
        pytorch_model = EgoIndustrialModule.load_from_checkpoint(args.pytorch_ckpt)

    all_results = {}
    for bs in args.batch_sizes:
        shapes = [(bs, *args.input_shape[1:])]
        print(f"\n=== Batch size {bs} ===")
        results = run_full_benchmark(
            pytorch_model=pytorch_model,
            onnx_path=args.onnx,
            engine_path=args.engine,
            input_shapes=shapes,
            device=args.device,
            warmup=args.warmup,
            runs=args.runs,
        )
        all_results[f"batch_{bs}"] = results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output}")
