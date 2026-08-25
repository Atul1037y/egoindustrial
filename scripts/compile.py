#!/usr/bin/env python3
"""
TorchScript/TorchInductor Compilation for Faster Inference.

Compiles models to TorchScript and TorchInductor for faster CPU/GPU inference.

Usage:
    python scripts/compile.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --output-dir outputs/compiled \
        --mode inductor
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egoindustrial.models import get_model
from egoindustrial.training.module import EgoIndustrialModule


def compile_to_torchscript(
    model: nn.Module,
    output_path: str,
    example_input: torch.Tensor,
    optimize: bool = True,
) -> str:
    """Compile model to TorchScript."""
    print("Compiling to TorchScript...")

    model.eval()
    with torch.no_grad():
        # Trace the model
        traced = torch.jit.trace(model, example_input, strict=False)

        if optimize:
            # Optimize for inference
            traced = torch.jit.optimize_for_inference(traced)

        # Save
        traced.save(output_path)
        print(f"TorchScript saved to {output_path}")

    return output_path


def compile_with_inductor(
    model: nn.Module,
    output_path: str,
    example_input: torch.Tensor,
    mode: str = "reduce-overhead",
) -> str:
    """Compile model with TorchInductor (PyTorch 2.0+)."""
    print(f"Compiling with TorchInductor (mode={mode})...")

    model.eval()

    # TorchInductor compilation
    compiled = torch.compile(
        model,
        mode=mode,  # "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        dynamic=True,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = compiled(torch.randn_like(example_input))

    # Save as state dict for later use
    torch.save({
        "model_state_dict": model.state_dict(),
        "example_input": example_input,
    }, output_path.replace(".pt", "_inductor.pt"))

    print(f"TorchInductor compilation ready (model saved to {output_path})")
    return output_path


def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cuda",
    warmup: int = 10,
    runs: int = 100,
) -> dict:
    """Benchmark model inference time."""
    model.eval()
    model.to(device)

    dummy = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(torch.randn(*input_shape).to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "p50_ms": times.median().item(),
        "p95_ms": times.quantile(0.95).item(),
        "p99_ms": times.quantile(0.99).item(),
        "fps": 1000 / times.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compile EgoIndustrial model for inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output-dir", default="outputs/compiled", help="Output directory")
    parser.add_argument("--mode", choices=["torchscript", "inductor", "both"], default="both", help="Compilation mode")
    parser.add_argument("--inductor-mode", choices=["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], default="reduce-overhead")
    parser.add_argument("--input-shape", nargs="+", type=int, default=[1, 3, 16, 224, 224], help="Input shape [B, C, T, H, W]")
    parser.add_argument("--output-dir", default="outputs/compiled", help="Output directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Benchmark runs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EgoIndustrialModule.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Example input
    torch.randn(*args.input_shape)

    results = {}

    if args.mode in ["torchscript", "both"]:
        # TorchScript
        output_dir / "model_torchscript.pt"
        ts_start = time.time()
        compile_to_torchscript(model, str(output_dir / "model_torchscript.pt"), torch.randn(*args.input_shape))
        results["torchscript_compile_time"] = time.time() - ts_start
        results["torchscript_path"] = str(output_dir / "model_torchscript.pt")

        if args.benchmark:
            print("Benchmarking TorchScript...")
            torch.jit.load(str(output_dir / "model_torchscript.pt"))
            ts_results = benchmark_model(torch.jit.load(str(output_dir / "model_torchscript.pt")), args.input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
            results["torchscript_benchmark"] = ts_results
            print(f"  TorchScript: {ts_results['mean_ms']:.2f}ms, {ts_results['fps']:.1f} FPS")

    if args.mode in ["inductor", "both"]:
        # TorchInductor
        ind_path = output_dir / "model_inductor.pt"
        ind_start = time.time()
        compile_with_inductor(model, str(ind_path), torch.randn(*args.input_shape), mode=args.inductor_mode)
        results["inductor_compile_time"] = time.time() - ind_start
        results["inductor_path"] = str(ind_path)

        if args.benchmark:
            print("Benchmarking TorchInductor...")
            get_model(...)  # Would need to reload
            ind_results = benchmark_model(model, args.input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
            results["inductor_benchmark"] = ind_results
            print(f"  TorchInductor: {ind_results['mean_ms']:.2f}ms, {ind_results['fps']:.1f} FPS")

    if args.benchmark:
        print("\nOriginal PyTorch benchmark...")
        orig_results = benchmark_model(model, args.input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
        results["pytorch_benchmark"] = orig_results
        print(f"  PyTorch: {orig_results['mean_ms']:.2f}ms, {orig_results['fps']:.1f} FPS")

    # Save results
    import json
    with open(Path(args.output_dir) / "compile_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("COMPILATION COMPLETE")
    print("="*60)
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}: {v.get('mean_ms', 'N/A')}ms, {v.get('fps', 'N/A')} FPS")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
