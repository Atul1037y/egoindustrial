#!/usr/bin/env python3
"""Export and benchmark script for EgoIndustrial models."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from egoindustrial.inference.benchmark import compare_accuracy, run_full_benchmark
from egoindustrial.inference.export_onnx import export_to_onnx, validate_onnx
from egoindustrial.inference.tensorrt_engine import build_tensorrt_engine
from egoindustrial.training.module import EgoIndustrialModule


def export_pipeline(
    checkpoint: str,
    output_dir: str,
    opset: int = 17,
    simplify: bool = True,
    precision: str = "int8",
    max_batch: int = 32,
    workspace: int = 1 << 30,
    calibration_data: str | None = None,
) -> dict:
    """Full export pipeline: checkpoint -> ONNX -> TensorRT -> benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Export to ONNX
    print("="*60)
    print("Step 1: Exporting to ONNX...")
    print("="*60)
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(
        checkpoint_path=checkpoint,
        output_path=str(onnx_path),
        opset_version=opset,
        simplify=True,
    )
    validate_onnx(str(onnx_path), [1, 3, 16, 224, 224])
    results["onnx_path"] = str(onnx_path)

    # 2. Build TensorRT engine (FP16)
    print("="*60)
    print(f"Step 2: Building TensorRT engine ({precision})...")
    print("="*60)

    engine_suffix = "fp16" if precision == "fp16" else "int8"
    engine_path = output_dir / f"model_{engine_suffix}.engine"

    input_shapes = {
        "video": ([1, 3, 16, 224, 224], [max_batch, 3, 16, 224, 224], [max_batch, 3, 16, 224, 224])
    }

    calibration_cache = str(Path(output_dir) / "calibration.cache")
    calibration_dataloader = None

    if precision == "int8":
        if calibration_data:
            # Build calibration dataloader from provided data
            # Would need config for this
            pass
        else:
            print("Warning: INT8 requested but no calibration data provided. Using FP16 instead.")
            precision = "fp16"
            engine_path = output_dir / "model_fp16.engine"

    build_tensorrt_engine(
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        precision=precision,
        max_batch_size=max_batch,
        max_workspace_size=workspace,
        calibration_dataloader=calibration_dataloader,
        calibration_cache=calibration_cache,
        input_shapes=input_shapes,
    )
    results["engine_path"] = str(engine_path)
    results["precision"] = precision

    # 3. Benchmark
    print("="*60)
    print("Step 3: Benchmarking...")
    print("="*60)

    # Load PyTorch model for comparison
    pytorch_model = EgoIndustrialModule.load_from_checkpoint(checkpoint)

    benchmark_results = run_full_benchmark(
        pytorch_model=pytorch_model,
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        input_shapes=[(1, 3, 16, 224, 224)],
        batch_sizes=[1, 4, 8, 16, max_batch],
        warmup=10,
        runs=100,
        output_json=str(output_dir / "benchmark.json"),
    )
    results["benchmark"] = benchmark_results

    # 4. Accuracy comparison
    print("="*60)
    print("Step 4: Accuracy verification...")
    print("="*60)

    pytorch_model.eval()
    pytorch_model.cuda()

    acc_results = compare_accuracy(
        pytorch_model=pytorch_model,
        onnx_path=str(onnx_path),
        input_shapes=[(1, 3, 16, 224, 224)],
        rtol=1e-3,
        atol=1e-5,
    )
    results["accuracy"] = acc_results

    # Save summary
    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"ONNX: {onnx_path}")
    print(f"TensorRT: {engine_path}")
    print(f"Benchmark: {output_dir / 'benchmark.json'}")
    print(f"Summary: {summary_path}")

    return results


def benchmark_only(
    engine_path: str,
    onnx_path: str | None = None,
    checkpoint: str | None = None,
    batch_sizes: list[int] = None,
    runs: int = 100,
    output: str | None = None,
) -> dict:
    """Run benchmark only on existing engine."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    input_shapes = [(bs, 3, 16, 224, 224) for bs in batch_sizes]

    pytorch_model = None
    if checkpoint:
        pytorch_model = EgoIndustrialModule.load_from_checkpoint(checkpoint)

    results = run_full_benchmark(
        pytorch_model=pytorch_model,
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shapes=input_shapes,
        warmup=10,
        runs=runs,
        output_json=output,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Export and benchmark EgoIndustrial models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export pipeline
    export_parser = subparsers.add_parser("export", help="Full export pipeline")
    export_parser.add_argument("checkpoint", help="Path to .ckpt file")
    export_parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    export_parser.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification")
    export_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8", help="TensorRT precision")
    export_parser.add_argument("--max-batch", type=int, default=32, help="Max batch size")
    export_parser.add_argument("--workspace", type=int, default=1<<30, help="Workspace size (bytes)")
    export_parser.add_argument("--calibration-data", help="Path to calibration data for INT8")

    # Benchmark only
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark existing engine")
    bench_parser.add_argument("engine", help="Path to TensorRT engine")
    bench_parser.add_argument("--onnx", help="Path to ONNX model")
    bench_parser.add_argument("--checkpoint", help="Path to PyTorch checkpoint")
    bench_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    bench_parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    bench_parser.add_argument("-o", "--output", help="Output JSON file")

    args = parser.parse_args()

    if args.command == "export":
        export_pipeline(
            checkpoint=args.checkpoint,
            output_dir=args.output,
            opset=args.opset,
            simplify=not args.no_simplify,
            precision=args.precision,
            max_batch=args.max_batch,
            workspace=args.workspace,
            calibration_data=args.calibration_data,
        )
    elif args.command == "benchmark":
        benchmark_only(
            engine_path=args.engine,
            onnx_path=args.onnx,
            checkpoint=args.checkpoint,
            batch_sizes=args.batch_sizes,
            runs=args.runs,
            output=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
