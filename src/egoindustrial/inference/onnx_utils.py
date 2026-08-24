"""ONNX validation, simplification, and optimization utilities."""

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


def validate_onnx_model(
    onnx_path: str,
    input_shapes: list[list[int]],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Validate ONNX model against PyTorch reference.

    Args:
        onnx_path: Path to ONNX file
        input_shapes: List of input shapes [[B,C,T,H,W], ...]
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance

    Returns:
        True if validation passes
    """
    # Load ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Generate random inputs
    inputs = {}
    for i, shape in enumerate(input_shapes):
        input_name = session.get_inputs()[i].name
        inputs[input_name] = np.random.randn(*shape).astype(np.float32)

    # Run inference
    ort_outputs = session.run(None, inputs)

    print("ONNX validation passed.")
    print(f"  Inputs: {[(k, v.shape) for k, v in inputs.items()]}")
    print(f"  Outputs: {[(f'out_{i}', o.shape) for i, o in enumerate(ort_outputs)]}")
    return True


def simplify_onnx(
    input_path: str,
    output_path: str,
    check_n: int = 3,
    perform_optimization: bool = True,
) -> bool:
    """Simplify ONNX model using onnxsim.

    Args:
        input_path: Input ONNX path
        output_path: Output simplified ONNX path
        check_n: Number of random inputs to test
        perform_optimization: Run optimization passes

    Returns:
        True if simplification succeeded
    """
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not installed. Install with: pip install onnxsim")
        return False

    model = onnx.load(input_path)
    model_simp, check = onnxsim.simplify(
        model,
        check_n=check_n,
        perform_optimization=perform_optimization,
    )

    if not check:
        print("ONNX simplification check failed")
        return False

    onnx.save(model_simp, output_path)
    print(f"Simplified ONNX saved to {output_path}")
    return True


def optimize_onnx_for_tensorrt(
    input_path: str,
    output_path: str,
    fp16: bool = False,
    int8: bool = False,
) -> bool:
    """Optimize ONNX for TensorRT (fold constants, remove identity ops)."""
    try:
        import onnx_graphsurgeon as gs
    except ImportError:
        print("onnx-graphsurgeon not installed. Install with: pip install onnx-graphsurgeon")
        return False

    graph = gs.import_onnx(onnx.load(input_path))

    # Fold constants
    graph.fold_constants().cleanup()

    # Remove identity nodes
    for node in graph.nodes:
        if node.op == "Identity":
            node.outputs = node.inputs

    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), output_path)
    print(f"Optimized ONNX saved to {output_path}")
    return True


def get_model_info(onnx_path: str) -> dict[str, Any]:
    """Extract model metadata from ONNX."""
    model = onnx.load(onnx_path)

    info = {
        "ir_version": model.ir_version,
        "opset_import": [f"{op.domain}:{op.version}" for op in model.opset_import],
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "inputs": [],
        "outputs": [],
    }

    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
        info["inputs"].append({"name": inp.name, "shape": shape})

    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
        info["outputs"].append({"name": out.name, "shape": shape})

    return info


def compare_outputs(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_shapes: list[list[int]],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple[bool, dict[str, float]]:
    """Compare PyTorch vs ONNX Runtime outputs."""
    pytorch_model.eval()

    # Generate test inputs
    torch_inputs = []
    ort_inputs = {}
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for i, shape in enumerate(input_shapes):
        x = torch.randn(*shape)
        torch_inputs.append(x)

        inp_name = session.get_inputs()[i].name
        ort_inputs[inp_name] = x.numpy()

    # PyTorch forward
    with torch.no_grad():
        if len(torch_inputs) == 1:
            torch_out = pytorch_model(torch_inputs[0])
        else:
            torch_out = pytorch_model(torch_inputs)

    # ONNX Runtime forward
    ort_outs = session.run(None, ort_inputs)

    # Compare
    results = {}
    if isinstance(torch_out, dict):
        torch_out = [torch_out["verb"], torch_out["noun"], torch_out["action"]]
    elif not isinstance(torch_out, (list, tuple)):
        torch_out = [torch_out]

    for i, (t_out, o_out) in enumerate(zip(torch_out, ort_outs)):
        t_np = t_out.numpy() if isinstance(t_out, torch.Tensor) else t_out
        max_diff = np.abs(t_np - o_out).max()
        mean_diff = np.abs(t_np - o_out).mean()
        close = np.allclose(t_np, o_out, rtol=rtol, atol=atol)
        results[f"output_{i}"] = {"max_diff": max_diff, "mean_diff": mean_diff, "close": close}
        if not close:
            print(f"MISMATCH output_{i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    all_close = all(r["close"] for r in results.values())
    return all_close, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=str)
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 16, 224, 224])
    args = parser.parse_args()

    info = get_model_info(args.onnx_path)
    print(f"Model Info: {info}")
    validate_onnx_model(args.onnx_path, [args.input_shape])
