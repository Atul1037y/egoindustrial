"""Export LightningModule to ONNX format."""


import torch
from omegaconf import DictConfig

from egoindustrial.training.module import EgoIndustrialModule


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: list[int] = [1, 3, 16, 224, 224],
    opset_version: int = 17,
    dynamic_axes: bool = True,
    simplify: bool = True,
    model_cfg: DictConfig | None = None,
) -> str:
    """Export trained model to ONNX.

    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Output .onnx path
        input_shape: [B, C, T, H, W] for VideoMAE/MViT, or list for SlowFast
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
        simplify: Run onnxsim after export
        model_cfg: Optional model config override

    Returns:
        Path to exported ONNX file
    """
    # Load model
    if model_cfg is None:
        model = EgoIndustrialModule.load_from_checkpoint(checkpoint_path)
    else:
        model = EgoIndustrialModule(model_cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    model.eval()

    # Prepare dummy input
    if isinstance(input_shape[0], list):
        # SlowFast: [slow_pathway, fast_pathway]
        dummy_input = [
            torch.randn(*shape) for shape in input_shape
        ]
        input_names = ["slow_pathway", "fast_pathway"]
    else:
        dummy_input = torch.randn(*input_shape)
        input_names = ["video"]

    # Dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        if isinstance(dummy_input, list):
            dynamic_axes_dict = {
                "slow_pathway": {0: "batch"},
                "fast_pathway": {0: "batch"},
                "verb": {0: "batch"},
                "noun": {0: "batch"},
                "action": {0: "batch"},
            }
        else:
            dynamic_axes_dict = {
                "video": {0: "batch"},
                "verb": {0: "batch"},
                "noun": {0: "batch"},
                "action": {0: "batch"},
            }

    # Export
    output_names = ["verb", "noun", "action"]
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        verbose=False,
    )

    # Simplify
    if simplify:
        try:
            import onnx
            import onnxsim

            onnx_model = onnx.load(output_path)
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "ONNX simplification failed"
            onnx.save(onnx_model, output_path)
            print(f"ONNX simplified and saved to {output_path}")
        except ImportError:
            print("onnxsim not installed, skipping simplification")

    return output_path


def validate_onnx(onnx_path: str, input_shape: list[int]) -> bool:
    """Validate ONNX model runs correctly."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(onnx_path)

    if isinstance(input_shape[0], list):
        dummy_input = [np.random.randn(*s).astype(np.float32) for s in input_shape]
        input_dict = {f"input_{i}": dummy_input[i] for i in range(len(dummy_input))}
    else:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {"video": dummy_input}

    outputs = session.run(None, input_dict)
    print(f"ONNX validation passed. Output shapes: {[o.shape for o in outputs]}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .ckpt")
    parser.add_argument("output", type=str, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    export_to_onnx(
        args.checkpoint,
        args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )
    validate_onnx(args.output, [1, 3, 16, 224, 224])
