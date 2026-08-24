"""
Modal.com deployment for EgoIndustrial inference server.
Run with: modal deploy scripts/deploy_modal.py
"""

import modal

app = modal.App("egoindustrial-inference")

# Base image with PyTorch and dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1",
    "torchvision>=0.16",
    "fastapi>=0.104",
    "uvicorn>=0.24",
    "prometheus-client>=0.19",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "pydantic>=2.0",
    "pydantic-core>=2.0",
)

# Try to install TensorRT (may need special handling on Modal)
try:
    image = image.pip_install("tensorrt", index_url="https://pypi.ngc.nvidia.com")
except Exception:
    pass

# Volume for model engine
engine_volume = modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)

# Model volume for engine file
MODEL_PATH = "/models/model_int8.engine"


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)},
    timeout=600,
    scaledown_window=300,
)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    import torch
    return {
        "status": "healthy",
        "engine_loaded": True,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)},
    timeout=600,
    scaledown_window=300,
    concurrency_limit=32,
)
@modal.web_endpoint(method="POST")
def infer(request_data: dict):
    """Main inference endpoint."""
    import logging

    import numpy as np
    import torch

    from egoindustrial.inference.models import InferenceRequest, InferenceResponse
    from egoindustrial.inference.tensorrt_engine import TensorRTEngine

    logger = logging.getLogger(__name__)

    try:
        # Load engine (cached in global scope for reuse)
        global _engine
        if "_engine" not in globals():
            logger.info("Loading TensorRT engine...")
            globals()["_engine"] = TensorRTEngine(MODEL_PATH)
            logger.info("Engine loaded successfully")

        globals()["_engine"]

        # Parse request
        request = InferenceRequest(**request_data)

        # Preprocess video input
        video = np.array(request.video, dtype=np.float32)
        if video.ndim == 4:  # [C, T, H, W]
            video = video[None]  # Add batch dim [1, C, T, H, W]
        elif video.ndim != 5:
            raise ValueError(f"Expected 4D or 5D video input, got {video.ndim}D")

        # Run inference
        outputs = globals()["_engine"].infer(video)

        # Process outputs
        verb_logits = outputs[0]   # [B, num_verb]
        noun_logits = outputs[1]   # [B, num_noun]
        action_logits = outputs[2]  # [B, num_action]

        # Apply softmax and get top-5
        verb_probs = torch.softmax(torch.from_numpy(verb_logits[0]), dim=-1)
        noun_probs = torch.softmax(torch.from_numpy(noun_logits[0]), dim=-1)
        action_probs = torch.softmax(torch.from_numpy(action_logits[0]), dim=-1)

        verb_top5 = verb_probs.topk(5)
        noun_top5 = noun_probs.topk(5)
        action_top5 = action_probs.topk(5)

        # Format response
        response = InferenceResponse(
            verb_probs=verb_probs.tolist(),
            noun_probs=noun_probs.tolist(),
            action_probs=action_probs.tolist(),
            verb_top5=[
                {"label": f"verb_{i}", "probability": float(v), "class_id": int(i)}
                for i, v in zip(verb_top5.indices, verb_top5.values)
            ],
            noun_top5=[
                {"label": f"noun_{i}", "probability": float(v), "class_id": int(i)}
                for i, v in zip(noun_top5.indices, noun_top5.values)
            ],
            action_top5=[
                {"label": f"action_{i}", "probability": float(v), "class_id": int(i)}
                for i, v in zip(action_top5.indices, action_top5.values)
            ],
            latency_ms=0.0,  # TODO: add timing
        )

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)},
    timeout=600,
)
@modal.web_endpoint(method="GET")
def model_info():
    """Model information endpoint."""
    return {
        "name": "egoindustrial",
        "version": "0.1.0",
        "input_shape": [3, 16, 224, 224],
        "output_classes": {
            "verb": 97,
            "noun": 300,
            "action": 3806,
        },
        "supported_tasks": ["verb", "noun", "action"],
        "precision": "int8",
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)},
    timeout=600,
)
@modal.web_endpoint(method="GET")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest

    # TODO: Add actual metrics collection
    return generate_latest()


# Batch inference endpoint
@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)},
    timeout=600,
    scaledown_window=300,
    concurrency_limit=32,
)
@modal.web_endpoint(method="POST")
def infer_batch(request_data: dict):
    """Batch inference endpoint."""
    from egoindustrial.inference.models import BatchInferenceRequest, BatchInferenceResponse

    request = BatchInferenceRequest(**request_data)

    results = []
    for req in request.requests:
        # Reuse single inference logic
        # For now, process sequentially (could be optimized)
        single_result = infer({"video": req.video})
        results.append(single_result)

    return BatchInferenceResponse(responses=results)


if __name__ == "__main__":
    modal.run()
