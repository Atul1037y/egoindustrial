#!/usr/bin/env python3
"""Deploy EgoIndustrial inference server to GCP Cloud Run."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def deploy_to_cloud_run(
    project_id: str,
    region: str = "us-central1",
    service_name: str = "egoindustrial-inference",
    image_name: str = None,
    engine_path: str = "outputs/model_int8.engine",
    port: int = 8000,
    cpu: int = 4,
    memory: str = "16Gi",
    gpu: int = 1,
    gpu_type: str = "nvidia-tesla-t4",
    min_instances: int = 0,
    max_instances: int = 10,
    concurrency: int = 32,
    allow_unauthenticated: bool = True,
) -> str:
    """Deploy inference server to Cloud Run."""

    # Build image name if not provided
    if image_name is None:
        image_name = f"gcr.io/{project_id}/egoindustrial-inference"

    print("Deploying to Cloud Run...")
    print(f"  Project: {project_id}")
    print(f"  Region: {region}")
    print(f"  Service: {service_name}")
    print(f"  Image: {image_name}")
    print(f"  Engine: {engine_path}")

    # Check engine exists
    engine_path = Path(engine_path)
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Run export pipeline first: python scripts/export.py export --checkpoint outputs/checkpoints/best.ckpt")
        sys.exit(1)

    # Build Docker image
    print("\nBuilding Docker image...")
    run_cmd([
        "docker", "build",
        "-f", "docker/Dockerfile.inference",
        "-t", f"{image_name}:latest",
        ".",
    ])

    # Push to GCR
    print("\nPushing to Google Container Registry...")
    run_cmd(["docker", "push", f"{image_name}:latest"])

    # Deploy to Cloud Run
    print("\nDeploying to Cloud Run...")

    cmd = [
        "gcloud", "run", "deploy", service_name,
        f"--image={image_name}:latest",
        f"--region={region}",
        "--platform=managed",
        f"--port={port}",
        f"--cpu={cpu}",
        f"--memory={memory}",
        f"--gpu={gpu}",
        f"--gpu-type={gpu_type}",
        f"--min-instances={min_instances}",
        f"--max-instances={max_instances}",
        f"--concurrency={concurrency}",
        "--set-env-vars=ENGINE_PATH=/app/model_int8.engine",
    ]

    if allow_unauthenticated:
        cmd.append("--allow-unauthenticated")

    run_cmd(cmd)

    # Get service URL
    url_result = run_cmd([
        "gcloud", "run", "services", "describe", service_name,
        f"--region={region}",
        "--format=value(status.url)",
    ])
    service_url = url_result.stdout.strip()

    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE")
    print("="*60)
    print(f"Service URL: {service_url}")
    print(f"Health check: {service_url}/health")
    print(f"Inference endpoint: {service_url}/infer")
    print(f"Metrics: {service_url}/metrics")
    print(f"Model info: {service_url}/model/info")
    print("="*60)

    return service_url


def deploy_with_modal(
    app_name: str = "egoindustrial-inference",
    engine_path: str = "outputs/model_int8.engine",
) -> str:
    """Deploy using Modal.com (alternative to Cloud Run)."""

    engine_path = Path(engine_path)
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        sys.exit(1)

    # Create Modal app file
    modal_app = f'''
import modal

app = modal.App("{app_name}")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision",
    "fastapi", "uvicorn",
    "prometheus-client",
    "numpy", "opencv-python",
)

# Add TensorRT (requires special handling)
# image = image.pip_install("tensorrt", index_url="https://pypi.ngc.nvidia.com")

# Mount engine file
engine_volume = modal.Volume.from_name("egoindustrial-engine", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    volumes={{"/models": engine_volume}},
    timeout=600,
)
@modal.web_endpoint(method="POST")
def infer(request_data: dict):
    import torch
    import numpy as np
    from egoindustrial.inference.tensorrt_engine import TensorRTEngine
    from egoindustrial.inference.models import InferenceRequest, InferenceResponse
    from egoindustrial.data.transforms import get_transforms
    import torch.nn.functional as F

    # Load engine
    engine = TensorRTEngine("/models/model_int8.engine")

    # Parse request
    request = InferenceRequest(**request_data)

    # Preprocess
    video = np.array(request.video, dtype=np.float32)
    if video.ndim == 4:
        video = video[None]

    # Run inference
    outputs = engine.infer(video)

    verb_probs = torch.softmax(torch.from_numpy(outputs[0][0]), dim=-1).tolist()
    noun_probs = torch.softmax(torch.from_numpy(outputs[1][0]), dim=-1).tolist()
    action_probs = torch.softmax(torch.from_numpy(outputs[2][0]), dim=-1).tolist()

    return InferenceResponse(
        verb_probs=verb_probs,
        noun_probs=noun_probs,
        action_probs=action_probs,
        latency_ms=0,
    )

if __name__ == "__main__":
    modal.run()
'''

    # Write modal app
    modal_file = Path("scripts/deploy_modal.py")
    modal_file.write_text(modal_app)

    # Deploy
    print("Deploying to Modal...")
    subprocess.run(["modal", "deploy", "scripts/deploy_modal.py"], check=True)

    print("\nModal deployment complete!")
    print("Check Modal dashboard for endpoint URL")


def main():
    parser = argparse.ArgumentParser(description="Deploy EgoIndustrial to cloud")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # GCP Cloud Run
    gcp_parser = subparsers.add_parser("gcp", help="Deploy to GCP Cloud Run")
    gcp_parser.add_argument("--project", required=True, help="GCP project ID")
    gcp_parser.add_argument("--region", default="us-central1", help="GCP region")
    gcp_parser.add_argument("--service", default="egoindustrial-inference", help="Service name")
    gcp_parser.add_argument("--image", help="Docker image name (default: gcr.io/PROJECT/egoindustrial-inference)")
    gcp_parser.add_argument("--engine", default="outputs/model_int8.engine", help="Path to TensorRT engine")
    gcp_parser.add_argument("--port", type=int, default=8000, help="Container port")
    gcp_parser.add_argument("--cpu", type=int, default=4, help="CPU cores")
    gcp_parser.add_argument("--memory", default="16Gi", help="Memory (e.g., 16Gi)")
    gcp_parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs")
    gcp_parser.add_argument("--gpu-type", default="nvidia-tesla-t4", help="GPU type")
    gcp_parser.add_argument("--min-instances", type=int, default=0, help="Min instances")
    gcp_parser.add_argument("--max-instances", type=int, default=10, help="Max instances")
    gcp_parser.add_argument("--concurrency", type=int, default=32, help="Concurrency per instance")
    gcp_parser.add_argument("--authenticated", action="store_true", help="Require authentication")

    # Modal
    modal_parser = subparsers.add_parser("modal", help="Deploy to Modal.com")
    modal_parser.add_argument("--app-name", default="egoindustrial-inference", help="Modal app name")
    modal_parser.add_argument("--engine", default="outputs/model_int8.engine", help="Path to TensorRT engine")

    args = parser.parse_args()

    if args.command == "gcp":
        deploy_to_cloud_run(
            project_id=args.project,
            region=args.region,
            service_name=args.service,
            image_name=args.image,
            engine_path=args.engine,
            port=args.port,
            cpu=args.cpu,
            memory=args.memory,
            gpu=args.gpu,
            gpu_type=args.gpu_type,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
            concurrency=args.concurrency,
            allow_unauthenticated=not args.authenticated,
        )
    elif args.command == "modal":
        deploy_with_modal(
            app_name=args.app_name,
            engine_path=args.engine,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
