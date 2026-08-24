#!/usr/bin/env bash
# Deploy EgoIndustrial to GCP Cloud Run
# Usage: ./scripts/deploy_gcp.sh [PROJECT_ID] [REGION] [SERVICE_NAME]

set -euo pipefail

PROJECT_ID="${1:-your-gcp-project-id}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-egoindustrial-inference}"
ENGINE_PATH="${4:-outputs/model_int8.engine}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/egoindustrial-inference"

echo "=========================================="
echo "Deploying EgoIndustrial to GCP Cloud Run"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "Engine: ${ENGINE_PATH}"
echo "Image: ${IMAGE_NAME}"
echo "=========================================="

# Check engine exists
if [ ! -f "${ENGINE_PATH}" ]; then
    echo "Error: Engine not found at ${ENGINE_PATH}"
    echo "Run export pipeline first:"
    echo "  python scripts/export.py export --checkpoint outputs/checkpoints/best.ckpt"
    exit 1
fi

# Configure gcloud
echo "Configuring gcloud..."
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"

# Build Docker image
echo "Building Docker image..."
docker build \
    -f docker/Dockerfile.inference \
    -t "${IMAGE_NAME}:latest" \
    -t "gcr.io/${PROJECT_ID}/egoindustrial-inference:latest" \
    .

# Push to GCR
echo "Pushing to Google Container Registry..."
docker push "${IMAGE_NAME}:latest"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy egoindustrial-inference \
    --image="${IMAGE_NAME}:latest" \
    --region=us-central1 \
    --platform=managed \
    --allow-unauthenticated \
    --port=8000 \
    --cpu=4 \
    --memory=16Gi \
    --gpu=1 \
    --gpu-type=nvidia-tesla-t4 \
    --min-instances=0 \
    --max-instances=10 \
    --concurrency=32 \
    --set-env-vars=ENGINE_PATH=/app/model_int8.engine

# Get service URL
SERVICE_URL=$(gcloud run services describe egoindustrial-inference \
    --region=us-central1 \
    --format='value(status.url)')

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Service URL: ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/health"
echo "Inference: ${SERVICE_URL}/infer"
echo "Batch inference: ${SERVICE_URL}/infer/batch"
echo "Metrics: ${SERVICE_URL}/metrics"
echo "Model info: ${SERVICE_URL}/model/info"
echo "=========================================="

# Test deployment
echo "Testing deployment..."
curl -s "${SERVICE_URL}/health" | jq .