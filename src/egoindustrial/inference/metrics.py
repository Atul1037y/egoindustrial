"""Prometheus metrics for inference server."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Global metrics registry
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "egoindustrial_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "egoindustrial_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "egoindustrial_batch_size",
    "Batch size distribution",
    buckets=[1, 2, 4, 8, 16, 32, 64],
    registry=REGISTRY,
)

# System metrics
GPU_MEMORY_USED = Gauge(
    "egoindustrial_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    registry=REGISTRY,
)

GPU_MEMORY_TOTAL = Gauge(
    "egoindustrial_gpu_memory_total_bytes",
    "Total GPU memory in bytes",
    registry=REGISTRY,
)

CPU_USAGE = Gauge(
    "egoindustrial_cpu_usage_percent",
    "CPU usage percentage",
    registry=REGISTRY,
)

MEMORY_USAGE = Gauge(
    "egoindustrial_memory_usage_bytes",
    "System memory usage in bytes",
    registry=REGISTRY,
)

# Model metrics
MODEL_LOAD_TIME = Histogram(
    "egoindustrial_model_load_seconds",
    "Model loading time in seconds",
    registry=REGISTRY,
)

INFERENCE_ERRORS = Counter(
    "egoindustrial_inference_errors_total",
    "Total inference errors",
    ["error_type"],
    registry=REGISTRY,
)

# Export all metrics for easy access
METRICS = {
    "request_count": REQUEST_COUNT,
    "request_latency": REQUEST_LATENCY,
    "batch_size": BATCH_SIZE,
    "gpu_memory_used": GPU_MEMORY_USED,
    "gpu_memory_total": GPU_MEMORY_TOTAL,
    "cpu_usage": CPU_USAGE,
    "memory_usage": MEMORY_USAGE,
    "model_load_time": MODEL_LOAD_TIME,
    "inference_errors": INFERENCE_ERRORS,
}


def setup_metrics(app) -> None:
    """Setup Prometheus metrics endpoint for FastAPI app."""
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def update_system_metrics() -> None:
    """Update system-level metrics."""
    import psutil
    import torch

    # CPU
    CPU_USAGE.set(psutil.cpu_percent())

    # Memory
    mem = psutil.virtual_memory()
    MEMORY_USAGE.set(mem.used)

    # GPU
    if torch.cuda.is_available():
        GPU_MEMORY_USED.set(torch.cuda.memory_allocated())
        GPU_MEMORY_TOTAL.set(torch.cuda.get_device_properties(0).total_memory)


def record_request(endpoint: str, status: str, latency: float) -> None:
    """Record request metrics."""
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


def record_batch_size(size: int) -> None:
    """Record batch size."""
    BATCH_SIZE.observe(size)


def record_error(error_type: str) -> None:
    """Record inference error."""
    INFERENCE_ERRORS.labels(error_type=error_type).inc()


def record_model_load_time(seconds: float) -> None:
    """Record model loading time."""
    MODEL_LOAD_TIME.observe(seconds)
