"""Health checks and Prometheus metrics for inference server."""

import time
from typing import Any

import psutil
import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

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
    from prometheus_client import CONTENT_TYPE_LATEST

    @app.get("/metrics")
    async def metrics():
        # Update system metrics before serving
        update_system_metrics()
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def update_system_metrics() -> None:
    """Update system-level metrics."""
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


class HealthChecker:
    """Comprehensive health checks."""

    def __init__(self, engine=None, batcher=None):
        self.engine = engine
        self.batcher = batcher
        self.start_time = time.time()

    def check_health(self) -> dict[str, Any]:
        """Run all health checks."""
        checks = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": {},
        }

        # Engine check
        checks["checks"]["engine"] = self._check_engine()

        # Batcher check
        checks["checks"]["batcher"] = self._check_batcher()

        # GPU check
        checks["checks"]["gpu"] = self._check_gpu()

        # Memory check
        checks["checks"]["memory"] = self._check_memory()

        # Disk check
        checks["checks"]["disk"] = self._check_disk()

        # Determine overall status
        if any(c["status"] != "healthy" for c in checks["checks"].values()):
            checks["status"] = "degraded"
        if any(c["status"] == "critical" for c in checks["checks"].values()):
            checks["status"] = "unhealthy"

        return checks

    def _check_engine(self) -> dict[str, Any]:
        if self.engine is None:
            return {"status": "critical", "message": "TensorRT engine not loaded"}

        try:
            # Quick inference test
            import numpy as np
            dummy = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
            _ = self.engine.infer(dummy)
            return {"status": "healthy", "message": "Engine responsive"}
        except Exception as e:
            return {"status": "critical", "message": f"Engine error: {str(e)}"}

    def _check_batcher(self) -> dict[str, Any]:
        if self.batcher is None:
            return {"status": "warning", "message": "Batcher not initialized"}

        if not self.batcher.running:
            return {"status": "critical", "message": "Batcher not running"}

        queue_size = self.batcher.queue.qsize() if hasattr(self.batcher, "queue") else 0
        return {
            "status": "healthy",
            "message": "Batcher running",
            "queue_size": queue_size,
        }

    def _check_gpu(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"status": "warning", "message": "CUDA not available"}

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_pct = allocated / total * 100

            if usage_pct > 95:
                status = "critical"
            elif usage_pct > 85:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "message": f"GPU memory: {usage_pct:.1f}% used",
                "allocated_mb": allocated / 1e6,
                "total_mb": total / 1e6,
            }
        except Exception as e:
            return {"status": "critical", "message": f"GPU check failed: {str(e)}"}

    def _check_memory(self) -> dict[str, Any]:
        mem = psutil.virtual_memory()
        usage_pct = mem.percent

        if usage_pct > 95:
            status = "critical"
        elif usage_pct > 85:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "message": f"System memory: {usage_pct:.1f}% used",
            "used_gb": mem.used / 1e9,
            "total_gb": mem.total / 1e9,
        }

    def _check_disk(self) -> dict[str, Any]:
        disk = psutil.disk_usage("/")
        usage_pct = disk.percent

        if usage_pct > 95:
            status = "critical"
        elif usage_pct > 85:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "message": f"Disk: {usage_pct:.1f}% used",
            "free_gb": disk.free / 1e9,
            "total_gb": disk.total / 1e9,
        }


def create_health_endpoint(app, engine=None, batcher=None):
    """Add health check endpoints to FastAPI app."""
    checker = HealthChecker(engine, batcher)

    @app.get("/health")
    async def health():
        return checker.check_health()

    @app.get("/health/live")
    async def liveness():
        """Kubernetes liveness probe."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness():
        """Kubernetes readiness probe."""
        health = checker.check_health()
        if health["status"] == "unhealthy":
            from fastapi import HTTPException
            raise HTTPException(503, detail=health)
        return {"status": "ready"}


if __name__ == "__main__":
    # Test metrics
    update_system_metrics()
    print("Metrics test:")
    print(generate_latest(REGISTRY).decode())
