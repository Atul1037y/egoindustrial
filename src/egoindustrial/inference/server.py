"""FastAPI inference server with dynamic batching and Prometheus metrics."""

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from egoindustrial.inference.metrics import METRICS, setup_metrics
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from egoindustrial.inference.models import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
)
from egoindustrial.inference.tensorrt_engine import TensorRTEngine


class DynamicBatcher:
    """Dynamic batching for high-throughput inference."""

    def __init__(
        self,
        engine: TensorRTEngine,
        max_batch_size: int = 32,
        max_wait_ms: int = 10,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds

        self.queue: asyncio.Queue = asyncio.Queue()
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.results: dict[str, asyncio.Future] = {}
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._batch_loop())

    async def stop(self):
        self.running = False

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        future = asyncio.get_event_loop().create_future()
        request_id = id(request)

        self.results[request_id] = future
        await self.queue.put((request_id, request))

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            self.results.pop(request_id, None)

    async def _batch_loop(self):
        while self.running:
            batch = []
            batch_ids = []

            # Collect first request
            try:
                req_id, request = await asyncio.wait_for(
                    self.queue.get(), timeout=self.max_wait_ms
                )
                batch.append(request)
                batch_ids.append(req_id)
            except asyncio.TimeoutError:
                continue

            # Collect additional requests up to max_batch_size
            wait_start = time.time()
            while len(batch) < self.max_batch_size:
                remaining_time = self.max_wait_ms - (time.time() - wait_start)
                if remaining_time <= 0:
                    break

                try:
                    req_id, request = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining_time
                    )
                    batch.append(request)
                    batch_ids.append(req_id)
                except asyncio.TimeoutError:
                    break

            # Process batch
            if batch:
                await self._process_batch(batch, batch_ids)

    async def _process_batch(self, requests: list[InferenceRequest], request_ids: list[int]):
        start_time = time.time()

        try:
            # Prepare inputs
            if len(requests) == 1:
                video = np.array(requests[0].video, dtype=np.float32)
                if video.ndim == 4:  # [C, T, H, W]
                    video = video[None]  # Add batch dim
            else:
                video = np.stack([np.array(r.video, dtype=np.float32) for r in requests])

            # Run inference
            outputs = self.engine.infer(video)

            # Format responses
            for i, req_id in enumerate(request_ids):
                if req_id in self.results:
                    verb_logits = outputs[0][i] if outputs[0].ndim > 1 else outputs[0]
                    noun_logits = outputs[1][i] if outputs[1].ndim > 1 else outputs[1]
                    action_logits = outputs[2][i] if outputs[2].ndim > 1 else outputs[2]

                    response = InferenceResponse(
                        verb_probs=torch.softmax(torch.from_numpy(verb_logits), dim=-1).tolist(),
                        noun_probs=torch.softmax(torch.from_numpy(noun_logits), dim=-1).tolist(),
                        action_probs=torch.softmax(torch.from_numpy(action_logits), dim=-1).tolist(),
                        latency_ms=(time.time() - start_time) * 1000 / len(requests),
                    )
                    self.results[req_id].set_result(response)

        except Exception as e:
            for req_id in request_ids:
                if req_id in self.results:
                    self.results[req_id].set_exception(e)

        # Record metrics
        latency = (time.time() - start_time) * 1000
        METRICS["request_latency"].observe(latency)
        METRICS["batch_size"].observe(len(requests))


# Global batcher instance
batcher: DynamicBatcher | None = None
engine: TensorRTEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, engine
    # Startup
    engine = TensorRTEngine(app.state.engine_path)
    batcher = DynamicBatcher(
        engine,
        max_batch_size=app.state.max_batch_size,
        max_wait_ms=app.state.max_wait_ms,
    )
    await batcher.start()
    yield
    # Shutdown
    await batcher.stop()


app = FastAPI(
    title="EgoIndustrial Action Recognition API",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup Prometheus metrics
setup_metrics(app)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Single video inference."""
    if batcher is None:
        raise HTTPException(503, "Server not ready")
    return await batcher.infer(request)


@app.post("/infer/batch", response_model=BatchInferenceResponse)
async def infer_batch(request: BatchInferenceRequest):
    """Batch inference."""
    if batcher is None:
        raise HTTPException(503, "Server not ready")

    responses = []
    for req in request.requests:
        responses.append(await batcher.infer(req))

    return BatchInferenceResponse(responses=responses)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if engine is not None else "unhealthy",
        engine_loaded=engine is not None,
        batcher_running=batcher is not None and batcher.running,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Model information."""
    return ModelInfo(
        name="egoindustrial",
        version="0.1.0",
        input_shape=[3, 16, 224, 224],
        output_classes={"verb": 97, "noun": 300, "action": 3806},
    )


def create_app(
    engine_path: str,
    max_batch_size: int = 32,
    max_wait_ms: int = 10,
) -> FastAPI:
    """Create FastAPI app with custom config."""
    app.state.engine_path = engine_path
    app.state.max_batch_size = max_batch_size
    app.state.max_wait_ms = max_wait_ms
    return app


def run_server(
    engine_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 32,
    max_wait_ms: int = 10,
    workers: int = 1,
):
    """Run the inference server."""
    app = create_app(engine_path, max_batch_size, max_wait_ms)
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=str, help="TensorRT engine path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--max-wait", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_server(
        args.engine,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch,
        max_wait_ms=args.max_wait,
        workers=args.workers,
    )
