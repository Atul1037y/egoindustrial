"""Pydantic models for FastAPI inference server."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    """Single video inference request."""

    video: list[list[list[list[float]]]] = Field(
        ...,
        description="Video frames as nested list [C, T, H, W] or [T, H, W, C]",
        examples=[[[[0.0]]]],  # Placeholder
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video": [
                    [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(16)]
                    for _ in range(3)
                ]
            }
        }
    )


class InferenceResponse(BaseModel):
    """Single video inference response."""

    verb_probs: list[float] = Field(..., description="Verb class probabilities (97 classes)")
    noun_probs: list[float] = Field(..., description="Noun class probabilities (300 classes)")
    action_probs: list[float] = Field(..., description="Action class probabilities (3806 classes)")
    verb_top5: list[dict[str, Any]] | None = Field(None, description="Top-5 verb predictions")
    noun_top5: list[dict[str, Any]] | None = Field(None, description="Top-5 noun predictions")
    action_top5: list[dict[str, Any]] | None = Field(None, description="Top-5 action predictions")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchInferenceRequest(BaseModel):
    """Batch inference request."""

    requests: list[InferenceRequest] = Field(..., min_length=1, max_length=32)


class BatchInferenceResponse(BaseModel):
    """Batch inference response."""

    responses: list[InferenceResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Server status: healthy/unhealthy")
    engine_loaded: bool = Field(..., description="TensorRT engine loaded")
    batcher_running: bool = Field(..., description="Dynamic batcher running")
    gpu_memory_used_mb: float | None = Field(None, description="GPU memory used")
    gpu_memory_total_mb: float | None = Field(None, description="GPU memory total")


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    version: str
    input_shape: list[int]
    output_classes: dict[str, int]
    supported_tasks: list[str] = ["verb", "noun", "action"]
    precision: str = "int8"


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str | None = None


# Label maps for human-readable predictions
EPIC_VERBS = [
    "take",
    "put",
    "open",
    "close",
    "wash",
    "cut",
    "pour",
    "mix",
    "shake",
    "sprinkle",
    "peel",
    "slice",
    "chop",
    "dice",
    "grate",
    "squeeze",
    "spread",
    "dip",
    "stir",
    "fold",
    "unwrap",
    "wrap",
    "scoop",
    "drop",
    "throw",
    "pick",
    "place",
    "move",
    "hold",
    "touch",
    "press",
    "push",
    "pull",
    "turn",
    "rotate",
    "flip",
    "shake",
    "tap",
    "knock",
    "hit",
    "bang",
    "scratch",
    "rub",
    "wipe",
    "clean",
    "dry",
    "soak",
    "rinse",
    "drain",
    "strain",
    "filter",
    "separate",
    "combine",
    "add",
    "remove",
    "replace",
    "adjust",
    "measure",
    "check",
    "taste",
    "smell",
    "look",
    "watch",
    "wait",
    "rest",
    "walk",
    "stand",
    "sit",
    "bend",
    "reach",
    "extend",
    "lift",
    "lower",
    "carry",
    "bring",
    "fetch",
    "get",
    "find",
]

EPIC_NOUNS = [
    "knife",
    "spoon",
    "fork",
    "cup",
    "bowl",
    "plate",
    "pan",
    "pot",
    "lid",
    "handle",
    "drawer",
    "door",
    "fridge",
    "oven",
    "microwave",
    "sink",
    "tap",
    "water",
    "oil",
    "salt",
    "pepper",
    "sugar",
    "flour",
    "egg",
    "milk",
    "butter",
    "cheese",
    "bread",
    "tomato",
    "onion",
    "garlic",
    "pepper",
    "carrot",
    "potato",
    "meat",
    "fish",
    "chicken",
    "beef",
    "pork",
    "rice",
    "pasta",
    "noodle",
    "sauce",
    "soup",
    "salad",
    "sandwich",
    "pizza",
    "cake",
    "cookie",
    "fruit",
    "vegetable",
    "hand",
    "finger",
    "thumb",
    "palm",
    "wrist",
    "arm",
    "elbow",
    "shoulder",
    "towel",
    "cloth",
    "sponge",
    "brush",
    "whisk",
    "ladle",
    "spatula",
    "tongs",
    "peeler",
    "grater",
    "opener",
    "scissors",
    "chopper",
    "blender",
    "mixer",
]


def get_top5_predictions(probs: list[float], labels: list[str]) -> list[dict[str, Any]]:
    """Get top-5 predictions with labels."""
    import numpy as np

    probs = np.array(probs)
    top5_idx = np.argsort(probs)[-5:][::-1]
    return [
        {
            "label": labels[i] if i < len(labels) else f"class_{i}",
            "probability": float(probs[i]),
            "class_id": int(i),
        }
        for i in top5_idx
    ]


def enrich_response(response: InferenceResponse) -> InferenceResponse:
    """Add top-5 predictions with labels to response."""
    response.verb_top5 = get_top5_predictions(response.verb_probs, EPIC_VERBS)
    response.noun_top5 = get_top5_predictions(response.noun_probs, EPIC_NOUNS)
    response.action_top5 = get_top5_predictions(response.action_probs, [])
    return response
