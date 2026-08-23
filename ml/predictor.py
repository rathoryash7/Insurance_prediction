"""Model loading and inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
from typing import Any

from ml.constants import MODEL_PATH, SCALER_PATH
from ml.preprocess import ValidationError, encode_features


@dataclass(frozen=True)
class PredictionResult:
    predicted_cost: float
    currency: str = "USD"

    @property
    def formatted_cost(self) -> str:
        return f"{self.currency} {self.predicted_cost:,.2f}"


class ModelLoadError(RuntimeError):
    """Raised when model artifacts cannot be loaded."""


@lru_cache(maxsize=1)
def _load_scaler() -> dict[str, list[float]]:
    if not SCALER_PATH.exists():
        raise ModelLoadError(f"Scaler file not found: {SCALER_PATH}")
    with SCALER_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise ModelLoadError(f"Model file not found: {MODEL_PATH}")
    with MODEL_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _predict_tree(tree: dict[str, Any], features: list[float]) -> float:
    node = 0
    while tree["left_children"][node] != -1:
        feature_index = tree["split_indices"][node]
        if features[feature_index] < tree["split_conditions"][node]:
            node = tree["left_children"][node]
        else:
            node = tree["right_children"][node]
    return tree["base_weights"][node]


def _predict_model(model: dict[str, Any], features: list[float]) -> float:
    learner = model["learner"]
    booster = learner["gradient_booster"]["model"]
    prediction = float(learner["learner_model_param"]["base_score"])
    for tree in booster["trees"]:
        prediction += _predict_tree(tree, features)
    return prediction


def predict_cost(payload: dict[str, Any]) -> PredictionResult:
    """Run inference for a single input record."""
    scaler = _load_scaler()
    model = _load_model()
    features = encode_features(payload, scaler)
    prediction = _predict_model(model, features[0])

    if not prediction == prediction or prediction in (float("inf"), float("-inf")) or prediction < 0:
        raise ModelLoadError("Model returned an invalid prediction.")

    return PredictionResult(predicted_cost=prediction)


def clear_model_cache() -> None:
    """Clear cached model artifacts (useful in tests)."""
    _load_scaler.cache_clear()
    _load_model.cache_clear()


__all__ = [
    "ModelLoadError",
    "PredictionResult",
    "ValidationError",
    "clear_model_cache",
    "predict_cost",
]
