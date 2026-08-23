"""Shared feature encoding used by training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.constants import FEATURE_COLUMNS, INPUT_BOUNDS, SCALE_COLUMNS, VALID_REGIONS, VALID_SEX, VALID_SMOKER


class ValidationError(ValueError):
    """Raised when user input fails validation."""


def validate_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize raw prediction input."""
    required = ("age", "sex", "bmi", "children", "smoker", "region")
    missing = [field for field in required if field not in payload or payload[field] in (None, "")]
    if missing:
        raise ValidationError(f"Missing required fields: {', '.join(missing)}")

    try:
        age = int(payload["age"])
        bmi = float(payload["bmi"])
        children = int(payload["children"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("Age, BMI, and children must be valid numbers.") from exc

    sex = str(payload["sex"]).strip().lower()
    smoker = str(payload["smoker"]).strip().lower()
    region = str(payload["region"]).strip().lower()

    if sex not in VALID_SEX:
        raise ValidationError("Sex must be 'male' or 'female'.")
    if smoker not in VALID_SMOKER:
        raise ValidationError("Smoker must be 'yes' or 'no'.")
    if region not in VALID_REGIONS:
        raise ValidationError(
            "Region must be one of: northeast, northwest, southeast, southwest."
        )

    age_min, age_max = INPUT_BOUNDS["age"]
    bmi_min, bmi_max = INPUT_BOUNDS["bmi"]
    children_min, children_max = INPUT_BOUNDS["children"]

    if not age_min <= age <= age_max:
        raise ValidationError(f"Age must be between {age_min} and {age_max}.")
    if not bmi_min <= bmi <= bmi_max:
        raise ValidationError(f"BMI must be between {bmi_min} and {bmi_max}.")
    if not children_min <= children <= children_max:
        raise ValidationError(f"Children must be between {children_min} and {children_max}.")

    return {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }


def encode_features(raw_input: dict[str, Any], scaler: StandardScaler) -> np.ndarray:
    """
    Encode a single record into the feature vector expected by the model.

    Matches pandas get_dummies(..., drop_first=True) used during training.
    """
    validated = validate_input(raw_input)

    features = {
        "age": validated["age"],
        "bmi": validated["bmi"],
        "children": validated["children"],
        "sex_male": 1 if validated["sex"] == "male" else 0,
        "smoker_yes": 1 if validated["smoker"] == "yes" else 0,
        "region_northwest": 1 if validated["region"] == "northwest" else 0,
        "region_southeast": 1 if validated["region"] == "southeast" else 0,
        "region_southwest": 1 if validated["region"] == "southwest" else 0,
    }

    frame = pd.DataFrame([features])[FEATURE_COLUMNS]
    frame[SCALE_COLUMNS] = scaler.transform(frame[SCALE_COLUMNS])
    return frame.values


def prepare_training_features(
    features: pd.DataFrame, scaler: StandardScaler | None = None, fit_scaler: bool = False
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare training/inference-aligned feature matrix from raw CSV columns."""
    encoded = pd.get_dummies(features, columns=["sex", "smoker", "region"], drop_first=True)

    for column in FEATURE_COLUMNS:
        if column not in encoded.columns:
            encoded[column] = 0

    encoded = encoded[FEATURE_COLUMNS]

    if fit_scaler:
        scaler = StandardScaler()
        encoded[SCALE_COLUMNS] = scaler.fit_transform(encoded[SCALE_COLUMNS])
        return encoded, scaler

    if scaler is None:
        raise ValueError("Scaler is required when fit_scaler=False.")

    encoded[SCALE_COLUMNS] = scaler.transform(encoded[SCALE_COLUMNS])
    return encoded, scaler
