"""Shared constants for training and inference."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model.json"
SCALER_PATH = PROJECT_ROOT / "scaler.json"

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "children",
    "sex_male",
    "smoker_yes",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]

SCALE_COLUMNS = ["age", "bmi", "children"]

VALID_SEX = {"female", "male"}
VALID_SMOKER = {"no", "yes"}
VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}

INPUT_BOUNDS = {
    "age": (0, 120),
    "bmi": (10.0, 60.0),
    "children": (0, 20),
}
