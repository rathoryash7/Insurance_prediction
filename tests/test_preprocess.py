import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.constants import FEATURE_COLUMNS, MODEL_PATH, SCALER_PATH
from ml.predictor import clear_model_cache, predict_cost
from ml.preprocess import ValidationError, encode_features, prepare_training_features, validate_input


@pytest.fixture(scope="session")
def scaler() -> StandardScaler:
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="session")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(autouse=True)
def reset_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def test_validate_input_accepts_valid_payload():
    payload = validate_input(
        {
            "age": "31",
            "sex": "male",
            "bmi": "25.7",
            "children": "0",
            "smoker": "yes",
            "region": "northwest",
        }
    )
    assert payload["age"] == 31
    assert payload["sex"] == "male"


@pytest.mark.parametrize(
    "payload,missing_field",
    [
        (
            {
                "sex": "male",
                "bmi": "25.7",
                "children": "0",
                "smoker": "yes",
                "region": "northwest",
            },
            "age",
        ),
        (
            {
                "age": "31",
                "sex": "male",
                "children": "0",
                "smoker": "yes",
                "region": "northwest",
            },
            "bmi",
        ),
    ],
)
def test_validate_input_rejects_missing_fields(payload, missing_field):
    with pytest.raises(ValidationError, match="Missing required fields"):
        validate_input(payload)


@pytest.mark.parametrize(
    "payload,message",
    [
        (
            {
                "age": "31",
                "sex": "other",
                "bmi": "25.7",
                "children": "0",
                "smoker": "yes",
                "region": "northwest",
            },
            "Sex must be",
        ),
        (
            {
                "age": "31",
                "sex": "male",
                "bmi": "25.7",
                "children": "0",
                "smoker": "maybe",
                "region": "northwest",
            },
            "Smoker must be",
        ),
        (
            {
                "age": "150",
                "sex": "male",
                "bmi": "25.7",
                "children": "0",
                "smoker": "yes",
                "region": "northwest",
            },
            "Age must be between",
        ),
        (
            {
                "age": "31",
                "sex": "male",
                "bmi": "5",
                "children": "0",
                "smoker": "yes",
                "region": "northwest",
            },
            "BMI must be between",
        ),
    ],
)
def test_validate_input_rejects_invalid_values(payload, message):
    with pytest.raises(ValidationError, match=message):
        validate_input(payload)


def test_encode_features_matches_training_pipeline(scaler, model):
    data = pd.read_csv("insurance.csv")
    features = data.drop("charges", axis=1)
    training_matrix, _ = prepare_training_features(features, scaler=scaler, fit_scaler=False)
    training_predictions = model.predict(training_matrix)

    for index in range(5):
        row = data.iloc[index]
        encoded = encode_features(
            {
                "age": row["age"],
                "sex": row["sex"],
                "bmi": row["bmi"],
                "children": row["children"],
                "smoker": row["smoker"],
                "region": row["region"],
            },
            scaler,
        )
        manual_prediction = model.predict(encoded)[0]
        assert manual_prediction == pytest.approx(training_predictions[index], rel=0, abs=1e-4)


def test_predict_cost_returns_positive_value():
    result = predict_cost(
        {
            "age": 31,
            "sex": "male",
            "bmi": 25.74,
            "children": 0,
            "smoker": "yes",
            "region": "northwest",
        }
    )
    assert result.predicted_cost > 0
    assert result.currency == "USD"


def test_feature_columns_length():
    assert len(FEATURE_COLUMNS) == 8
