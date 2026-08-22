import json

import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"


def test_index_renders_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Estimate Your Medical Insurance Cost" in response.data


def test_api_predict_valid_input(client):
    response = client.post(
        "/api/predict",
        data=json.dumps(
            {
                "age": 31,
                "sex": "male",
                "bmi": 25.74,
                "children": 0,
                "smoker": "yes",
                "region": "northwest",
            }
        ),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["predicted_cost"] > 0


def test_api_predict_missing_input(client):
    response = client.post(
        "/api/predict",
        data=json.dumps({"age": 31}),
        content_type="application/json",
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_api_predict_invalid_input(client):
    response = client.post(
        "/api/predict",
        data=json.dumps(
            {
                "age": 31,
                "sex": "invalid",
                "bmi": 25.74,
                "children": 0,
                "smoker": "yes",
                "region": "northwest",
            }
        ),
        content_type="application/json",
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False


def test_form_predict_valid_input(client):
    response = client.post(
        "/predict",
        data={
            "age": "31",
            "sex": "male",
            "bmi": "25.74",
            "children": "0",
            "smoker": "yes",
            "region": "northwest",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Estimated Premium" in response.data


def test_form_predict_invalid_input(client):
    response = client.post(
        "/predict",
        data={
            "age": "31",
            "sex": "male",
            "bmi": "25.74",
            "children": "0",
            "smoker": "yes",
            "region": "invalid-region",
        },
    )
    assert response.status_code == 400
    assert b"Region must be one of" in response.data
