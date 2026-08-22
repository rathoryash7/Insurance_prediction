import os

from flask import Flask, jsonify, render_template, request

from ml.predictor import ModelLoadError, predict_cost
from ml.preprocess import ValidationError

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _error_response(code: str, message: str, status: int):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status


def _success_response(data: dict, status: int = 200):
    return jsonify({"success": True, "data": data}), status


def _extract_payload() -> dict:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload, dict):
            return payload

    return {
        "age": request.form.get("age"),
        "sex": request.form.get("sex"),
        "bmi": request.form.get("bmi"),
        "children": request.form.get("children"),
        "smoker": request.form.get("smoker"),
        "region": request.form.get("region"),
    }


@app.route("/")
def index():
    return render_template("insurance-prediction.html")


@app.route("/health")
def health():
    return _success_response({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        result = predict_cost(_extract_payload())
        return _success_response(
            {
                "predicted_cost": round(result.predicted_cost, 2),
                "currency": result.currency,
                "formatted_cost": result.formatted_cost,
            }
        )
    except ValidationError as exc:
        return _error_response("VALIDATION_ERROR", str(exc), 400)
    except ModelLoadError as exc:
        return _error_response("MODEL_ERROR", str(exc), 500)
    except Exception:
        return _error_response(
            "INTERNAL_ERROR",
            "An unexpected error occurred while generating the prediction.",
            500,
        )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("insurance-prediction.html")

    try:
        result = predict_cost(_extract_payload())
        return render_template(
            "insurance-prediction.html",
            result={
                "predicted_cost": round(result.predicted_cost, 2),
                "currency": result.currency,
                "formatted_cost": result.formatted_cost,
            },
        )
    except ValidationError as exc:
        return render_template("insurance-prediction.html", error=str(exc)), 400
    except ModelLoadError as exc:
        return render_template("insurance-prediction.html", error=str(exc)), 500
    except Exception:
        return (
            render_template(
                "insurance-prediction.html",
                error="Something went wrong. Please try again.",
            ),
            500,
        )


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
