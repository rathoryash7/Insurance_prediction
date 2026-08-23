# Insurance Cost Prediction

Machine learning web application that estimates annual medical insurance costs from patient profile details. The production app uses a Flask API with an XGBoost regressor trained on the classic [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`).

## Architecture

```text
User form / JSON API
        |
        v
   Flask (app.py)
        |
        v
   ml/preprocess.py  -> validation + feature encoding
        |
        v
   ml/predictor.py   -> load model/scaler + inference
        |
        v
   xgb_model_optimized.pkl + scaler.pkl
```

## Tech Stack

- Python 3.11+
- Flask
- Portable JSON model inference
- pandas / scikit-learn / XGBoost (training only)
- pytest

## ML Pipeline

1. Raw inputs: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
2. Categorical encoding matches training `pd.get_dummies(..., drop_first=True)`:
   - `sex_male`
   - `smoker_yes`
   - region one-hot with `northeast` as the reference category
3. Continuous features `age`, `bmi`, and `children` are scaled with the saved scaler values
4. The XGBoost model predicts annual insurance charges in USD

Shared preprocessing lives in `ml/preprocess.py` so training and inference stay aligned.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py
```

Open `http://127.0.0.1:5000`.

Optional training:

```bash
python train_model.py
```

This regenerates both the development pickle artifacts and the lightweight runtime artifacts (`model.json` and `scaler.json`).

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `5000` | Local server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode locally |

No secrets are required for inference.

## Development Commands

```bash
python app.py
python -m pytest
```

## API

### `GET /health`

Health check.

### `POST /api/predict`

JSON request:

```json
{
  "age": 31,
  "sex": "male",
  "bmi": 25.74,
  "children": 0,
  "smoker": "yes",
  "region": "northwest"
}
```

Success response:

```json
{
  "success": true,
  "data": {
    "predicted_cost": 18666.72,
    "currency": "USD",
    "formatted_cost": "USD 18,666.72"
  }
}
```

Error response:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Region must be one of: northeast, northwest, southeast, southwest."
  }
}
```

### `POST /predict`

HTML form endpoint used by the web UI.

## Testing

Tests cover:

- Valid input
- Missing input
- Invalid categorical values
- Boundary validation
- API success/error responses
- Training/inference preprocessing consistency

Run:

```bash
python -m pytest -q
```

## Deployment

### Vercel

This repo includes `vercel.json` for Python serverless deployment via `@vercel/python`.

Requirements:

- `app.py` exports the Flask `app`
- `requirements.txt` pins `scikit-learn==1.3.0` to match serialized model artifacts
- Model files must be committed or otherwise available at build/runtime

Deploy with the Vercel CLI or by connecting the GitHub repository.

### Static Landing Page

`index.html` is a separate marketing template from the Flask prediction UI. Update its CTA links if your production prediction URL changes.

## Troubleshooting

- **Model file not found**: ensure `model.json` and `scaler.json` exist in the project root.
- **Unexpected predictions**: confirm inputs use semantic values (`male`, `yes`, `northwest`) rather than numeric codes.
- **Vercel cold starts**: first request may be slower while Python dependencies and model artifacts load.

## Legacy Scripts

These files are exploratory notebooks/scripts and are not used by production inference:

- `mip.py`
- `insureance.py`
- `new.py`

Use `train_model.py` and the `ml/` package for the supported pipeline.
