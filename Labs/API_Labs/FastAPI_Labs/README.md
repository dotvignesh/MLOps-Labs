# FastAPI Lab (Custom Version)

This is my custom version of the FastAPI lab.

## What I changed

1. I changed the dataset from Iris to Wine dataset (`sklearn.datasets.load_wine`).
2. I changed the model from Decision Tree to Random Forest.
3. I added a small dashboard endpoint: `/dashboard`.
4. I added a training report file (`model/training_report.json`) with metrics.

## How this extends the original repo

The original lab trains a Decision Tree on Iris and returns only a class id from `/predict`.
My version extends that baseline in three ways:

1. It uses a different dataset (Wine) with 13 features, so input design and model behavior are different.
2. It uses a different model (Random Forest), which is an ensemble method and gives class probabilities.
3. It adds a lightweight dashboard (`/dashboard`) that shows saved training metrics (accuracy and per-class scores), so the API is easier to inspect and explain.

## Project structure

```text
FastAPI_Labs/
├── assets/
├── model/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- `uv` installed

If you do not have `uv`, install it from: https://docs.astral.sh/uv/

## Setup with uv

From project root:

```bash
uv sync
```

This command creates `.venv` and installs all dependencies from `pyproject.toml`.

## Train the model

Run:

```bash
uv run python -m src.train
```

After training, you should see:

- `model/wine_model.pkl`
- `model/training_report.json`

## Start the FastAPI app

Run:

```bash
uv run uvicorn src.main:app --reload
```

App URLs:

- API root: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8000/dashboard

## API usage

### Health check

- Method: `GET`
- Endpoint: `/`

Response example:

```json
{
  "status": "healthy"
}
```

### Predict wine class

- Method: `POST`
- Endpoint: `/predict`
- Input: one list with 13 numeric features

Request example:

```json
{
  "features": [
    13.2,
    1.78,
    2.14,
    11.2,
    100.0,
    2.65,
    2.76,
    0.26,
    1.28,
    4.38,
    1.05,
    3.4,
    1050.0
  ]
}
```

Response example:

```json
{
  "class_id": 0,
  "class_name": "class_0",
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.02,
    "class_2": 0.0
  }
}
```

Note: Class names come from the trained model metadata.

## Dashboard

Endpoint: `/dashboard`

This page shows:

- dataset name
- model name
- accuracy
- test sample count
- per-class precision/recall/F1
- training time (UTC)

## Quick test flow

1. `uv sync`
2. `uv run python -m src.train`
3. `uv run uvicorn src.main:app --reload`
4. Open `/docs` and test `/predict`
5. Open `/dashboard` and check metrics

## Notes

- If prediction fails with model not found, run training again.
- Keep feature order exactly as expected by the Wine dataset.
