from pathlib import Path
from typing import Sequence

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"


def predict_data(X: Sequence[Sequence[float]]):
    """
    Predict class labels and probabilities for input data.
    Args:
        X (Sequence[Sequence[float]]): Input data for predictions.
    Returns:
        tuple: Predicted labels, class probabilities, and class names.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Train the model first with `uv run python -m src.train`."
        )

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    class_names = [str(name) for name in model_bundle["target_names"]]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob, class_names
