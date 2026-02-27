from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data import load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
REPORT_PATH = MODEL_DIR / "training_report.json"


def fit_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=42,
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def train_and_save_model():
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)

    per_class_metrics = []
    for class_id, class_name in enumerate(target_names):
        class_data = class_report[str(class_id)]
        per_class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "precision": round(float(class_data["precision"]), 3),
                "recall": round(float(class_data["recall"]), 3),
                "f1_score": round(float(class_data["f1-score"]), 3),
            }
        )

    training_report = {
        "dataset": "wine",
        "model": "RandomForestClassifier",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(accuracy, 4),
        "test_samples": int(len(y_test)),
        "feature_count": int(len(feature_names)),
        "class_metrics": per_class_metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_names": target_names,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(training_report, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training report saved to: {REPORT_PATH}")
    print(f"Accuracy: {training_report['accuracy']}")

if __name__ == "__main__":
    train_and_save_model()
