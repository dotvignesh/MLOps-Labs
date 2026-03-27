from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Use a different built-in dataset to make the lab submission clearly distinct.
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=wine.target_names)

    output_dir = Path(".")
    joblib.dump(model, output_dir / "wine_model.pkl")

    summary = "\n".join(
        [
            "Model training completed successfully.",
            f"Dataset: {wine['DESCR'].splitlines()[0]}",
            "Model: LogisticRegression with StandardScaler pipeline",
            f"Test accuracy: {accuracy:.4f}",
            "",
            "Classification report:",
            report,
        ]
    )

    (output_dir / "training_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
