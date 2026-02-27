import json
from pathlib import Path

from fastapi import FastAPI, status, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic import Field

from src.predict import predict_data


app = FastAPI(
    title="Wine Classification API",
    description="Predict wine class from 13 chemical features.",
    version="1.0.0",
)
REPORT_PATH = Path(__file__).resolve().parent.parent / "model" / "training_report.json"
EXPECTED_FEATURE_COUNT = 13


class WineData(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=EXPECTED_FEATURE_COUNT,
        max_length=EXPECTED_FEATURE_COUNT,
        description="13 wine features in the same order as sklearn load_wine.",
    )


class WineResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: dict[str, float]


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        predictions, probabilities, class_names = predict_data([wine_features.features])
        predicted_class_id = int(predictions[0])
        probability_map = {
            class_name: round(float(probabilities[0][index]), 4)
            for index, class_name in enumerate(class_names)
        }

        return WineResponse(
            class_id=predicted_class_id,
            class_name=class_names[predicted_class_id],
            probabilities=probability_map,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
async def training_dashboard():
    if not REPORT_PATH.exists():
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h2>Training report not found</h2>
                    <p>Please run <code>uv run python -m src.train</code> first.</p>
                </body>
            </html>
            """
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    class_rows = "".join(
        [
            f"<tr><td>{item['class_id']}</td><td>{item['class_name']}</td>"
            f"<td>{item['precision']}</td><td>{item['recall']}</td><td>{item['f1_score']}</td></tr>"
            for item in report["class_metrics"]
        ]
    )

    html_content = f"""
    <html>
        <head>
            <title>Wine Model Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                .card {{ max-width: 760px; padding: 1rem; border: 1px solid #ccc; border-radius: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
                th, td {{ border: 1px solid #ddd; padding: 0.6rem; text-align: left; }}
                th {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Wine Classification Dashboard</h1>
                <p><strong>Dataset:</strong> {report['dataset']}</p>
                <p><strong>Model:</strong> {report['model']}</p>
                <p><strong>Accuracy:</strong> {report['accuracy']}</p>
                <p><strong>Test Samples:</strong> {report['test_samples']}</p>
                <p><strong>Feature Count:</strong> {report['feature_count']}</p>
                <p><strong>Trained At (UTC):</strong> {report['trained_at_utc']}</p>

                <h2>Per-Class Metrics</h2>
                <table>
                    <tr>
                        <th>Class ID</th>
                        <th>Class Name</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
                    {class_rows}
                </table>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
