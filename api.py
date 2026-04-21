from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.pipeline import (
    best_model,
    detect_task_type,
    forecast_next_steps,
    load_model_artifact,
    save_model_artifact,
    train_and_compare,
)


app = FastAPI(title="Universal ML Predictor API", version="1.0.0")
DEFAULT_MODEL_PATH = "models/latest_model.joblib"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    feature_cols: str = Form(...),
    task_type: Optional[str] = Form(None),
) -> dict:
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    parsed_features: List[str] = [c.strip() for c in feature_cols.split(",") if c.strip()]
    if not parsed_features:
        raise HTTPException(status_code=400, detail="feature_cols must include at least one column.")

    try:
        effective_task = task_type if task_type else detect_task_type(df[target_col])
        results = train_and_compare(
            df=df, target_col=target_col, feature_cols=parsed_features, task_type=effective_task
        )
        winner = best_model(results)
        save_model_artifact(
            pipeline=winner.fitted_pipeline,
            model_name=winner.model_name,
            task_type=winner.task_type,
            target_col=target_col,
            feature_cols=parsed_features,
            output_path=DEFAULT_MODEL_PATH,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Column not found: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metrics = []
    for run in results:
        row = {"model": run.model_name, "task_type": run.task_type}
        row.update(run.metrics)
        metrics.append(row)

    return {
        "message": "Training complete. Best model persisted.",
        "model_path": DEFAULT_MODEL_PATH,
        "best_model": winner.model_name,
        "task_type": winner.task_type,
        "metrics": metrics,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_path: str = Form(DEFAULT_MODEL_PATH)) -> dict:
    try:
        artifact = load_model_artifact(model_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load model: {exc}") from exc

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        preds = artifact.pipeline.predict(df[artifact.feature_cols])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    out = df.copy()
    out[f"predicted_{artifact.target_col}"] = preds
    return {
        "model_name": artifact.model_name,
        "target_col": artifact.target_col,
        "predictions": out.to_dict(orient="records"),
    }


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    target_col: str = Form(...),
    periods: int = Form(7),
    lags: int = Form(3),
) -> dict:
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        fc = forecast_next_steps(df=df, date_col=date_col, target_col=target_col, periods=periods, lags=lags)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Forecast failed: {exc}") from exc

    return {
        "date_col": date_col,
        "target_col": target_col,
        "periods": periods,
        "forecast": fc.to_dict(orient="records"),
    }
