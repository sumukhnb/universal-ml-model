from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelRunResult:
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    fitted_pipeline: Pipeline


@dataclass
class PersistedModelArtifact:
    pipeline: Pipeline
    model_name: str
    task_type: str
    target_col: str
    feature_cols: List[str]


def detect_task_type(series: pd.Series) -> str:
    if series.dtype == "object" or str(series.dtype).startswith("category") or str(series.dtype) == "bool":
        return "classification"
    unique_values = series.nunique(dropna=True)
    if unique_values <= 20:
        return "classification"
    return "regression"


def prepare_xy(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    filtered_df = df[feature_cols + [target_col]].copy()
    filtered_df = filtered_df.dropna(subset=[target_col])
    x = filtered_df[feature_cols]
    y = filtered_df[target_col]
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def get_models(task_type: str) -> Dict[str, object]:
    if task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
    }


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1 (weighted)": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_and_compare(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: Optional[str] = None,
    test_size: float = 0.2,
) -> List[ModelRunResult]:
    x, y = prepare_xy(df, target_col, feature_cols)
    task = task_type or detect_task_type(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y if task == "classification" else None
    )

    preprocessor = build_preprocessor(x)
    model_map = get_models(task)
    results: List[ModelRunResult] = []

    for model_name, model in model_map.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)

        metrics = (
            evaluate_regression(y_test, preds)
            if task == "regression"
            else evaluate_classification(y_test, preds)
        )

        results.append(
            ModelRunResult(
                model_name=model_name,
                task_type=task,
                metrics=metrics,
                fitted_pipeline=pipeline,
            )
        )
    return results


def best_model(results: List[ModelRunResult]) -> ModelRunResult:
    if not results:
        raise ValueError("No model results available.")
    task = results[0].task_type
    if task == "regression":
        return sorted(results, key=lambda r: r.metrics["RMSE"])[0]
    return sorted(results, key=lambda r: r.metrics["F1 (weighted)"], reverse=True)[0]


def build_lag_features(
    df: pd.DataFrame, date_col: str, target_col: str, lags: int = 3
) -> pd.DataFrame:
    work = df[[date_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    for lag in range(1, lags + 1):
        work[f"lag_{lag}"] = work[target_col].shift(lag)
    return work.dropna().reset_index(drop=True)


def forecast_next_steps(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    periods: int = 7,
    lags: int = 3,
) -> pd.DataFrame:
    lag_df = build_lag_features(df, date_col, target_col, lags=lags)
    if lag_df.empty:
        raise ValueError("Not enough rows to build lag features. Add more historical data.")

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]
    results = train_and_compare(lag_df, target_col, feature_cols, task_type="regression")
    model = best_model(results).fitted_pipeline

    original_dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    inferred_freq = pd.infer_freq(original_dates.tail(min(len(original_dates), 8)))
    freq = inferred_freq if inferred_freq else "D"

    history = lag_df[target_col].tolist()
    future_dates = pd.date_range(start=original_dates.iloc[-1], periods=periods + 1, freq=freq)[1:]

    preds: List[float] = []
    for _ in range(periods):
        row = {f"lag_{lag}": history[-lag] for lag in range(1, lags + 1)}
        next_val = float(model.predict(pd.DataFrame([row]))[0])
        preds.append(next_val)
        history.append(next_val)

    return pd.DataFrame({date_col: future_dates, f"predicted_{target_col}": preds})


def save_model_artifact(
    pipeline: Pipeline,
    model_name: str,
    task_type: str,
    target_col: str,
    feature_cols: List[str],
    output_path: str,
) -> None:
    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": feature_cols,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model_artifact(model_path: str) -> PersistedModelArtifact:
    payload = joblib.load(model_path)
    return PersistedModelArtifact(
        pipeline=payload["pipeline"],
        model_name=payload["model_name"],
        task_type=payload["task_type"],
        target_col=payload["target_col"],
        feature_cols=payload["feature_cols"],
    )
