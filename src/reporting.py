from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def build_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_pct": [float(df[col].isna().mean() * 100) for col in df.columns],
            "unique_count": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )
    return profile


def write_run_report(
    report_dir: str,
    metrics_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    best_model_name: str,
    task_type: str,
) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_{timestamp}.csv"
    profile_path = out_dir / f"data_profile_{timestamp}.csv"
    summary_path = out_dir / f"summary_{timestamp}.json"

    metrics_df.to_csv(metrics_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_at": timestamp,
                "best_model": best_model_name,
                "task_type": task_type,
                "models_tested": metrics_df["Model"].tolist() if "Model" in metrics_df.columns else [],
            },
            fp,
            indent=2,
        )

    return {
        "metrics_csv": str(metrics_path),
        "profile_csv": str(profile_path),
        "summary_json": str(summary_path),
    }
