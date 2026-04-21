import pandas as pd
import streamlit as st

from src.pipeline import (
    best_model,
    detect_task_type,
    forecast_next_steps,
    load_model_artifact,
    save_model_artifact,
    train_and_compare,
)
from src.reporting import build_data_profile, write_run_report


st.set_page_config(page_title="Universal ML Predictor", layout="wide")
st.title("Universal ML Predictor")
st.caption(
    "Upload a dataset, choose target/feature columns, compare models, and generate predictions."
)

uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    columns = df.columns.tolist()
    target_col = st.selectbox("Select target column", columns)
    feature_cols = st.multiselect(
        "Select feature columns (default: all except target)",
        [c for c in columns if c != target_col],
        default=[c for c in columns if c != target_col],
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    detected_task = detect_task_type(df[target_col])
    task_choice = st.selectbox(
        "Task type",
        ["auto", "regression", "classification"],
        index=0,
        help=f"Auto currently detects: {detected_task}",
    )
    task_type = None if task_choice == "auto" else task_choice

    if st.button("Train and Compare Models", type="primary"):
        with st.spinner("Training models..."):
            results = train_and_compare(
                df=df, target_col=target_col, feature_cols=feature_cols, task_type=task_type
            )

        st.success("Training complete.")
        metrics_rows = []
        for res in results:
            row = {"Model": res.model_name, "Task": res.task_type}
            row.update(res.metrics)
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        st.subheader("Model Performance")
        st.dataframe(metrics_df, use_container_width=True)

        top = best_model(results)
        st.info(f"Best model: **{top.model_name}**")
        save_model_artifact(
            pipeline=top.fitted_pipeline,
            model_name=top.model_name,
            task_type=top.task_type,
            target_col=target_col,
            feature_cols=feature_cols,
            output_path="models/latest_model.joblib",
        )
        st.success("Best model saved to `models/latest_model.joblib`.")

        st.subheader("Predictions on Uploaded Data")
        preds = top.fitted_pipeline.predict(df[feature_cols])
        out_df = df.copy()
        out_df[f"predicted_{target_col}"] = preds
        st.dataframe(out_df.head(100), use_container_width=True)
        st.download_button(
            label="Download predictions CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.subheader("Automatic Report Files")
        profile_df = build_data_profile(df)
        st.dataframe(profile_df, use_container_width=True)
        report_paths = write_run_report(
            report_dir="reports",
            metrics_df=metrics_df,
            profile_df=profile_df,
            best_model_name=top.model_name,
            task_type=top.task_type,
        )
        st.caption(f"Saved report files to `reports/`: {report_paths}")

    st.markdown("---")
    st.subheader("Predict Using Saved Model")
    if st.button("Use `models/latest_model.joblib` for prediction"):
        try:
            artifact = load_model_artifact("models/latest_model.joblib")
            pred_df = df.copy()
            pred_df[f"predicted_{artifact.target_col}"] = artifact.pipeline.predict(
                df[artifact.feature_cols]
            )
            st.success(f"Loaded model: {artifact.model_name}")
            st.dataframe(pred_df.head(100), use_container_width=True)
            st.download_button(
                label="Download saved-model predictions CSV",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="saved_model_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Saved-model prediction failed: {exc}")

    st.markdown("---")
    st.subheader("Time-Based Forecasting (Optional)")
    st.caption(
        "Use this section when your data has a date/time column and you want future values."
    )
    date_candidates = ["<None>"] + columns
    date_col = st.selectbox("Date column", date_candidates, index=0)
    forecast_periods = st.slider("Future periods to forecast", min_value=1, max_value=30, value=7)
    lag_count = st.slider("Number of lag steps", min_value=2, max_value=12, value=3)

    if date_col != "<None>" and st.button("Forecast Future", type="secondary"):
        with st.spinner("Forecasting future values..."):
            try:
                forecast_df = forecast_next_steps(
                    df=df,
                    date_col=date_col,
                    target_col=target_col,
                    periods=forecast_periods,
                    lags=lag_count,
                )
                st.success("Forecast complete.")
                st.dataframe(forecast_df, use_container_width=True)
                st.line_chart(
                    forecast_df.set_index(date_col)[f"predicted_{target_col}"], use_container_width=True
                )
                st.download_button(
                    label="Download forecast CSV",
                    data=forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name="forecast.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Could not forecast: {exc}")
else:
    st.info("Upload a CSV file to begin.")
