# Universal ML Predictor

A small end-to-end machine learning project for public GitHub showcase.

This app lets you:
- Upload any CSV dataset.
- Pick target and feature columns.
- Auto-detect or manually choose **regression** vs **classification**.
- Compare:
  - `Linear Regression` and `Random Forest Regressor` (regression)
  - `Logistic Regression` and `Random Forest Classifier` (classification)
- Download predictions.
- Optionally run simple future forecasting from time-based trend data (for sales/weather-like use cases).
- Persist the best model and reuse it for future predictions.
- Expose a FastAPI endpoint for deployment.
- Auto-generate report files for each model run.

## 1) Project Structure

```text
universal-ml-predictor/
  app.py
  api.py
  requirements.txt
  .gitignore
  README.md
  src/
    __init__.py
    pipeline.py
    reporting.py
  sample_data/
    sales_sample.csv
```

## 2) Setup

```bash
python -m venv .venv
```

### Windows (PowerShell)
```bash
.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) Run the App

```bash
streamlit run app.py
```

## 4) Run FastAPI

```bash
uvicorn api:app --reload
```

API docs:
- `http://127.0.0.1:8000/docs`

### Endpoints
- `GET /health`
- `POST /train`
- `POST /predict`
- `POST /forecast`

## 5) Run with Docker (FastAPI)

Build image:
```bash
docker build -t universal-ml-predictor .
```

Run container:
```bash
docker run --rm -p 8000:8000 universal-ml-predictor
```

API docs in browser:
- `http://127.0.0.1:8000/docs`

## 6) How to Use (Streamlit)

1. Upload a CSV file.
2. Choose the target column (value you want to predict).
3. Choose feature columns.
4. Keep task type on `auto` (or set manually).
5. Click **Train and Compare Models**.
6. Review metrics and download predictions.

### Future Forecasting

If your dataset has a date column and historical target values:
1. Choose a date column.
2. Choose forecast horizon (for example, 7 days).
3. Click **Forecast Future**.

### Model Persistence
- Best model is saved to `models/latest_model.joblib`.
- You can reuse it in Streamlit and through FastAPI `/predict`.

### Automatic Reports
- Run reports are saved under `reports/`.
- Files include metrics CSV, data profile CSV, and JSON summary.

## 7) Notes for Multi-Domain Data

- You can upload different domains (sales, weather, etc.) as long as data is tabular CSV.
- Model quality depends on the uploaded features and data quality.
- For forecasting, future exogenous variables are not required because this implementation uses lag-based trend features of the target.

## 8) Suggested GitHub Upload Steps

```bash
git init
git add .
git commit -m "Initial commit: universal ML predictor with model comparison and forecasting"
```

Then create a new GitHub repo and push.

---

If you want, this can be extended with:
- XGBoost/LightGBM
- cross-validation
- explainability (SHAP)
- deployment with Docker/cloud
