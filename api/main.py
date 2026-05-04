"""
main.py — FastAPI Real-Time Prediction API
Serves churn probability + SHAP explanations via REST endpoints.
Includes A/B testing and batch prediction capabilities.
"""

import pickle
import random
import time
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_engineering import run_full_pipeline, FEATURE_COLUMNS
from src.shap_explain import load_shap_explainer, compute_shap_values, get_top_churn_reasons

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction with SHAP explainability",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")

# ── Model cache ────────────────────────────────────────────────────────────────
_model_cache       = {}
_explainer_cache   = {}
_feature_columns   = []
_request_log       = []           # In-memory request log for A/B tracking


def load_models():
    """Load all models and explainer into memory at startup."""
    global _feature_columns

    try:
        with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
            _feature_columns = pickle.load(f)

        for name in ["xgboost", "random_forest", "logistic_regression"]:
            path = MODELS_DIR / f"{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    _model_cache[name] = pickle.load(f)
                logger.info(f"Loaded model: {name}")

        # Load SHAP explainer for XGBoost
        exp_path = MODELS_DIR / "shap_explainer.pkl"
        if exp_path.exists():
            with open(exp_path, "rb") as f:
                _explainer_cache["xgboost"] = pickle.load(f)
            logger.info("SHAP explainer loaded")

    except Exception as e:
        logger.warning(f"Model loading skipped (run pipeline first): {e}")


@app.on_event("startup")
async def startup_event():
    load_models()
    logger.info("🚀 Churn Prediction API started")


# ── Schemas ────────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender:             str   = Field(default="Male",            description="Male or Female")
    senior_citizen:     int   = Field(default=0,   ge=0, le=1)
    partner:            str   = Field(default="No")
    dependents:         str   = Field(default="No")
    tenure_months:      int   = Field(default=12,  ge=1, le=72)
    phone_service:      str   = Field(default="Yes")
    internet_service:   str   = Field(default="DSL")
    online_security:    str   = Field(default="No")
    tech_support:       str   = Field(default="No")
    streaming_tv:       str   = Field(default="No")
    contract:           str   = Field(default="Month-to-month")
    paperless_billing:  str   = Field(default="No")
    payment_method:     str   = Field(default="Electronic check")
    monthly_charges:    float = Field(default=65.0, ge=0)
    total_charges:      float = Field(default=780.0, ge=0)
    calls_per_month:    int   = Field(default=4,   ge=0)
    data_usage_gb:      float = Field(default=20.0, ge=0)
    support_tickets:    int   = Field(default=1,   ge=0)
    late_payments:      int   = Field(default=0,   ge=0)
    avg_call_duration:  float = Field(default=8.0, ge=0)


class PredictionResponse(BaseModel):
    customer_id:         Optional[str]
    churn_probability:   float
    churn_prediction:    int
    risk_level:          str
    top_churn_reasons:   list
    model_used:          str
    response_time_ms:    float


# ── Helper ─────────────────────────────────────────────────────────────────────
def customer_to_df(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


def get_risk_level(prob: float) -> str:
    if prob >= 0.70:  return "🔴 HIGH"
    if prob >= 0.40:  return "🟡 MEDIUM"
    return "🟢 LOW"


def engineer_and_predict(customer: CustomerInput, model_name: str = "xgboost"):
    """Feature-engineer customer data and return prediction."""
    df_raw = customer_to_df(customer)
    df_eng = run_full_pipeline(df_raw, drop_id=False)

    available = [f for f in _feature_columns if f in df_eng.columns]
    X = df_eng[available]

    model = _model_cache.get(model_name)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded. Run pipeline first.")

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    return X, prob, pred


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "✅ Churn Prediction API is running",
        "models_loaded": list(_model_cache.keys()),
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "models": list(_model_cache.keys())}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput, model: str = "xgboost"):
    """
    Predict churn probability for a single customer.
    Returns probability, risk level, and top SHAP-based churn reasons.
    """
    t0 = time.time()
    X, prob, pred = engineer_and_predict(customer, model)

    # SHAP reasons
    reasons = []
    if model == "xgboost" and "xgboost" in _explainer_cache:
        try:
            explainer  = _explainer_cache["xgboost"]
            shap_vals   = compute_shap_values(explainer, X)
            reasons     = get_top_churn_reasons(shap_vals, X, idx=0, top_n=5)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    elapsed = round((time.time() - t0) * 1000, 2)

    response = PredictionResponse(
        customer_id=None,
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_level=get_risk_level(prob),
        top_churn_reasons=reasons,
        model_used=model,
        response_time_ms=elapsed,
    )

    # Log for A/B tracking
    _request_log.append({"model": model, "prob": prob, "pred": pred})

    return response


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(customers: list[CustomerInput], model: str = "xgboost"):
    """Batch prediction for multiple customers."""
    results = []
    for i, c in enumerate(customers):
        try:
            X, prob, pred = engineer_and_predict(c, model)
            results.append({
                "index":             i,
                "churn_probability": round(prob, 4),
                "churn_prediction":  pred,
                "risk_level":        get_risk_level(prob),
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return {"batch_size": len(customers), "results": results}


@app.get("/ab-test/summary", tags=["A/B Testing"])
async def ab_test_summary():
    """
    Summarise A/B test metrics across models.
    Returns average predicted churn probability per model variant.
    """
    if not _request_log:
        return {"message": "No requests logged yet."}

    import pandas as pd
    df = pd.DataFrame(_request_log)
    summary = df.groupby("model").agg(
        requests=("prob", "count"),
        avg_churn_prob=("prob", "mean"),
        churn_rate=("pred", "mean"),
    ).reset_index().to_dict(orient="records")

    return {"ab_test_summary": summary}


@app.get("/model/metrics", tags=["Model Info"])
async def get_metrics():
    """Return saved model evaluation metrics."""
    metrics_path = MODELS_DIR / "metrics_summary.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Run pipeline.py first to generate metrics.")
    import json
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/retrain", tags=["Pipeline"])
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger automated model retraining in the background."""
    from src.pipeline import retrain_pipeline
    background_tasks.add_task(retrain_pipeline)
    return {"status": "Retraining started in background. Check logs for progress."}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
