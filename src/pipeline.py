"""
pipeline.py — Master Training Pipeline + Automated Retraining
Ties together data generation, feature engineering, model training,
SHAP computation, and model persistence in one orchestrated flow.
"""

import pickle
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import generate_churn_dataset
from src.feature_engineering import run_full_pipeline, FEATURE_COLUMNS, TARGET_COLUMN
from src.model import train_all_models, load_model
from src.shap_explain import (
    load_shap_explainer, compute_shap_values,
    plot_global_summary, plot_bar_importance,
)

MODELS_DIR   = Path("models")
DATA_DIR     = Path("data")
REPORTS_DIR  = Path("reports")

for d in [MODELS_DIR, DATA_DIR / "processed", REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_pipeline(regenerate_data: bool = False, n_customers: int = 10_000):
    """
    Full end-to-end pipeline:
    1. Load or generate data
    2. Feature engineering
    3. Train/test split + SMOTE
    4. Train all 3 models
    5. SHAP explainability
    6. Save artifacts
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Churn Prediction Pipeline")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    raw_path = DATA_DIR / "raw" / "telecom_churn.csv"
    if regenerate_data or not raw_path.exists():
        logger.info("Generating synthetic dataset...")
        df_raw = generate_churn_dataset(n_customers)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_path, index=False)
    else:
        logger.info(f"Loading existing dataset from {raw_path}")
        df_raw = pd.read_csv(raw_path)

    logger.info(f"Dataset shape: {df_raw.shape}  |  Churn rate: {df_raw['churn'].mean():.1%}")

    # ── Step 2: Feature Engineering ───────────────────────────────────────
    df_eng = run_full_pipeline(df_raw.copy())
    proc_path = DATA_DIR / "processed" / "churn_engineered.csv"
    df_eng.to_csv(proc_path, index=False)
    logger.info(f"Engineered dataset saved → {proc_path}")

    # ── Step 3: Train/Test Split ──────────────────────────────────────────
    available_features = [f for f in FEATURE_COLUMNS if f in df_eng.columns]
    X = df_eng[available_features]
    y = df_eng[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    logger.info(f"Churn in train: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")

    # Save test set for evaluation scripts
    X_test.to_csv(DATA_DIR / "processed" / "X_test.csv", index=False)
    y_test.to_csv(DATA_DIR / "processed" / "y_test.csv", index=False)

    # Save feature list
    with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(available_features, f)

    # ── Step 4: Model Training ────────────────────────────────────────────
    results = train_all_models(X_train, X_test, y_train, y_test, available_features)

    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    logger.success(f"🏆 Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']})")

    # ── Step 5: SHAP Explainability ───────────────────────────────────────
    logger.info("Computing SHAP values...")
    xgb_model  = load_model("xgboost")
    explainer   = load_shap_explainer(xgb_model, X_train, "xgboost")

    # Use a sample for speed
    X_sample  = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_vals  = compute_shap_values(explainer, X_sample)

    # Save SHAP values array + feature names (explainer has lambda, can't pickle)
    import numpy as np
    sv_array = shap_vals.values if hasattr(shap_vals, "values") else np.array(shap_vals)
    with open(MODELS_DIR / "shap_values.pkl", "wb") as f:
        pickle.dump({"values": sv_array, "feature_names": list(X_sample.columns)}, f)

    plot_global_summary(shap_vals, X_sample)
    plot_bar_importance(shap_vals, X_sample)

    logger.success("✅ Pipeline complete! All models trained and SHAP plots generated.")
    logger.info(f"   Reports → {REPORTS_DIR.absolute()}")
    logger.info(f"   Models  → {MODELS_DIR.absolute()}")

    return results


def retrain_pipeline():
    """
    Automated retraining: re-generates data, retrains all models.
    Can be triggered by a scheduler or CI/CD.
    """
    logger.info("🔄 Automated retraining triggered...")
    results = run_pipeline(regenerate_data=True)
    logger.success(f"Retraining complete. Results: {results}")
    return results


if __name__ == "__main__":
    run_pipeline(regenerate_data=True)
