"""
model.py — ML Pipeline: Logistic Regression → Random Forest → XGBoost
Trains, evaluates and saves all three models with MLflow tracking.
"""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_URI = "sqlite:///mlflow_tracking/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("churn-prediction")


# ── Model definitions ──────────────────────────────────────────────────────────
def get_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        )),
    ])


def get_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def get_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,   # handles class imbalance
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )


# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Compute and print all relevant classification metrics."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":           name,
        "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":          round(average_precision_score(y_test, y_prob), 4),
        "precision":       round(precision_score(y_test, y_pred), 4),
        "recall":          round(recall_score(y_test, y_pred), 4),
        "f1":              round(f1_score(y_test, y_pred), 4),
        "accuracy":        round((y_pred == y_test).mean(), 4),
    }

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<20} {v}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    return metrics


# ── Training orchestrator ──────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, feature_names: list) -> dict:
    """Train all 3 models, log to MLflow, save to disk, return metrics dict."""

    results = {}

    model_configs = [
        ("Logistic Regression", get_logistic_regression()),
        ("Random Forest",       get_random_forest()),
        ("XGBoost",             get_xgboost()),
    ]

    for name, model in model_configs:
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)
            results[name] = metrics

            # Log params & metrics to MLflow
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            mlflow.log_param("model_type", name)

            # Save model artifact
            safe_name = name.lower().replace(" ", "_")
            model_path = MODELS_DIR / f"{safe_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path))

            logger.success(f"Done: {name} | ROC-AUC: {metrics['roc_auc']}")

    # Save metrics summary
    metrics_path = MODELS_DIR / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_model(model_name: str = "xgboost"):
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Cross-validation ───────────────────────────────────────────────────────────
def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """Run stratified k-fold cross validation."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores  = cross_val_score(model, X, y, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
    return {
        "mean_auc":  round(auc_scores.mean(), 4),
        "std_auc":   round(auc_scores.std(), 4),
        "all_folds": auc_scores.tolist(),
    }
