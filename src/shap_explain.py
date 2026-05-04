"""
shap_explain.py — SHAP-based Model Explainability
Computes global & local SHAP values, generates plots and
provides human-readable churn reason summaries.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_shap_explainer(model, X_background: pd.DataFrame, model_type: str = "xgboost"):
    """Create the appropriate SHAP explainer for the model type."""
    if model_type == "xgboost":
        # For XGBoost 3.x, use predict_proba to ensure we get probability-space SHAP values
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        bg_sample = X_background.sample(min(100, len(X_background)), random_state=42)
        explainer = shap.Explainer(predict_fn, bg_sample)
    elif model_type == "random_forest":
        explainer = shap.TreeExplainer(model)
    else:
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        background = shap.sample(X_background, 100)
        explainer = shap.KernelExplainer(predict_fn, background)

    logger.info(f"SHAP explainer created for {model_type}")
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """Compute SHAP values; returns shap.Explanation object."""
    shap_values = explainer(X)
    return shap_values


def plot_global_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    """Beeswarm summary plot — global feature importance with direction."""
    plt.figure(figsize=(12, 8))
    # SHAP 0.49 Explanation objects have .values
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap.summary_plot(values, X, show=False, plot_size=None)
    plt.title("SHAP Feature Importance — Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved → {path}")
    return path


def plot_bar_importance(shap_values, X: pd.DataFrame, save_path: str = None):
    """Bar plot of mean absolute SHAP values — top 15 features."""
    plt.figure(figsize=(10, 6))
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap.summary_plot(values, X, plot_type="bar", show=False,
                      max_display=15, plot_size=None)
    plt.title("Top Churn Drivers (Mean |SHAP|)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_waterfall_single(shap_values, idx: int, save_path: str = None):
    """Waterfall plot for a single customer — explains WHY they are predicted to churn."""
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.title(f"Churn Explanation — Customer #{idx}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / f"shap_waterfall_{idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_force_single(explainer, shap_values, idx: int, X: pd.DataFrame, save_path: str = None):
    """HTML force plot for a single customer prediction."""
    # Ensure expected_value is compatible
    ev = explainer.expected_value if hasattr(explainer, "expected_value") else shap_values.base_values[idx]
    html = shap.force_plot(
        ev,
        shap_values.values[idx],
        X.iloc[idx],
        show=False,
    )
    path = save_path or str(REPORTS_DIR / f"shap_force_{idx}.html")
    shap.save_html(path, html)
    return path


def get_top_churn_reasons(shap_values, X: pd.DataFrame, idx: int, top_n: int = 5) -> list[dict]:
    """
    Return the top N reasons (feature + contribution) why a customer is predicted to churn.
    """
    sv = shap_values.values[idx] if hasattr(shap_values, "values") else shap_values[idx]
    feat = X.columns.tolist()

    reasons = sorted(
        [{"feature": f, "shap_value": round(float(v), 4), "feature_value": float(X.iloc[idx][f])}
         for f, v in zip(feat, sv)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    for r in reasons:
        r["direction"] = "↑ Increases" if r["shap_value"] > 0 else "↓ Decreases"
        r["churn_risk"] = "HIGH" if r["shap_value"] > 0 else "LOW"

    return reasons[:top_n]


def format_churn_reasons_text(reasons: list[dict]) -> str:
    """Human-readable churn reason summary for Streamlit display."""
    lines = ["**🔍 Top Reasons for Churn Prediction:**\n"]
    for i, r in enumerate(reasons, 1):
        direction = "🔴" if r["shap_value"] > 0 else "🟢"
        lines.append(
            f"{i}. {direction} **{r['feature'].replace('_', ' ').title()}** "
            f"= `{r['feature_value']:.2f}` "
            f"({r['direction']} churn risk, SHAP: {r['shap_value']:+.3f})"
        )
    return "\n".join(lines)
