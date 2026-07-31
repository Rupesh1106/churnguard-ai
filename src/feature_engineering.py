"""
feature_engineering.py — Advanced Feature Engineering
Creates RFM, behavioural, and interaction features that differentiate
this project from a basic ML pipeline.
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── Categorical encoding map ───────────────────────────────────────────────────────
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_MAP = {"No": 0, "DSL": 1, "Fiber optic": 2}
PAYMENT_MAP = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}
SECURITY_MAP = {"No": 0, "Yes": 1, "No internet service": -1}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all categorical columns into numeric representations."""
    df = df.copy()

    df["gender"]            = df["gender"].map(BINARY_MAP)
    df["partner"]           = df["partner"].map(BINARY_MAP)
    df["dependents"]        = df["dependents"].map(BINARY_MAP)
    df["phone_service"]     = df["phone_service"].map(BINARY_MAP)
    df["paperless_billing"] = df["paperless_billing"].map(BINARY_MAP)
    df["contract"]          = df["contract"].map(CONTRACT_MAP)
    df["internet_service"]  = df["internet_service"].map(INTERNET_MAP)
    df["payment_method"]    = df["payment_method"].map(PAYMENT_MAP)
    df["online_security"]   = df["online_security"].map(SECURITY_MAP)
    df["tech_support"]      = df["tech_support"].map(SECURITY_MAP)
    df["streaming_tv"]      = df["streaming_tv"].map(SECURITY_MAP)

    return df


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM-inspired features adapted for telecom:
      Recency   → inverse of tenure (newer = higher churn risk)
      Frequency → calls per month (engagement proxy)
      Monetary  → monthly charges (revenue & churn risk signal)
    """
    df = df.copy()

    df["recency_score"] = 1 / (df["tenure_months"] + 1)
    df["frequency_score"] = df["calls_per_month"] / (df["calls_per_month"].max() + 1e-9)
    df["monetary_score"] = df["monthly_charges"] / (df["monthly_charges"].max() + 1e-9)

    df["rfm_churn_risk"] = (
        0.4 * df["recency_score"] +
        0.2 * (1 - df["frequency_score"]) +
        0.4 * df["monetary_score"]
    )

    return df


def build_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced behavioural features:
      - Charge-per-minute (value perception)
      - Support intensity (complaint frequency)
      - Payment reliability score
      - Data efficiency (data vs charge ratio)
    """
    df = df.copy()

    total_minutes = df["avg_call_duration"] * df["calls_per_month"] * df["tenure_months"] + 1
    df["charge_per_minute"] = df["total_charges"] / total_minutes

    df["support_intensity"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["high_complaint_flag"] = (df["support_tickets"] >= 3).astype(int)
    df["payment_reliability"] = 1 - (df["late_payments"] / (df["tenure_months"] + 1))
    df["data_per_charge"] = df["data_usage_gb"] / (df["monthly_charges"] + 1)
    df["avg_calls_per_week"] = df["calls_per_month"] / 4.33
    df["revenue_risk_proxy"] = df["monthly_charges"] * df["rfm_churn_risk"]

    return df


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction terms that capture compound risk signals.
    Example: month-to-month contract AND high charges = very high risk.
    """
    df = df.copy()

    df["high_charge_new_customer"] = (
        (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)) &
        (df["tenure_months"] < 12)
    ).astype(int)

    if "tech_support" in df.columns:
        df["fiber_no_support"] = (
            (df["internet_service"] == 2) & (df["tech_support"] == 0)
        ).astype(int)

    df["mtm_echeck_risk"] = (
        (df["contract"] == 0) & (df["payment_method"] == 0)
    ).astype(int)

    df["longtime_frustrated"] = (
        (df["tenure_months"] > 36) & (df["support_tickets"] > 2)
    ).astype(int)

    return df


def run_full_pipeline(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Master pipeline: encode → RFM → behavioural → interactions.
    Returns a fully engineered DataFrame ready for model training.
    """
    logger.info("Starting feature engineering pipeline...")
    df = encode_categoricals(df)
    df = build_rfm_features(df)
    df = build_behavioural_features(df)
    df = build_interaction_features(df)

    if drop_id and "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ── Feature lists for downstream use ──────────────────────────────────────────
FEATURE_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents",
    "tenure_months", "phone_service", "internet_service",
    "online_security", "tech_support", "streaming_tv",
    "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges",
    "calls_per_month", "data_usage_gb", "support_tickets",
    "late_payments", "avg_call_duration",
    # Engineered
    "recency_score", "frequency_score", "monetary_score", "rfm_churn_risk",
    "charge_per_minute", "support_intensity", "high_complaint_flag",
    "payment_reliability", "data_per_charge", "avg_calls_per_week",
    "revenue_risk_proxy", "high_charge_new_customer",
    "fiber_no_support", "mtm_echeck_risk", "longtime_frustrated",
]

TARGET_COLUMN = "churn"
