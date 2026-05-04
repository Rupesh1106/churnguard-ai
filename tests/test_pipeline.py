"""
test_pipeline.py — Unit Tests for Core Pipeline Components
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_churn_dataset
from src.feature_engineering import (
    encode_categoricals, build_rfm_features,
    build_behavioural_features, run_full_pipeline,
    FEATURE_COLUMNS, TARGET_COLUMN,
)


@pytest.fixture
def sample_df():
    return generate_churn_dataset(200)


class TestDataGenerator:
    def test_shape(self, sample_df):
        assert sample_df.shape[0] == 200
        assert "churn" in sample_df.columns

    def test_churn_binary(self, sample_df):
        assert set(sample_df["churn"].unique()).issubset({0, 1})

    def test_no_nulls(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0

    def test_churn_rate_realistic(self, sample_df):
        rate = sample_df["churn"].mean()
        assert 0.10 <= rate <= 0.60, f"Churn rate {rate:.1%} seems unrealistic"

    def test_tenure_range(self, sample_df):
        assert sample_df["tenure_months"].between(1, 72).all()

    def test_charges_positive(self, sample_df):
        assert (sample_df["monthly_charges"] > 0).all()
        assert (sample_df["total_charges"] > 0).all()


class TestFeatureEngineering:
    def test_encode_categoricals(self, sample_df):
        df_enc = encode_categoricals(sample_df.copy())
        assert df_enc["gender"].dtype in [np.int64, np.float64, int]
        assert df_enc["contract"].isin([0, 1, 2]).all()

    def test_rfm_features_created(self, sample_df):
        df_enc = encode_categoricals(sample_df.copy())
        df_rfm = build_rfm_features(df_enc)
        for col in ["recency_score", "frequency_score", "monetary_score", "rfm_churn_risk"]:
            assert col in df_rfm.columns, f"Missing: {col}"

    def test_rfm_values_bounded(self, sample_df):
        df_enc = encode_categoricals(sample_df.copy())
        df_rfm = build_rfm_features(df_enc)
        assert df_rfm["rfm_churn_risk"].between(0, 1).all()

    def test_behavioural_features(self, sample_df):
        df_enc = encode_categoricals(sample_df.copy())
        df_rfm = build_rfm_features(df_enc)
        df_beh = build_behavioural_features(df_rfm)
        for col in ["support_intensity", "high_complaint_flag", "payment_reliability"]:
            assert col in df_beh.columns

    def test_full_pipeline_no_nulls(self, sample_df):
        df_eng = run_full_pipeline(sample_df.copy())
        assert df_eng.isnull().sum().sum() == 0, "NaNs found in engineered dataset"

    def test_full_pipeline_shape(self, sample_df):
        df_eng = run_full_pipeline(sample_df.copy())
        assert df_eng.shape[0] == 200
        assert df_eng.shape[1] > sample_df.shape[1]

    def test_target_preserved(self, sample_df):
        df_eng = run_full_pipeline(sample_df.copy())
        assert TARGET_COLUMN in df_eng.columns
        assert set(df_eng[TARGET_COLUMN].unique()).issubset({0, 1})


class TestBusinessLogic:
    """Validate that churn drivers are correctly captured."""

    def test_mtm_higher_churn(self):
        """Month-to-month contracts should produce higher churn than 2-year."""
        large_df = generate_churn_dataset(5000)
        mtm  = large_df[large_df["contract"] == "Month-to-month"]["churn"].mean()
        two_yr = large_df[large_df["contract"] == "Two year"]["churn"].mean()
        assert mtm > two_yr, "Month-to-month should churn more than two-year"

    def test_high_tickets_churn(self):
        """High support tickets should correlate with higher churn."""
        large_df = generate_churn_dataset(5000)
        high_tick = large_df[large_df["support_tickets"] >= 3]["churn"].mean()
        no_tick   = large_df[large_df["support_tickets"] == 0]["churn"].mean()
        assert high_tick > no_tick
