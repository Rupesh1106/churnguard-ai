"""
data_generator.py — Synthetic Telecom Churn Dataset Generator
Generates realistic telecom customer data with behavioural patterns
that align with real-world churn dynamics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_churn_dataset(n_customers: int = 10000) -> pd.DataFrame:
    """
    Generate a realistic telecom churn dataset with behavioural features.

    Features engineered to mirror real-world churn patterns:
    - High monthly charges → higher churn
    - Month-to-month contracts → higher churn
    - Electronic check payment → higher churn
    - Fewer support calls + low tenure → higher churn
    """
    customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n_customers + 1)]

    gender           = np.random.choice(["Male", "Female"], n_customers)
    senior_citizen   = np.random.binomial(1, 0.16, n_customers)
    partner          = np.random.choice(["Yes", "No"], n_customers)
    dependents       = np.random.choice(["Yes", "No"], n_customers, p=[0.30, 0.70])

    contract         = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_customers, p=[0.55, 0.24, 0.21]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n_customers, p=[0.34, 0.44, 0.22]
    )
    phone_service    = np.random.choice(["Yes", "No"], n_customers, p=[0.90, 0.10])
    online_security  = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_customers)
    )
    tech_support     = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_customers)
    )
    streaming_tv     = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n_customers)
    )
    paperless_billing = np.random.choice(["Yes", "No"], n_customers, p=[0.59, 0.41])
    payment_method    = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )

    tenure_months    = np.random.exponential(scale=30, size=n_customers).clip(1, 72).astype(int)
    monthly_charges  = np.random.normal(65, 30, n_customers).clip(18, 120).round(2)
    total_charges    = (tenure_months * monthly_charges + np.random.normal(0, 50, n_customers)).clip(18).round(2)

    calls_per_month  = np.random.poisson(lam=4, size=n_customers)
    data_usage_gb    = np.where(
        internet_service == "No", 0,
        np.random.exponential(scale=20, size=n_customers).clip(0, 150).round(2)
    )
    support_tickets  = np.random.poisson(lam=0.8, size=n_customers)
    late_payments    = np.random.binomial(3, 0.15, n_customers)
    avg_call_duration = np.random.normal(8, 4, n_customers).clip(1, 30).round(2)  # minutes

    churn_prob = np.zeros(n_customers)

    churn_prob += np.where(contract == "Month-to-month", 0.30, 0.0)
    churn_prob += np.where(contract == "One year",        0.08, 0.0)

    churn_prob += (monthly_charges - 65) / 200

    # Tenure effect — new customers churn more
    churn_prob -= tenure_months / 200

    # Support tickets — dissatisfied customers churn
    churn_prob += support_tickets * 0.08

    # Late payments flag
    churn_prob += late_payments * 0.05

    # Internet service type
    churn_prob += np.where(internet_service == "Fiber optic", 0.12, 0.0)

    # Payment method — electronic check users churn more
    churn_prob += np.where(payment_method == "Electronic check", 0.10, 0.0)

    # No tech support → more churn
    churn_prob += np.where(tech_support == "No", 0.07, 0.0)

    # Clip to valid probability range and add noise
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, n_customers), 0.02, 0.95)
    churn      = np.random.binomial(1, churn_prob, n_customers)

    df = pd.DataFrame({
        "customer_id":        customer_ids,
        "gender":             gender,
        "senior_citizen":     senior_citizen,
        "partner":            partner,
        "dependents":         dependents,
        "tenure_months":      tenure_months,
        "phone_service":      phone_service,
        "internet_service":   internet_service,
        "online_security":    online_security,
        "tech_support":       tech_support,
        "streaming_tv":       streaming_tv,
        "contract":           contract,
        "paperless_billing":  paperless_billing,
        "payment_method":     payment_method,
        "monthly_charges":    monthly_charges,
        "total_charges":      total_charges,
        "calls_per_month":    calls_per_month,
        "data_usage_gb":      data_usage_gb,
        "support_tickets":    support_tickets,
        "late_payments":      late_payments,
        "avg_call_duration":  avg_call_duration,
        "churn":              churn,
    })

    return df


if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_churn_dataset(10_000)
    df.to_csv(out_dir / "telecom_churn.csv", index=False)
    print(f"✅ Dataset generated: {len(df):,} rows  |  Churn rate: {df['churn'].mean():.1%}")
    print(df.head())
