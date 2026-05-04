# 🛡️ ChurnGuard AI — End-to-End Customer Churn Prediction System

> **Advanced ML system** combining XGBoost, SHAP explainability, real-time API, and a Streamlit web app for actionable churn prevention.

---

## 🎯 Problem Statement

Telecom companies lose **15–25% of customers annually** to churn, costing billions in revenue and wasted customer acquisition costs (CAC).

| Metric | Impact |
|--------|--------|
| Monthly revenue loss | ₹ tracked per churned customer |
| CAC waste | ~₹350 per churned customer |
| Annual impact | Estimated in-app from real data |

**This system predicts churn before it happens** and explains *why*, enabling targeted retention campaigns.

---

## 🏗️ Architecture

```
Raw CSV / API Data
       ↓
  Data Generator (src/data_generator.py)
       ↓
  Feature Engineering (src/feature_engineering.py)
  RFM + Behavioural + Interaction Features
       ↓
  Model Training (src/model.py)
  Logistic Regression → Random Forest → XGBoost
       ↓
  SHAP Explainability (src/shap_explain.py)
       ↓
  ┌────────────┬─────────────┐
  │ FastAPI    │ Streamlit   │
  │ (api/)     │ (app/)      │
  └────────────┴─────────────┘
       ↓
  MLflow Tracking + Auto-Retraining (GitHub Actions)
       ↓
  Docker / Cloud Deployment
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python src/pipeline.py
```
This will:
- Generate 10,000 synthetic customers
- Engineer 35+ features (RFM, behavioural, interaction)
- Train Logistic Regression, Random Forest, XGBoost
- Compute SHAP values and save plots
- Log experiments to MLflow

### 3. Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### 4. Launch FastAPI (optional)
```bash
uvicorn api.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### 5. Run EDA analysis
```bash
python notebooks/01_eda.py
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up --build
```
- Streamlit → http://localhost:8501
- FastAPI   → http://localhost:8000/docs

---

## 📊 Features

### 🔬 Feature Engineering (35+ features)
| Category | Features |
|----------|----------|
| **RFM** | recency_score, frequency_score, monetary_score, rfm_churn_risk |
| **Behavioural** | support_intensity, high_complaint_flag, payment_reliability, data_per_charge |
| **Interaction** | mtm_echeck_risk, fiber_no_support, high_charge_new_customer |
| **Raw** | tenure, monthly_charges, support_tickets, late_payments |

### 🤖 Models
| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression | Baseline | Interpretable |
| Random Forest | +5–8% | Handles non-linearity |
| **XGBoost** | **Best** | Production model |

### 🧠 Explainability
- **Global**: SHAP beeswarm + bar plots
- **Local**: Per-customer waterfall + force plots
- **Business**: Human-readable churn reasons in Streamlit

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single customer churn prediction |
| `/predict/batch` | POST | Bulk predictions |
| `/ab-test/summary` | GET | A/B test metrics per model |
| `/model/metrics` | GET | ROC-AUC, Precision, Recall |
| `/retrain` | POST | Trigger background retraining |

---

## 💡 Key Business Insights

> "Month-to-month contracts have **~3x higher churn** than two-year contracts"

> "Customers with >3 support tickets have **65%+ churn rate**"

> "High-charge customers (top 25%) are **2x more likely** to churn"

> "Electronic check payment method correlates with **10% higher churn probability**"

### Retention Actions
- 📄 Offer annual contracts with 15% discount to month-to-month customers
- 🎧 Fast-track support for customers with 3+ tickets
- 💰 Loyalty pricing for high-tenure, high-charge customers
- 🎁 Welcome bonus for customers in first 6 months (churn hotspot)

---

## 🔄 Automated Retraining

The GitHub Actions workflow (`.github/workflows/retrain.yml`) triggers **every Sunday at 02:00 UTC**:
1. Run pytest test suite (gate)
2. Retrain all models on fresh data
3. Upload model artifacts to GitHub

---

## 📁 Project Structure

```
churn-prediction/
├── src/
│   ├── data_generator.py       # Synthetic data generation
│   ├── feature_engineering.py  # RFM + 35+ feature pipeline
│   ├── model.py                # LR + RF + XGBoost training
│   ├── shap_explain.py         # SHAP global & local plots
│   └── pipeline.py             # Master orchestrator
├── api/
│   └── main.py                 # FastAPI real-time API
├── app/
│   └── streamlit_app.py        # 4-page Streamlit web app
├── notebooks/
│   └── 01_eda.py               # EDA + business insights
├── tests/
│   └── test_pipeline.py        # pytest unit + business tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/
│   └── retrain.yml             # Automated retraining CI/CD
└── requirements.txt
```
