import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="ChurnGuard AI | Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    text-align: center;
}

.risk-high   { background: linear-gradient(135deg,#ff416c,#ff4b2b); color:white; border-radius:12px; padding:1rem; }
.risk-medium { background: linear-gradient(135deg,#f7971e,#ffd200); color:#111;  border-radius:12px; padding:1rem; }
.risk-low    { background: linear-gradient(135deg,#11998e,#38ef7d); color:white; border-radius:12px; padding:1rem; }

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 12px;
    padding: 0.75rem 2rem; font-weight: 600; font-size: 1rem;
    width: 100%; transition: all 0.3s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.5); }

.section-header {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")


@st.cache_resource
def load_model_cached(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_shap_data():
    """Load pre-computed SHAP values array saved by the pipeline."""
    path = MODELS_DIR / "shap_values.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_feature_columns():
    path = MODELS_DIR / "feature_columns.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


@st.cache_data
def load_dataset():
    p = Path("data/processed/churn_engineered.csv")
    if p.exists():
        return pd.read_csv(p)
    return None


def get_risk_color(prob):
    if prob >= 0.70: return "risk-high",   "🔴 HIGH RISK"
    if prob >= 0.40: return "risk-medium", "🟡 MEDIUM RISK"
    return "risk-low", "🟢 LOW RISK"


st.sidebar.markdown("## 🛡️ ChurnGuard AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Predict Churn", "📊 Analytics Dashboard", "🧠 Model Insights", "📋 Batch Prediction"],
)
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["xgboost", "random_forest", "logistic_regression"],
    index=0,
)
st.sidebar.info("💡 XGBoost gives best ROC-AUC performance")

model = load_model_cached(model_choice)
shap_data = load_shap_data()
feature_cols = load_feature_columns()
df_data = load_dataset()

models_ready = model is not None


if page == "🎯 Predict Churn":
    st.markdown('<h1 style="text-align:center">🛡️ ChurnGuard AI — Customer Churn Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;opacity:0.7">Enter customer details to predict churn probability with AI explainability</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not models_ready:
        st.error("⚠️ Models not found. Run `python src/pipeline.py` first to train models.")
        st.stop()

    # ── Input form ─────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            gender           = st.selectbox("Gender",          ["Male", "Female"])
            senior_citizen   = st.selectbox("Senior Citizen",  [0, 1])
            partner          = st.selectbox("Has Partner",     ["Yes", "No"])
            dependents       = st.selectbox("Has Dependents",  ["Yes", "No"])
        with c2:
            tenure_months    = st.slider("Tenure (months)",    1, 72, 12)
            contract         = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
            payment_method   = st.selectbox("Payment Method",  ["Electronic check", "Mailed check",
                                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
        with c3:
            monthly_charges  = st.slider("Monthly Charges (₹)", 18.0, 120.0, 65.0, 0.5)
            total_charges    = st.number_input("Total Charges (₹)", value=float(tenure_months * monthly_charges), min_value=0.0)
            support_tickets  = st.slider("Support Tickets",    0, 10, 1)
            late_payments    = st.slider("Late Payments",      0, 5, 0)

        st.markdown('<div class="section-header">📡 Usage Behaviour</div>', unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        with u1:
            calls_per_month    = st.slider("Calls/Month",       0, 20, 4)
            avg_call_duration  = st.slider("Avg Call Duration (min)", 1.0, 30.0, 8.0)
        with u2:
            data_usage_gb      = st.slider("Data Usage (GB)",   0.0, 150.0, 20.0)
            online_security    = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        with u3:
            tech_support       = st.selectbox("Tech Support",   ["Yes", "No", "No internet service"])
            streaming_tv       = st.selectbox("Streaming TV",   ["Yes", "No", "No internet service"])
            paperless_billing  = st.selectbox("Paperless Billing", ["Yes", "No"])

        submitted = st.form_submit_button("🔮 Predict Churn Probability")

    if submitted:
        from src.feature_engineering import run_full_pipeline

        raw_input = {
            "gender": gender, "senior_citizen": senior_citizen,
            "partner": partner, "dependents": dependents,
            "tenure_months": tenure_months, "phone_service": "Yes",
            "internet_service": internet_service, "online_security": online_security,
            "tech_support": tech_support, "streaming_tv": streaming_tv,
            "contract": contract, "paperless_billing": paperless_billing,
            "payment_method": payment_method, "monthly_charges": monthly_charges,
            "total_charges": total_charges, "calls_per_month": calls_per_month,
            "data_usage_gb": data_usage_gb, "support_tickets": support_tickets,
            "late_payments": late_payments, "avg_call_duration": avg_call_duration,
        }

        df_input   = pd.DataFrame([raw_input])
        df_eng     = run_full_pipeline(df_input.copy(), drop_id=False)
        available  = [f for f in feature_cols if f in df_eng.columns]
        X          = df_eng[available]
        prob       = float(model.predict_proba(X)[0, 1])
        pred       = int(prob >= 0.5)
        css_class, risk_label = get_risk_color(prob)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(f"""
            <div class="{css_class}" style="text-align:center">
                <h2>{risk_label}</h2>
                <h1 style="font-size:3rem">{prob:.1%}</h1>
                <p>Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 40],   "color": "#11998e"},
                    {"range": [40, 70],  "color": "#f7971e"},
                    {"range": [70, 100], "color": "#ff416c"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": prob * 100},
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP Reasons from pre-computed global values (fast)
        if shap_data is not None:
            try:
                from src.shap_explain import get_top_churn_reasons, format_churn_reasons_text
                import numpy as np

                sv_array = shap_data["values"]  # shape (n_sample, n_features)
                feat_names = shap_data["feature_names"]

                # Compute per-input SHAP inline using the loaded XGBoost model
                from src.shap_explain import load_shap_explainer, compute_shap_values
                import pandas as pd
                xgb_model = load_model_cached("xgboost")
                # Build explainer on-the-fly (fast for single row)
                bg = pd.DataFrame(sv_array[:50], columns=feat_names)
                live_explainer = load_shap_explainer(xgb_model, bg[available], "xgboost")
                live_sv = compute_shap_values(live_explainer, X)
                reasons = get_top_churn_reasons(live_sv, X, 0, top_n=5)

                st.markdown("---")
                st.markdown("### Why is this customer at risk?")
                st.markdown(format_churn_reasons_text(reasons))

                reason_df = pd.DataFrame(reasons)
                fig2 = px.bar(
                    reason_df, x="shap_value", y="feature", orientation="h",
                    color="shap_value", color_continuous_scale=["#11998e", "#f7971e", "#ff416c"],
                    title="SHAP Contribution per Feature",
                    labels={"shap_value": "SHAP Value", "feature": "Feature"},
                )
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # Retention recommendations
        st.markdown("---")
        st.markdown("### 💡 Retention Recommendations")
        recs = []
        if contract == "Month-to-month":    recs.append("📄 Offer annual contract with 15% discount")
        if monthly_charges > 80:            recs.append("💰 Provide loyalty pricing — charges above avg")
        if support_tickets >= 3:            recs.append("🎧 Assign dedicated support agent immediately")
        if late_payments >= 2:             recs.append("💳 Offer flexible payment plan or EMI option")
        if tenure_months < 6:              recs.append("🎁 Early-tenure loyalty reward / welcome bonus")
        if not recs:                        recs.append("✅ Customer is low-risk — maintain engagement")
        for r in recs:
            st.success(r)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Churn Analytics Dashboard")

    if df_data is None:
        st.error("Run the pipeline first: `python src/pipeline.py`")
        st.stop()

    raw = pd.read_csv("data/raw/telecom_churn.csv") if Path("data/raw/telecom_churn.csv").exists() else df_data

    # KPI Row
    total        = len(raw)
    churned      = raw["churn"].sum()
    churn_rate   = churned / total
    revenue_risk = (raw[raw["churn"] == 1]["monthly_charges"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Total Customers", f"{total:,}")
    k2.metric("🚨 Churned",         f"{churned:,}", delta=f"{churn_rate:.1%}")
    k3.metric("📉 Churn Rate",      f"{churn_rate:.1%}")
    k4.metric("💸 Monthly Revenue at Risk", f"₹{revenue_risk:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Churn by contract
        contract_churn = raw.groupby("contract")["churn"].mean().reset_index()
        fig = px.bar(contract_churn, x="contract", y="churn",
                     title="Churn Rate by Contract Type",
                     color="churn", color_continuous_scale="RdYlGn_r")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by internet service
        inet_churn = raw.groupby("internet_service")["churn"].mean().reset_index()
        fig = px.pie(inet_churn, values="churn", names="internet_service",
                     title="Churn Distribution by Internet Service",
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Monthly charges distribution by churn
        fig = px.box(raw, x="churn", y="monthly_charges",
                     title="Monthly Charges vs Churn",
                     color="churn", color_discrete_map={0: "#11998e", 1: "#ff416c"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Tenure distribution
        fig = px.histogram(raw, x="tenure_months", color="churn",
                           title="Tenure Distribution by Churn",
                           barmode="overlay", nbins=30,
                           color_discrete_map={0: "#38ef7d", 1: "#ff416c"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # Cohort analysis
    st.markdown("### 🔬 Cohort Churn Analysis")
    raw["tenure_cohort"] = pd.cut(raw["tenure_months"],
                                   bins=[0,6,12,24,36,72],
                                   labels=["0-6m","6-12m","12-24m","24-36m","36m+"])
    cohort = raw.groupby("tenure_cohort", observed=True)["churn"].agg(["mean","count"]).reset_index()
    cohort.columns = ["Cohort", "Churn Rate", "Customers"]
    fig = px.bar(cohort, x="Cohort", y="Churn Rate", text="Customers",
                 title="Churn Rate by Tenure Cohort",
                 color="Churn Rate", color_continuous_scale="RdYlGn_r")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown("## 🧠 Model Performance & SHAP Insights")

    import json
    metrics_path = Path("models/metrics_summary.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        metric_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        metric_df = metric_df.drop(columns=["model"], errors="ignore")

        st.markdown("### 📊 Model Comparison")
        st.dataframe(metric_df.style.highlight_max(axis=0, color="#667eea"), use_container_width=True)

        fig = px.bar(metric_df, x="Model", y=["roc_auc", "precision", "recall", "f1"],
                     barmode="group", title="Model Metrics Comparison",
                     color_discrete_sequence=["#667eea", "#11998e", "#ff416c", "#f7971e"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run the pipeline to generate model metrics.")

    # SHAP plots
    shap_bar = Path("reports/shap_bar.png")
    shap_summary = Path("reports/shap_summary.png")
    if shap_bar.exists():
        st.markdown("### 🎯 Global SHAP Feature Importance")
        st.image(str(shap_bar), use_column_width=True)
    if shap_summary.exists():
        st.markdown("### 🌊 SHAP Beeswarm Summary")
        st.image(str(shap_summary), use_column_width=True)

    if not shap_bar.exists():
        st.info("SHAP plots will appear here after running the pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Prediction":
    st.markdown("## 📋 Batch Customer Churn Prediction")
    st.markdown("Upload a CSV file to predict churn for multiple customers at once.")

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded:
        if not models_ready:
            st.error("Run the pipeline first.")
            st.stop()

        df_up = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df_up):,} customers**")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("🔮 Run Batch Prediction"):
            from src.feature_engineering import run_full_pipeline
            with st.spinner("Predicting..."):
                df_eng  = run_full_pipeline(df_up.copy(), drop_id=False)
                avail   = [f for f in feature_cols if f in df_eng.columns]
                X_batch = df_eng[avail]
                probs   = model.predict_proba(X_batch)[:, 1]
                df_up["churn_probability"] = probs.round(4)
                df_up["churn_prediction"]  = (probs >= 0.5).astype(int)
                df_up["risk_level"]        = pd.cut(probs, bins=[0,.4,.7,1],
                                                     labels=["LOW","MEDIUM","HIGH"])

            st.success(f"✅ Predicted {len(df_up):,} customers")
            st.dataframe(df_up[["churn_probability","churn_prediction","risk_level"]].head(20),
                         use_container_width=True)

            csv = df_up.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results", csv, "churn_predictions.csv", "text/csv")

            fig = px.histogram(df_up, x="churn_probability", color="risk_level",
                               title="Churn Probability Distribution",
                               color_discrete_map={"LOW":"#11998e","MEDIUM":"#f7971e","HIGH":"#ff416c"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Upload a CSV with the same columns as the training data to get predictions.")
