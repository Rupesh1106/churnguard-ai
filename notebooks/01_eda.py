"""
01_eda.py — Exploratory Data Analysis Script
Run: python notebooks/01_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import generate_churn_dataset

# Style
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13})
REPORT = Path("reports"); REPORT.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
raw_path = Path("data/raw/telecom_churn.csv")
if not raw_path.exists():
    df = generate_churn_dataset(10_000)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
else:
    df = pd.read_csv(raw_path)

print("=" * 60)
print(f"Dataset shape : {df.shape}")
print(f"Churn rate    : {df['churn'].mean():.1%}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("=" * 60)
print(df.describe().T.round(2))

# ── 1. Churn distribution ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Telecom Churn — Exploratory Data Analysis", fontsize=16, fontweight="bold")

churn_counts = df["churn"].value_counts()
axes[0,0].bar(["Retained", "Churned"], churn_counts.values,
              color=["#38ef7d", "#ff416c"], edgecolor="white")
axes[0,0].set_title("Overall Churn Distribution")
axes[0,0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v + 50, f"{v:,}\n({v/len(df):.1%})", ha="center", fontweight="bold")

# Churn by contract
contract_churn = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
axes[0,1].bar(contract_churn.index, contract_churn.values,
              color=["#ff416c", "#f7971e", "#38ef7d"])
axes[0,1].set_title("Churn Rate by Contract Type")
axes[0,1].set_ylabel("Churn Rate")
for i, (c, v) in enumerate(contract_churn.items()):
    axes[0,1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

# Monthly charges distribution
df[df["churn"]==0]["monthly_charges"].hist(ax=axes[0,2], alpha=0.6, color="#38ef7d", label="Retained", bins=30)
df[df["churn"]==1]["monthly_charges"].hist(ax=axes[0,2], alpha=0.6, color="#ff416c", label="Churned", bins=30)
axes[0,2].set_title("Monthly Charges Distribution")
axes[0,2].legend()

# Tenure vs Churn
axes[1,0].hist([df[df["churn"]==0]["tenure_months"], df[df["churn"]==1]["tenure_months"]],
               label=["Retained", "Churned"], color=["#38ef7d","#ff416c"],
               bins=20, alpha=0.7, stacked=False)
axes[1,0].set_title("Tenure Distribution by Churn")
axes[1,0].legend()

# Support tickets
support_churn = df.groupby("support_tickets")["churn"].mean()
axes[1,1].bar(support_churn.index, support_churn.values, color="#667eea")
axes[1,1].set_title("Churn Rate by Support Tickets")
axes[1,1].set_xlabel("Support Tickets"); axes[1,1].set_ylabel("Churn Rate")
axes[1,1].axhline(df["churn"].mean(), color="red", linestyle="--", label="Overall avg")
axes[1,1].legend()

# Payment method churn
pay_churn = df.groupby("payment_method")["churn"].mean().sort_values(ascending=True)
axes[1,2].barh(pay_churn.index, pay_churn.values, color="#764ba2")
axes[1,2].set_title("Churn Rate by Payment Method")
axes[1,2].set_xlabel("Churn Rate")

plt.tight_layout()
plt.savefig(REPORT / "01_eda_overview.png", bbox_inches="tight")
plt.close()
print(f"\n✅ EDA plot saved → {REPORT / '01_eda_overview.png'}")

# ── 2. Correlation heatmap ────────────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdYlGn", center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT / "02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print(f"✅ Heatmap saved → {REPORT / '02_correlation_heatmap.png'}")

# ── 3. Revenue loss estimation ────────────────────────────────────────────────
churned_df  = df[df["churn"] == 1]
retained_df = df[df["churn"] == 0]

monthly_loss = churned_df["monthly_charges"].sum()
annual_loss  = monthly_loss * 12
avg_cac      = 350  # assumed customer acquisition cost in ₹

print("\n" + "=" * 50)
print("💸 Business Impact Analysis")
print("=" * 50)
print(f"Churned customers    : {len(churned_df):,}")
print(f"Monthly revenue loss : ₹{monthly_loss:,.0f}")
print(f"Annual revenue loss  : ₹{annual_loss:,.0f}")
print(f"CAC waste (at ₹{avg_cac}): ₹{len(churned_df) * avg_cac:,.0f}")
print(f"Total annual impact  : ₹{annual_loss + len(churned_df)*avg_cac:,.0f}")

# ── 4. Key insights ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🔑 Key Business Insights")
print("=" * 50)

mtm_churn = df[df["contract"]=="Month-to-month"]["churn"].mean()
annual_churn = df[df["contract"]=="One year"]["churn"].mean()
print(f"Month-to-month churn : {mtm_churn:.1%}")
print(f"Annual contract churn: {annual_churn:.1%}")
print(f"→ Annual contract reduces churn by {(mtm_churn-annual_churn)/mtm_churn:.0%}")

high_charge = df[df["monthly_charges"] > df["monthly_charges"].quantile(0.75)]["churn"].mean()
low_charge  = df[df["monthly_charges"] <= df["monthly_charges"].median()]["churn"].mean()
print(f"\nHigh-charge (top 25%) churn : {high_charge:.1%}")
print(f"Low-charge (bottom 50%) churn: {low_charge:.1%}")
print(f"→ High-charge customers have {high_charge/low_charge:.1f}x churn risk")

high_tickets = df[df["support_tickets"] >= 3]["churn"].mean()
print(f"\n>3 support tickets churn rate: {high_tickets:.1%}")
print(f"→ Users with >3 complaints have {high_tickets:.0%} churn rate")
