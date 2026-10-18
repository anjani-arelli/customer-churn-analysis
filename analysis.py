"""Customer churn analysis on synthetic telecom data.

Does three things: explores what separates churners from stayers, builds a
simple logistic-regression model to rank churn risk, and saves the charts that
tell the story. Run it, then read README.md for what I found.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(11)
OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def make_data(n=4000):
    tenure = rng.integers(1, 73, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"],
                          n, p=[0.5, 0.3, 0.2])
    support_calls = rng.poisson(1.5, n)
    # churn probability: short tenure + month-to-month + high bill + many
    # support calls = the classic at-risk profile
    logit = (-2.2
             - 0.045 * tenure
             + 0.012 * monthly
             + 0.45 * support_calls
             + np.where(contract == "Month-to-month", 1.1, 0)
             - np.where(contract == "Two year", 0.7, 0))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "contract": contract,
        "support_calls": support_calls,
        "churned": churn,
    })


def main():
    df = make_data()
    print(f"dataset: {len(df)} customers, "
          f"churn rate {df.churned.mean():.1%}")

    # --- chart 1: churn rate by contract type
    fig, ax = plt.subplots(figsize=(7, 4.5))
    (df.groupby("contract")["churned"].mean()
       .reindex(["Month-to-month", "One year", "Two year"])
       .plot(kind="bar", ax=ax, color="#2b6cb0"))
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("churn rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_contract.png", dpi=120)
    plt.close(fig)

    # --- chart 2: tenure distribution, churned vs stayed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=df, x="tenure_months", hue="churned", ax=ax,
                fill=True, alpha=0.35, palette={0: "#38a169", 1: "#e53e3e"})
    ax.set_title("Tenure distribution: churned vs stayed")
    ax.set_xlabel("tenure (months)")
    fig.tight_layout()
    fig.savefig(OUT / "tenure_distribution.png", dpi=120)
    plt.close(fig)

    # --- model: who is at risk?
    X = pd.get_dummies(df[["tenure_months", "monthly_charges",
                            "contract", "support_calls"]],
                       columns=["contract"], drop_first=True)
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    print(f"AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, model.predict(X_test_s),
                                target_names=["stayed", "churned"]))

    coef = (pd.Series(model.coef_[0], index=X.columns)
              .sort_values(key=abs, ascending=False))
    print("\nfeature impact (standardized coefficients):")
    print(coef.round(3).to_string())

    # --- chart 3: coefficient impact
    fig, ax = plt.subplots(figsize=(7, 4.5))
    coef.plot(kind="barh", ax=ax, color="#6b46c1")
    ax.set_title("What drives churn? (model coefficients)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "feature_impact.png", dpi=120)
    plt.close(fig)

    print(f"\ncharts saved to {OUT}/")


if __name__ == "__main__":
    main()
