"""
evaluate.py
------------
Generates the diagnostic plots that go in the README / portfolio
write-up:
  1. Model comparison bar chart (CV RMSE across candidates)
  2. Predicted vs Actual scatter
  3. Residual plot
  4. Feature importance (from the tuned tree model, or coefficient
     magnitude if the winner is linear)
  5. Correlation heatmap of engineered features

"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).parent))
from data_preprocessing import load_raw_data, clean_data, get_feature_names, TARGET
from feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "models" / "training_results.json"
VISUALS_DIR = PROJECT_ROOT / "visuals"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def plot_model_comparison(results: dict):
    df = pd.DataFrame(results["cv_comparison"]).sort_values("cv_rmse_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["model"], df["cv_rmse_mean"], xerr=df["cv_rmse_std"],
                    color=sns.color_palette("viridis", len(df)), capsize=4)
    ax.set_xlabel("Cross-Validated RMSE ($, lower is better)")
    ax.set_title("Model Comparison — 5-Fold Cross-Validation")
    ax.invert_yaxis()
    for bar, val in zip(bars, df["cv_rmse_mean"]):
        ax.text(val + 500, bar.get_y() + bar.get_height() / 2, f"${val:,.0f}",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "model_comparison.png")
    plt.close()


def plot_pred_vs_actual(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#3b6fb6", edgecolor="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title("Predicted vs Actual House Prices (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "predicted_vs_actual.png")
    plt.close()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color="#e0724a", edgecolor="none")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Sale Price ($)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#e0724a")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "residuals.png")
    plt.close()


def plot_feature_importance(pipeline):
    wrapped_model = pipeline.named_steps["model"]
    model = wrapped_model.regressor_
    preprocessor = pipeline.named_steps["preprocessor"]

    from train_model import ALL_NUMERIC_FEATURES
    feature_names = get_feature_names(preprocessor, numeric_features=ALL_NUMERIC_FEATURES)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importance ({type(model).__name__})"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        title = f"Feature Importance — |Coefficient| ({type(model).__name__})"
    else:
        print("Model type has no interpretable importances; skipping plot.")
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color=sns.color_palette("mako", len(imp_df)))
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "correlation_heatmap.png")
    plt.close()


def main():
    VISUALS_DIR.mkdir(exist_ok=True)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    pipeline = joblib.load(MODEL_PATH)

    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    from sklearn.model_selection import train_test_split
    from train_model import ALL_NUMERIC_FEATURES, RANDOM_SEED
    from data_preprocessing import CATEGORICAL_FEATURES
    X = df[ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    y_pred = pipeline.predict(X_test)

    print("Generating plots...")
    plot_model_comparison(results)
    print("  model_comparison.png")
    plot_pred_vs_actual(y_test, y_pred)
    print("  predicted_vs_actual.png")
    plot_residuals(y_test, y_pred)
    print("  residuals.png")
    plot_feature_importance(pipeline)
    print("  feature_importance.png")
    plot_correlation_heatmap(df)
    print("  correlation_heatmap.png")
    print(f"\nAll plots saved to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
