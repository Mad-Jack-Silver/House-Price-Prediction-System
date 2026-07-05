import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import sys
sys.path.append(str(Path(__file__).parent))
from data_preprocessing import load_raw_data, clean_data, build_preprocessor, get_feature_names, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from feature_engineering import engineer_features, ENGINEERED_NUMERIC_FEATURES

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = MODEL_DIR / "training_results.json"

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


def make_model_pipeline(model) -> Pipeline:
    log_model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(ALL_NUMERIC_FEATURES)),
        ("model", log_model),
    ])


def evaluate_candidates(X_train, y_train) -> pd.DataFrame:
    """5-fold CV comparison across several model families."""
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows = []
    for name, model in candidates.items():
        pipe = make_model_pipeline(model)
        t0 = time.time()
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        rows.append({
            "model": name,
            "cv_rmse_mean": -scores["test_rmse"].mean(),
            "cv_rmse_std": scores["test_rmse"].std(),
            "cv_mae_mean": -scores["test_mae"].mean(),
            "cv_r2_mean": scores["test_r2"].mean(),
            "fit_time_sec": round(elapsed, 2),
        })
        print(f"  {name:<18} CV RMSE = {-scores['test_rmse'].mean():>10,.0f}  "
              f"CV R2 = {scores['test_r2'].mean():.4f}  ({elapsed:.1f}s)")

    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)


def tune_best_model(best_name: str, X_train, y_train) -> Pipeline:
    print(f"\nHyperparameter tuning: {best_name}")

    if best_name == "XGBoost":
        model = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
        param_dist = {
            "model__regressor__n_estimators": randint(150, 600),
            "model__regressor__max_depth": randint(2, 8),
            "model__regressor__learning_rate": uniform(0.01, 0.25),
            "model__regressor__subsample": uniform(0.6, 0.4),
            "model__regressor__colsample_bytree": uniform(0.6, 0.4),
            "model__regressor__reg_lambda": uniform(0.5, 3.0),
        }
    elif best_name == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=RANDOM_SEED)
        param_dist = {
            "model__regressor__n_estimators": randint(100, 500),
            "model__regressor__max_depth": randint(2, 6),
            "model__regressor__learning_rate": uniform(0.01, 0.25),
            "model__regressor__subsample": uniform(0.6, 0.4),
        }
    elif best_name == "RandomForest":
        model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
        param_dist = {
            "model__regressor__n_estimators": randint(150, 600),
            "model__regressor__max_depth": randint(4, 30),
            "model__regressor__min_samples_split": randint(2, 12),
            "model__regressor__min_samples_leaf": randint(1, 8),
            "model__regressor__max_features": uniform(0.4, 0.6),
        }
    else:  # Ridge / LinearRegression fallback
        model = Ridge(random_state=RANDOM_SEED)
        param_dist = {"model__regressor__alpha": uniform(0.01, 50)}

    pipe = make_model_pipeline(model)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=40, cv=5,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_SEED,
        n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV RMSE: {-search.best_score_:,.0f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP 1: Load & prepare data")
    print("=" * 70)
    df = load_and_prepare(DATA_PATH)
    X = df[ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("\n" + "=" * 70)
    print("STEP 2: 5-fold cross-validation across candidate models")
    print("=" * 70)
    cv_results = evaluate_candidates(X_train, y_train)
    print("\nModel comparison (sorted by CV RMSE, best first):")
    print(cv_results.to_string(index=False))
    best_name = cv_results.iloc[0]["model"]

    print("\n" + "=" * 70)
    print("STEP 3: Hyperparameter tuning of best model")
    print("=" * 70)
    tuned_pipeline, best_params = tune_best_model(best_name, X_train, y_train)

    print("\n" + "=" * 70)
    print("STEP 4: Final evaluation on held-out test set")
    print("=" * 70)
    y_pred = tuned_pipeline.predict(X_test)
    test_metrics = {
        "rmse": rmse(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }
    for k, v in test_metrics.items():
        print(f"  {k.upper():<6}: {v:,.4f}")

    # Baseline comparison: predicting the mean every time
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_rmse = rmse(y_test, baseline_pred)
    improvement = (1 - test_metrics["rmse"] / baseline_rmse) * 100
    print(f"\n  Naive baseline (predict mean) RMSE: {baseline_rmse:,.0f}")
    print(f"  Model improves on baseline by: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("STEP 5: Persist model + results")
    print("=" * 70)
    model_path = MODEL_DIR / "house_price_model.pkl"
    joblib.dump(tuned_pipeline, model_path)
    print(f"Saved trained pipeline -> {model_path}")

    results = {
        "best_model": best_name,
        "best_params": best_params,
        "cv_comparison": cv_results.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "baseline_rmse": baseline_rmse,
        "improvement_over_baseline_pct": improvement,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features_used": ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved training results -> {RESULTS_PATH}")

    return tuned_pipeline, results, X_test, y_test, y_pred


if __name__ == "__main__":
    main()
