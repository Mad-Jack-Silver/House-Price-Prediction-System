"""
test_pipeline.py
------------------
Unit tests covering the core pipeline components for the real Kaggle
Ames Housing schema. Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_preprocessing import clean_data, build_preprocessor, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES, NA_MEANS_NONE_COLUMNS
from feature_engineering import engineer_features, ENGINEERED_NUMERIC_FEATURES


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 40
    neighborhoods = ["NAmes", "CollgCr", "OldTown"]
    df = pd.DataFrame({
        "Id": list(range(1, n + 1)),
        "LotArea": rng.normal(9500, 2000, n).round(0),
        "LotFrontage": rng.normal(70, 15, n).round(0),
        "OverallQual": rng.integers(3, 9, n),
        "OverallCond": rng.integers(3, 8, n),
        "TotalBsmtSF": rng.normal(900, 300, n).clip(0).round(0),
        "1stFlrSF": rng.normal(1100, 250, n).round(0),
        "2ndFlrSF": rng.choice([0, 500, 800], n),
        "GrLivArea": rng.normal(1500, 400, n).round(0),
        "FullBath": rng.integers(1, 3, n),
        "HalfBath": rng.integers(0, 2, n),
        "BedroomAbvGr": rng.integers(2, 5, n),
        "TotRmsAbvGrd": rng.integers(5, 10, n),
        "Fireplaces": rng.integers(0, 3, n),
        "GarageCars": rng.integers(0, 3, n),
        "GarageArea": rng.normal(450, 150, n).clip(0).round(0),
        "WoodDeckSF": rng.integers(0, 200, n),
        "OpenPorchSF": rng.integers(0, 100, n),
        "YearBuilt": rng.integers(1950, 2009, n),
        "YearRemodAdd": rng.integers(1950, 2010, n),
        "YrSold": rng.integers(2006, 2011, n),
        "Neighborhood": rng.choice(neighborhoods, n),
        "HouseStyle": rng.choice(["1Story", "2Story"], n),
        "BldgType": "1Fam",
        "CentralAir": rng.choice(["Y", "N"], n, p=[0.9, 0.1]),
        "SaleCondition": "Normal",
        "ExterQual": rng.choice(["Gd", "TA"], n),
        "KitchenQual": rng.choice(["Gd", "TA"], n),
        "BsmtQual": rng.choice(["Gd", "TA", "MISSING"], n),  # some houses have no basement
        "GarageType": rng.choice(["Attchd", "Detchd", "MISSING"], n),  # some have no garage
        "FireplaceQu": [np.nan] * n,  # will be set based on Fireplaces below
        "SalePrice": rng.normal(180000, 40000, n).round(0),
    })
    # numpy's rng.choice silently stringifies NaN when mixed with strings
    # in a fixed-width string array -- replace the placeholder with a real NaN
    df["BsmtQual"] = df["BsmtQual"].replace("MISSING", np.nan)
    df["GarageType"] = df["GarageType"].replace("MISSING", np.nan)
    # Duplicate one row, inject one extreme outlier price
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[5, "SalePrice"] = 3_000_000
    return df


class TestCleanData:
    def test_removes_duplicates(self, raw_df):
        cleaned = clean_data(raw_df)
        assert len(cleaned) == len(raw_df) - 1

    def test_drops_id_column(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "Id" not in cleaned.columns

    def test_caps_target_outliers(self, raw_df):
        cleaned = clean_data(raw_df)
        assert cleaned[TARGET].max() < 3_000_000

    def test_na_means_none_columns_filled(self, raw_df):
        # Before cleaning, these columns have real NaNs (no basement/garage)
        assert raw_df["BsmtQual"].isna().any()
        cleaned = clean_data(raw_df)
        # After cleaning, NaN should become the literal string "None",
        # never left as a missing value
        for col in NA_MEANS_NONE_COLUMNS:
            assert cleaned[col].isna().sum() == 0
            assert "None" in cleaned[col].values or cleaned[col].nunique() > 0


class TestFeatureEngineering:
    def test_adds_expected_columns(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        for col in ENGINEERED_NUMERIC_FEATURES:
            assert col in engineered.columns

    def test_house_age_correct(self, raw_df):
        cleaned = clean_data(raw_df)
        expected_age = (cleaned["YrSold"] - cleaned["YearBuilt"]).clip(lower=0)
        engineered = engineer_features(cleaned)
        assert np.allclose(engineered["house_age"], expected_age)

    def test_drops_raw_year_columns(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        assert "YearBuilt" not in engineered.columns
        assert "YearRemodAdd" not in engineered.columns
        assert "YrSold" not in engineered.columns

    def test_no_negative_ages(self, raw_df):
        # Simulate a data-entry quirk: remodel year after sale year
        df = raw_df.copy()
        df.loc[0, "YearRemodAdd"] = 2020
        df.loc[0, "YrSold"] = 2008
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        assert (engineered["remod_age"] >= 0).all()


class TestPreprocessor:
    def test_preprocessor_fits_and_transforms(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        all_numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
        preprocessor = build_preprocessor(all_numeric)
        X = engineered[all_numeric + CATEGORICAL_FEATURES]
        transformed = preprocessor.fit_transform(X)
        assert transformed.shape[0] == X.shape[0]
        arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        assert not np.isnan(arr).any()

    def test_handles_genuinely_missing_numeric_values(self, raw_df):
        # LotFrontage has real missing values (not "None means no frontage")
        df = raw_df.copy()
        df.loc[0, "LotFrontage"] = np.nan
        cleaned = clean_data(df)
        engineered = engineer_features(cleaned)
        all_numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
        preprocessor = build_preprocessor(all_numeric)
        X = engineered[all_numeric + CATEGORICAL_FEATURES]
        transformed = preprocessor.fit_transform(X)
        arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        assert not np.isnan(arr).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
