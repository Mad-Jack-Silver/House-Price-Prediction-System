import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "SalePrice"

NUMERIC_FEATURES = [
    "LotArea", "LotFrontage", "OverallQual", "OverallCond",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea",
    "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF",
]

CATEGORICAL_FEATURES = [
    "Neighborhood", "HouseStyle", "BldgType", "CentralAir",
    "SaleCondition", "ExterQual", "KitchenQual",
    "BsmtQual", "GarageType", "FireplaceQu",
]

NA_MEANS_NONE_COLUMNS = ["BsmtQual", "GarageType", "FireplaceQu"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix the genuine-vs-meaningful NA distinction,
    and cap extreme outliers in the target using the IQR rule."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Id"])
    removed = before - len(df)

    for col in NA_MEANS_NONE_COLUMNS:
        df[col] = df[col].fillna("None")

    drop_cols = [c for c in ["Id"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Cap target outliers using IQR (keeps them in-sample but limits leverage)
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    lower_bound = max(0, q1 - 3 * iqr)
    n_capped = ((df[TARGET] > upper_bound) | (df[TARGET] < lower_bound)).sum()
    df[TARGET] = df[TARGET].clip(lower_bound, upper_bound)

    print(f"[clean_data] removed {removed} duplicate rows, capped {n_capped} target outliers")
    return df


def build_preprocessor(numeric_features: list = None) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_features: list = None) -> list:
    numeric_features = numeric_features or NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numeric_features + cat_names