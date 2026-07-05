import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["house_age"] = df["YrSold"] - df["YearBuilt"]
    df["remod_age"] = df["YrSold"] - df["YearRemodAdd"]

    df["house_age"] = df["house_age"].clip(lower=0)
    df["remod_age"] = df["remod_age"].clip(lower=0)


    df["total_sqft"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    df["total_baths"] = df["FullBath"] + 0.5 * df["HalfBath"]

    df["has_garage"] = (df["GarageCars"] > 0).astype(int)
    df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["was_remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

  
    df = df.drop(columns=["YearBuilt", "YearRemodAdd", "YrSold"])

    return df


ENGINEERED_NUMERIC_FEATURES = [
    "house_age", "remod_age", "total_sqft", "total_baths",
    "has_garage", "has_fireplace", "was_remodeled",
]
