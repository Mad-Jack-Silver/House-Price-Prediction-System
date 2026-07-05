import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from feature_engineering import engineer_features
from data_preprocessing import NA_MEANS_NONE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"


class HousePricePredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.pipeline = joblib.load(model_path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in NA_MEANS_NONE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        return engineer_features(df)

    def predict_one(self, house: dict) -> float:
        """Predict the sale price for a single house described as a dict
        of raw (pre-feature-engineering) attributes, using the original
        Kaggle column names (e.g. 'GrLivArea', 'OverallQual')."""
        df = pd.DataFrame([house])
        df = self._prepare(df)
        pred = self.pipeline.predict(df)[0]
        return float(pred)

    def predict_batch(self, houses: pd.DataFrame) -> pd.Series:
        """Predict sale prices for a DataFrame of raw house attributes."""
        df = self._prepare(houses)
        preds = self.pipeline.predict(df)
        return pd.Series(preds, index=houses.index, name="predicted_price")


if __name__ == "__main__":
    predictor = HousePricePredictor()
    sample_house = {
        "LotArea": 9600, "LotFrontage": 80, "OverallQual": 6, "OverallCond": 7,
        "TotalBsmtSF": 1000, "1stFlrSF": 1200, "2ndFlrSF": 0, "GrLivArea": 1200,
        "FullBath": 2, "HalfBath": 0, "BedroomAbvGr": 3, "TotRmsAbvGrd": 6,
        "Fireplaces": 1, "GarageCars": 2, "GarageArea": 480,
        "WoodDeckSF": 0, "OpenPorchSF": 40,
        "YearBuilt": 1976, "YearRemodAdd": 1976, "YrSold": 2008,
        "Neighborhood": "NAmes", "HouseStyle": "1Story", "BldgType": "1Fam",
        "CentralAir": "Y", "SaleCondition": "Normal", "ExterQual": "TA",
        "KitchenQual": "TA", "BsmtQual": "TA", "GarageType": "Attchd",
        "FireplaceQu": "TA",
    }
    price = predictor.predict_one(sample_house)
    print(f"Predicted sale price: ${price:,.0f}")
