"""
app.py
-------
Interactive Streamlit demo for the House Price Prediction Platform,
running on the real Kaggle Ames Housing model.

Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).parent / "src"))
from predict import HousePricePredictor

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "house_price_model.pkl"

QUALITY_OPTIONS = ["Ex", "Gd", "TA", "Fa", "Po"]  # Excellent, Good, Typical, Fair, Poor
NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill",
    "NWAmes", "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW",
    "Somerst", "StoneBr", "Timber", "Veenker",
]


@st.cache_resource
def load_predictor():
    return HousePricePredictor(MODEL_PATH)


st.title("🏠 House Price Prediction Platform")
st.markdown(
    "Estimate a home's sale price using a gradient boosting model trained "
    "on the real Kaggle **Ames Housing** dataset (1,460 home sales, Ames, Iowa)."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python src/train_model.py` first "
        "to train and save the model."
    )
    st.stop()

predictor = load_predictor()

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Structure")
    gr_liv_area = st.slider("Above-ground living area (sqft)", 400, 5000, 1500, step=50)
    total_bsmt_sf = st.slider("Basement area (sqft)", 0, 3000, 800, step=50)
    lot_area = st.slider("Lot area (sqft)", 1000, 40000, 9000, step=100)
    bedrooms = st.number_input("Bedrooms", 0, 8, 3)
    full_bath = st.number_input("Full bathrooms", 0, 4, 2)
    half_bath = st.number_input("Half bathrooms", 0, 2, 0)
    garage_cars = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
    fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)

with col2:
    st.subheader("Quality & Location")
    overall_qual = st.slider("Overall quality (1-10)", 1, 10, 6)
    overall_cond = st.slider("Overall condition (1-9)", 1, 9, 5)
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=NEIGHBORHOODS.index("NAmes"))
    house_style = st.selectbox("House style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"])
    year_built = st.slider("Year built", 1900, 2010, 1975)
    year_sold = st.slider("Year sold", 2006, 2010, 2008)
    kitchen_qual = st.selectbox("Kitchen quality", QUALITY_OPTIONS, index=2)
    bsmt_qual = st.selectbox("Basement quality (None = no basement)", QUALITY_OPTIONS + ["None"], index=2)

st.divider()

if st.button("Predict Sale Price", type="primary", use_container_width=True):
    house = {
        "LotArea": lot_area,
        "LotFrontage": 70,  # median-ish default; not exposed in the UI for simplicity
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": gr_liv_area,
        "2ndFlrSF": 0,
        "GrLivArea": gr_liv_area,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": bedrooms + 3,
        "Fireplaces": fireplaces,
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 260,
        "WoodDeckSF": 0,
        "OpenPorchSF": 40,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "YrSold": year_sold,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "BldgType": "1Fam",
        "CentralAir": "Y",
        "SaleCondition": "Normal",
        "ExterQual": kitchen_qual,
        "KitchenQual": kitchen_qual,
        "BsmtQual": bsmt_qual,
        "GarageType": "Attchd" if garage_cars > 0 else "None",
        "FireplaceQu": "TA" if fireplaces > 0 else "None",
    }
    predicted_price = predictor.predict_one(house)

    st.success(f"### Estimated Sale Price: ${predicted_price:,.0f}")

    margin = predicted_price * 0.10
    st.caption(
        f"Approximate range: ${predicted_price - margin:,.0f} – ${predicted_price + margin:,.0f} "
        "(based on typical model error; not a formal prediction interval)."
    )

    with st.expander("See input summary"):
        st.json(house)

st.divider()
st.caption(
    "Model: Gradient Boosting Regressor trained on log(SalePrice), scikit-learn + XGBoost. "
    "See the README for full methodology, model comparison, and evaluation metrics."
)
