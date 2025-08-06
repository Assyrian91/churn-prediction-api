"""
Production-ready Churn Prediction API logic
Author: Khoshaba Odeesho
"""

import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Union
from pydantic import BaseModel
from fastapi import FastAPI
import dvc.api 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Load model and encoders
# =========================

try:
    # Load the model using DVC
    with dvc.api.open('models/churn_prediction_model.pkl', mode='rb') as f:
        model = joblib.load(f)
    # Load the scaler using DVC
    with dvc.api.open('models/scaler.pkl', mode='rb') as f:
        scaler = joblib.load(f)
    # Load the encoder using DVC
    with dvc.api.open('models/encoder.pkl', mode='rb') as f:
        encoder = joblib.load(f)
    logger.info("✅ Model and artifacts loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model artifacts.")
    raise e

# =========================
# Input Schema
# =========================

class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Union[float, str]

# =========================
# Preprocessing
# =========================

def preprocess_input(customer_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    numeric_cols = scaler.feature_names_in_.tolist()
    expected_categorical_cols = encoder.feature_names_in_.tolist()

    df_numeric = df[numeric_cols]
    df_categorical = df[expected_categorical_cols]

    df_scaled_numeric = pd.DataFrame(scaler.transform(df_numeric), columns=numeric_cols)
    df_encoded_categorical = pd.DataFrame(
        encoder.transform(df_categorical).toarray(),
        columns=encoder.get_feature_names_out(expected_categorical_cols)
    )

    processed_df = pd.concat([df_scaled_numeric, df_encoded_categorical], axis=1)
    return processed_df

# =========================
# Prediction
# =========================

def predict_churn(customer_data: Dict) -> Dict:
    processed = preprocess_input(customer_data)

    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0]

    churn_risk = round(proba[1] * 100)