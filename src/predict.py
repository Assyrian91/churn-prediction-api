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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Load model and encoders
# =========================

try:
    model = joblib.load("models/churn_prediction_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    logger.info("✅ Model and artifacts loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model artifacts.")
    raise e

# =========================
# Input Data Schema
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
# Prediction Logic
# =========================

def preprocess_input(customer_data: Dict) -> pd.DataFrame:
    """
    Preprocess customer data: encode, scale, and prepare for prediction.
    """
    df = pd.DataFrame([customer_data])

    # Handle TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Define numeric and categorical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # 'SeniorCitizen' is a numeric feature but should not be scaled
    categorical_cols = df.drop(columns=['customerID'] + numeric_cols + ['SeniorCitizen']).columns.tolist()

    # Separate numeric and categorical data
    df_numeric = df[numeric_cols]
    df_categorical = df[categorical_cols]

    # Apply scaling to numeric data
    df_scaled_numeric = pd.DataFrame(scaler.transform(df_numeric), columns=numeric_cols)

    # Apply encoding to categorical data
    df_encoded_categorical = pd.DataFrame(
        encoder.transform(df_categorical).toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    
    # Add back the SeniorCitizen column which is already a number
    df_senior = df[['SeniorCitizen']].reset_index(drop=True)

    # Concatenate the processed features
    processed_df = pd.concat([df_scaled_numeric, df_encoded_categorical, df_senior], axis=1)
    
    # Reorder columns to match the training data
    # (This is important for the model to work correctly)
    all_features = df_scaled_numeric.columns.tolist() + df_encoded_categorical.columns.tolist() + df_senior.columns.tolist()
    processed_df = processed_df[all_features]

    return processed_df
# =========================
# FastAPI Integration
# =========================

# Only import FastAPI if this module is run directly or through app
try:
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/predict")
    def predict_churn_api(customer_data: CustomerData):
        """
        API endpoint for predicting churn.
        """
        customer_dict = customer_data.model_dump()  # Compatible with Pydantic v2
        result = predict_churn(customer_dict)
        return {"prediction": result}

except ImportError:
    logger.warning("FastAPI not installed – skipping app setup.")
