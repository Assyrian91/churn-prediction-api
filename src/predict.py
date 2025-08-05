"""
Production-ready Churn Prediction API logic
Author: Khoshaba Odeesho
"""

import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Union, List
from pydantic import BaseModel
from fastapi import FastAPI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Load model and encoders
# =========================

# Global variables for the model artifacts
model: any
scaler: any
encoder: any

try:
    model = joblib.load("models/churn_prediction_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    logger.info("✅ Model and artifacts loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model artifacts. Please ensure 'models/churn_prediction_model.pkl', 'models/scaler.pkl', and 'models/encoder.pkl' exist.")
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
    This logic must perfectly match the training preprocessing.
    """
    df = pd.DataFrame([customer_data])
    
    # Ensure numeric conversion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Get the expected numeric and categorical columns from the encoder and scaler
    numeric_cols = scaler.feature_names_in_.tolist()
    expected_categorical_cols = encoder.feature_names_in_.tolist()

    # Filter only the columns seen during training
    df_numeric = df[numeric_cols]
    df_categorical = df[expected_categorical_cols]

    # Transform
    df_scaled_numeric = pd.DataFrame(scaler.transform(df_numeric), columns=numeric_cols)
    df_encoded_categorical = pd.DataFrame(
        encoder.transform(df_categorical).toarray(),
        columns=encoder.get_feature_names_out(expected_categorical_cols)
    )

    # Final processed dataframe
    processed_df = pd.concat([df_scaled_numeric, df_encoded_categorical], axis=1)

    return processed_df
def predict_churn(customer_data: Dict) -> str:
    """
    Make a churn prediction for the given customer data.
    """
    processed = preprocess_input(customer_data)
    print("Processed columns:", processed.columns.tolist())
    # print("Expected by model:", model.feature_names_in_.tolist())

    prediction = model.predict(processed)[0]  
    return "Churn" if prediction == 1 else "No Churn"
# =========================
# FastAPI Integration
# =========================

app = FastAPI(
    title="Churn Prediction API",
    description="An API to predict customer churn risk."
)

@app.post("/predict")
def predict_churn_api(customer_data: CustomerData):
    """
    API endpoint for predicting churn.
    """
    customer_dict = customer_data.model_dump()
    print("Received data:", customer_dict)
    result = predict_churn(customer_dict)
    return {"prediction": result}