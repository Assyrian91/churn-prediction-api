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
import dvc.api  # أضفت هذا لدعم تحميل الملفات من DVC

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

    churn_risk = round(proba[1] * 100, 2)
    confidence = round(max(proba) * 100, 2)
    prediction_label = "Churn" if prediction == 1 else "No Churn"
    will_churn = bool(prediction == 1)

    # Risk level for user-friendly labeling
    if churn_risk >= 75:
        risk_level = "High"
    elif churn_risk >= 50:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": prediction_label,
        "will_churn": will_churn,
        "churn_probability": round(proba[1], 4),
        "confidence": confidence,
        "risk_level": risk_level
    }

# =========================
# FastAPI Integration
# =========================

app = FastAPI(
    title="Churn Prediction API",
    description="An API to predict customer churn risk."
)

@app.post("/predict")
def predict_churn_api(customer_data: CustomerData):
    customer_dict = customer_data.model_dump()
    result = predict_churn(customer_dict)
    return result