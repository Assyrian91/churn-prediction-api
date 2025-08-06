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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Load model and preprocessor
# =========================

try:
    model = joblib.load("models/churn_prediction_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
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
    
    # Drop columns not used for prediction
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Apply the preprocessor to the entire dataframe
    processed_data = preprocessor.transform(df)

    # The preprocessor output is a numpy array, we convert it back to a dataframe
    # with the correct feature names.
    processed_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())

    return processed_df

# =========================
# Prediction
# =========================

def predict_churn(customer_data: Dict) -> Dict:
    processed = preprocess_input(customer_data)

    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0]

    churn_risk = float(round(proba[1] * 100, 2))
    confidence = float(round(max(proba) * 100, 2))
    prediction_label = "Churn" if int(prediction) == 1 else "No Churn"
    will_churn = bool(prediction == 1)

    if churn_risk >= 70:
        risk_level = "High"
    elif churn_risk >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": str(prediction_label),
        "churn_risk_percent": churn_risk,
        "confidence_percent": confidence,
        "will_churn": will_churn,
        "risk_level": str(risk_level)
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