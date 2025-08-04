# app.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import ChurnPredictor
from typing import Dict, Any
import os

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Churn Prediction API",
    description="A simple API for predicting customer churn.",
    version="1.0.0"
)

# Initialize our predictor class to load the model
# This ensures the model is loaded only once when the app starts
try:
    predictor = ChurnPredictor(model_path="models/churn_prediction_model.pkl")
except FileNotFoundError as e:
    # If the model file is not found, raise a RuntimeError to stop the application
    raise RuntimeError(f"❌ Error: Model could not be loaded. Details: {e}")

# Define the input data model for validation
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
    TotalCharges: float

# Define a welcome endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

# Define our main prediction endpoint
@app.post("/predict", response_model=Dict[str, Any])
def predict_churn(customer_data: CustomerData):
    try:
        # Convert Pydantic object to a dictionary
        data_dict = customer_data.dict()

        # Get prediction from our ChurnPredictor class
        prediction_result = predictor.predict_churn(data_dict)

        if prediction_result.get('status') == 'success':
            return prediction_result
        else:
            raise HTTPException(status_code=500, detail=prediction_result.get('error', 'Prediction failed'))

    except Exception as e:
        # Catch any unexpected errors during prediction
        raise HTTPException(status_code=500, detail=str(e))

# Entry point to run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)