"""
Production-ready prediction module
Author: Khoshaba Odeesho
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
from pydantic import BaseModel
import os
import logging 

# Define the data structure for the API
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='predictions.log'
)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self, model_path: str = 'models/churn_prediction_model.pkl'):
        """Initialize the predictor with trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            self.feature_names = self._get_feature_names()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Error: Model file not found at {model_path}")

    def _get_feature_names(self) -> List[str]:
        """
        Get the expected feature names from training, excluding identifier columns.
        This list now represents the features of a correctly trained model.
        """
        return [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
            'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 
            'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
            'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
            'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
            'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 
            'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
            'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]

    def preprocess_input(self, customer_data: Dict) -> pd.DataFrame:
        """
        Preprocess a single customer's data to match the format expected by the model.
        This function must replicate all preprocessing steps from the training phase.
        """
        df = pd.DataFrame([customer_data])
        
        # --- NEW CODE: Remove identifier columns before any processing ---
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        # ----------------------------------------------------------------

        # Convert TotalCharges to numeric (it's stored as string) and handle errors
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

        # Binary encoding for 'Yes'/'No' and gender columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        
        # Handling gender column explicitly
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # One-hot encoding for multi-category columns
        multi_category_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                              'OnlineBackup', 'DeviceProtection', 'TechSupport',
                              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        df = pd.get_dummies(df, columns=multi_category_cols, prefix=multi_category_cols, dtype=int)
        
        # Ensure the final DataFrame has the exact same columns as the training data.
        missing_cols = set(self.feature_names) - set(df.columns)
        for c in missing_cols:
            df[c] = 0

        # Select and reorder columns to match the training set
        df = df[self.feature_names]

        return df

    def predict_churn(self, customer_data: Dict) -> Dict:
        """
        Predict customer churn probability
        
        Args:
            customer_data: Dictionary with customer information
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(customer_data)
            
            # Get prediction probability
            churn_probability = self.model.predict_proba(processed_data)[0][1]
            
            # Get binary prediction
            churn_prediction = self.model.predict(processed_data)[0]
            
            # Risk level
            if churn_probability > 0.7:
                risk_level = "HIGH"
            elif churn_probability > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            result = {
                'customer_id': customer_data.get('customer_id', 'unknown'),
                'churn_probability': float(churn_probability),
                'will_churn': bool(churn_prediction),
                'risk_level': risk_level,
                'confidence': float(max(churn_probability, 1-churn_probability)),
                'status': 'success'
            }

            # Log the prediction result
            logger.info(f"Prediction for customer {customer_data.get('customerID', 'unknown')}: {result}")
            
            return result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'status': 'error'
            }
            # Log the error
            logger.error(f"Error for customer {customer_data.get('customerID', 'unknown')}: {error_result}")
            return error_result

# Initialize the FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="An API to predict customer churn risk."
)

# Initialize the predictor once when the app starts
model_path = os.path.join(os.path.dirname(__file__), '../models/churn_prediction_model.pkl')
predictor = ChurnPredictor(model_path=model_path)

@app.post("/predict")
def predict_churn_api(customer_data: CustomerData):
    """
    API endpoint for predicting churn.
    """
    customer_dict = customer_data.dict()
    prediction_result = predictor.predict_churn(customer_dict)
    return prediction_result


# Test the predictor with a realistic sample
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Test with a sample customer dictionary (raw data format)
    sample_customer_raw = {
        'customerID': 'TEST001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 99.85,
        'TotalCharges': 2354.35
    }
    
    result = predictor.predict_churn(sample_customer_raw)
    print("🧪 Test Prediction:", result)
