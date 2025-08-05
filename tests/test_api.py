import pytest
from fastapi.testclient import TestClient
from src.predict import app  # Assuming your FastAPI app is in src/predict.py

client = TestClient(app)

def test_predict_endpoint_success():
    """Test the /predict endpoint with a valid payload."""
    sample_input = {
        "customerID": "12345",  
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 1900.0
    }
    response = client.post("/predict", json=sample_input)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the response body contains a "churn_prediction" key
    assert "will_churn" in response.json()
    
    # Assert that the prediction is either 0 or 1
    prediction = response.json()["will_churn"]
    assert prediction in [0, 1]

def test_predict_endpoint_invalid_input():
    """Test the /predict endpoint with an invalid payload."""
    invalid_input = {
        "gender": "Male",
        "tenure": "not_a_number", # Invalid data type
        "Contract": "Month-to-month"
    }
    response = client.post("/predict", json=invalid_input)
    
    # Assert that the request was unsuccessful due to invalid input
    assert response.status_code == 422
