import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model and data
def load_model_and_data():
    """Loads the trained model and test data for evaluation."""
    try:
        logging.info("Attempting to load model and test data from local file system...")
        
        # Load the trained model directly from the file system
        model = joblib.load('models/churn_prediction_model.pkl')
        
        # Load the test data directly from the file system
        test_df = pd.read_csv('data/Telco_customer_Churn.csv')
        
        X_test = test_df.drop('Churn', axis=1)
        y_test = test_df['Churn']
        
        logging.info("✅ Model and test data loaded successfully.")
        return model, X_test, y_test
    except FileNotFoundError as e:
        logging.error(f"❌ File not found error: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ An error occurred during loading: {e}")
        raise e

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model's performance and logs metrics."""
    try:
        logging.info("Evaluating model performance...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"✨ Model Accuracy: {accuracy:.4f}")
        
        # Optional: Save metrics to a file if needed for other processes
        with open("metrics.json", "w") as f:
            f.write(f'{{"accuracy": {accuracy}}}')
            
        logging.info("✅ Model evaluation complete.")
    except Exception as e:
        logging.error(f"❌ An error occurred during evaluation: {e}")
        raise e

if __name__ == "__main__":
    if not os.path.exists('models/churn_prediction_model.pkl'):
        logging.error("❌ The model file 'models/churn_prediction_model.pkl' was not found.")
        logging.info("Hint: The CI/CD pipeline does not run the training script. You must commit a pre-trained model file.")
    if not os.path.exists('data/Telco_customer_Churn_test.csv'):
        logging.error("❌ The data file 'data/Telco_customer_Churn_test.csv' was not found.")
        logging.info("Hint: The CI/CD pipeline does not pull the data. You must commit the data file.")
    
    model, X_test, y_test = load_model_and_data()
    evaluate_model(model, X_test, y_test)