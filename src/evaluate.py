import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config_manager import ConfigManager
from datetime import datetime

def evaluate_model():
    """Evaluate the trained model and save metrics."""
    # Load configuration
    config = ConfigManager()
    
    # Define file paths from config
    test_data_path = config.get('paths.processed_test')
    models_dir = config.get('paths.models_dir')
    metrics_dir = config.get('paths.metrics_dir')
    
    # Load test data
    try:
        test_df = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}. Please run prepare_data.py first.")
        return

    # Load model artifacts
    try:
        with open(f'{models_dir}/churn_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{models_dir}/encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open(f'{models_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run train.py first.")
        return

    # Separate features and target
    target_col = config.get('data.target_column')
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    # Preprocess data using loaded encoders and scaler
    categorical_columns = X_test.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in encoders:
            # Handle unknown categories that might appear in test data
            for label in np.unique(X_test[col]):
                if label not in encoders[col].classes_:
                    encoders[col].classes_ = np.append(encoders[col].classes_, label)
            X_test[col] = encoders[col].transform(X_test[col])
            
    X_test_scaled = scaler.transform(X_test)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Yes')
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')

    # Save metrics to a JSON file
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'evaluation_date': datetime.now().isoformat()
    }

    metrics_path = f'{metrics_dir}/eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model evaluated successfully!")
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    print(f"Evaluation F1-Score: {f1:.4f}")
    print(f"Evaluation metrics saved to: {metrics_path}")

if __name__ == "__main__":
    evaluate_model()