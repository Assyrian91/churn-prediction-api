import pandas as pd
import pickle
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config_manager import ConfigManager
from datetime import datetime

def train_model():
    mlflow.set_tracking_uri("mlruns")
    config = ConfigManager()
    
    # MLflow
    experiment_name = config.get('training.experiment_name')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        
        mlflow.log_param("training_date", datetime.now().isoformat())
        
        train_path = config.get('paths.processed_train')
        train_df = pd.read_csv(train_path)
        
        target_col = config.get('data.target_column')
        X = train_df.drop(target_col, axis=1)
        y = train_df[target_col]
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model_params = config.get('model.parameters')
        model = RandomForestClassifier(**model_params)
        model.fit(X_scaled, y)
        
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        y_pred = model.predict(X_scaled)
        
        train_accuracy = accuracy_score(y, y_pred)
        train_precision = precision_score(y, y_pred, pos_label='Yes')
        train_recall = recall_score(y, y_pred, pos_label='Yes')
        train_f1 = f1_score(y, y_pred, pos_label='Yes')
        
        # regis metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        
        models_dir = config.get('paths.models_dir')
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f'{models_dir}/churn_prediction_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(f'{models_dir}/encoder.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        with open(f'{models_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(f'{models_dir}/encoder.pkl')
        mlflow.log_artifact(f'{models_dir}/scaler.pkl')
        
        metrics_dir = config.get('paths.metrics_dir')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1': float(train_f1),
            'training_date': datetime.now().isoformat()
        }
        
        with open(f'{metrics_dir}/train_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Model trained successfully!")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training F1-Score: {train_f1:.4f}")
        
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    run_id = train_model()
    print(f"Run ID: {run_id}")