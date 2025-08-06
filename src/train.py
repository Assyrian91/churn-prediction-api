# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
import dvc.api

# Set MLflow tracking URI for local storage
# mlflow.set_tracking_uri("file:///path/to/your/mlruns")
mlflow.set_experiment("Churn Prediction")

# Load the dataset using DVC
with dvc.api.open('data/Telco_Customer_Churn.csv') as f:
    df = pd.read_csv(f)

# Data Preprocessing
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Define feature and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Combine preprocessor and model into a single pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Start the MLflow run and train the model inside it
with mlflow.start_run():
    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained with Accuracy: {accuracy:.4f}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    # Save the model and preprocessors to the 'models' directory
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline['preprocessor'], 'models/preprocessor.pkl')
    joblib.dump(model_pipeline['classifier'], 'models/churn_prediction_model.pkl')
    print("✅ Model and Preprocessor saved locally.")

    # Log the entire pipeline as an MLflow artifact
    mlflow.sklearn.log_model(model_pipeline, "churn_model_pipeline")
    print("✅ Model pipeline logged to MLflow.")