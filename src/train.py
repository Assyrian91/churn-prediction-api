import pandas as pd
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Load data
# =========================

def load_data():
    """Loads preprocessed data."""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, test_df

# =========================
# Train model
# =========================

def train_model(train_df):
    """Trains a Logistic Regression model."""
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']

    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    return model

# =========================
# Evaluate model
# =========================

def evaluate_model(model, test_df):
    """Evaluates the model on test data."""
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    return metrics

# =========================
# Save artifacts and metrics
# =========================

def save_artifacts(model, metrics):
    """Saves the trained model and metrics."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    joblib.dump(model, 'models/churn_prediction_model.pkl')

    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    """Main function to run the training pipeline."""
    train_df, test_df = load_data()
    model = train_model(train_df)
    metrics = evaluate_model(model, test_df)
    save_artifacts(model, metrics)
    print("Model trained and evaluated successfully!")

if __name__ == "__main__":
    main()