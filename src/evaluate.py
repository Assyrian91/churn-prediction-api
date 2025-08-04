import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load the trained model
try:
    model = joblib.load('../models/churn_prediction_model.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found. Please check the path.")

# 2. Load the data and preprocess it
try:
    df = pd.read_csv('../data/telco_customer_churn.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Data file not found. Please check the path.")
    exit()

if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

multi_category_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=multi_category_cols, prefix=multi_category_cols, dtype=int)

# 3. Prepare data for evaluation
target = df['Churn'].map({'Yes': 1, 'No': 0})
features = df.drop('Churn', axis=1)

# 4. Make predictions and evaluate the model
predictions = model.predict(features)

accuracy = accuracy_score(target, predictions)
precision = precision_score(target, predictions)
recall = recall_score(target, predictions)

print("\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("---------------------------------")
