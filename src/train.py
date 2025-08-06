# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import dvc.api  # استيراد DVC لتحميل البيانات

# Load the dataset using DVC
with dvc.api.open('data/Telco_Customer_Churn.csv') as f:
    df = pd.read_csv(f)

# Drop irrelevant columns
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fix Pandas warning: Use assignment instead of inplace
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Define feature and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Define numeric and categorical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
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

# Apply preprocessing and get the preprocessed data
X_preprocessed = preprocessor.fit_transform(X)

# Extract and save the scaler and encoder
scaler = preprocessor.named_transformers_['num']
encoder = preprocessor.named_transformers_['cat']

# Save the scaler and encoder to the 'models' directory
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved successfully.")

joblib.dump(encoder, 'models/encoder.pkl')
print("✅ Encoder saved successfully.")

# Now, let's train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_preprocessed, y)

# Save the trained model
joblib.dump(model, 'models/churn_prediction_model.pkl')
print("✅ Model saved successfully.")