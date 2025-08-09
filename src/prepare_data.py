import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dvc.api import DVCFileSystem
import joblib
import os

def load_data():
    fs = DVCFileSystem()
    path = fs.open('data/raw/Telco_Customer_Churn.csv', mode='rb')
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df['Churn'] = y.reset_index(drop=True)

    return processed_df, preprocessor

def save_artifacts(processed_df, preprocessor):
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print("Data prepared successfully!")
    print(f"Preprocessor saved to: models/preprocessor.pkl")
    print(f"Train data saved to: data/processed/train.csv")
    print(f"Test data saved to: data/processed/test.csv")

def main():
    df = load_data()
    processed_df, preprocessor = preprocess_data(df)
    save_artifacts(processed_df, preprocessor)

if __name__ == "__main__":
    main()