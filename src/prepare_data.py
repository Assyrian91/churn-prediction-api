import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config_manager import ConfigManager
import os
import joblib

def prepare_data():
    """Load raw data, split, preprocess, and save preprocessor and data."""
    # Load configuration
    config = ConfigManager()

    # Define file paths from config
    raw_data_path = config.get('paths.raw_data')
    processed_train_path = config.get('paths.processed_train')
    processed_test_path = config.get('paths.processed_test')
    models_dir = config.get('paths.models_dir')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Read the raw data file
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}. Please make sure it exists.")
        return

    # Handle 'TotalCharges' column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    # Handle 'SeniorCitizen' column
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Drop 'customerID'
    df = df.drop('customerID', axis=1)

    # Define features and target
    target_column = config.get('data.target_column')
    features = df.drop(columns=target_column).columns.tolist()

    # Split data into train and test sets
    test_size = config.get('data.test_size')
    random_state = config.get('data.random_state')
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    # Define preprocessing steps
    categorical_features = train_df.select_dtypes(include=['object']).drop(columns=[target_column], errors='ignore').columns.tolist()
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit the preprocessor on the training data
    preprocessor.fit(train_df[features])

    # Save the fitted preprocessor
    joblib.dump(preprocessor, preprocessor_path)

    # Save the processed data
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)

    print(f"Data prepared successfully!")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Train data saved to: {processed_train_path}")
    print(f"Test data saved to: {processed_test_path}")

if __name__ == "__main__":
    prepare_data()