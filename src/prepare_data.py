import pandas as pd
from sklearn.model_selection import train_test_split
from config_manager import ConfigManager
import os

def prepare_data():
    """Load raw data, split into train and test sets, and save them."""
    # Load configuration
    config = ConfigManager()

    # Define file paths from config
    raw_data_path = config.get('paths.raw_data')
    processed_train_path = config.get('paths.processed_train')
    processed_test_path = config.get('paths.processed_test')
    
    # Create processed data directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    
    # Read the raw data file
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}. Please make sure it exists.")
        return

    # Handle 'TotalCharges' column which contains spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Handle 'SeniorCitizen' column, which is an int, convert to object
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Drop 'customerID'
    df = df.drop('customerID', axis=1)

    # Split data into train and test sets
    test_size = config.get('data.test_size')
    random_state = config.get('data.random_state')
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # Save the processed data
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)

    print(f"Data prepared successfully!")
    print(f"Train data saved to: {processed_train_path}")
    print(f"Test data saved to: {processed_test_path}")

if __name__ == "__main__":
    prepare_data()
