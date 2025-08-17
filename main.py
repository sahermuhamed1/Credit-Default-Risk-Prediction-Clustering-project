import pandas as pd
from src.data_cleaning import preprocess_data
from src.feature_engineering import engineer_features
from src.RFM import add_rfm_features
from src.model import prepare_data_for_model, train_model, evaluate_model

def run_training_pipeline():
    """
    Executes the full data preprocessing, feature engineering, RFM analysis,
    and model training/evaluation pipeline.
    """
    print("--- Starting Data Processing and Model Training Pipeline ---")

    # 1. Data Cleaning
    print("\n1. Running Data Cleaning...")
    df = preprocess_data(filepath='data/UCI_Credit_Card.csv')
    print(f"Initial data shape after cleaning: {df.shape}")
    print(f"Columns after cleaning: {df.columns.tolist()}")

    # 2. Feature Engineering
    print("\n2. Running Feature Engineering...")
    df = engineer_features(df)
    print(f"Data shape after feature engineering: {df.shape}")
    print(f"New columns: {[col for col in df.columns if 'utilization' in col or 'pay_ratio' in col or 'delay' in col]}")

    # 3. RFM Analysis
    print("\n3. Running RFM Analysis...")
    df = add_rfm_features(df)
    print(f"Data shape after RFM analysis: {df.shape}")
    print(f"RFM columns: {[col for col in df.columns if col in ['R', 'F', 'M']]}")

    # 4. Prepare Data for Model Training
    print("\n4. Preparing Data for Model Training...")
    # X_train_scaled and X_test_scaled are DataFrames here
    # `feature_names` returned here will contain 'id' if `model.py` is configured to keep it.
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = prepare_data_for_model(df.copy())
    
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    # This print is critical to see the exact features used for training
    print(f"Features used for training/prediction: {feature_names}")


    # 5. Model Training
    print("\n5. Training Model (XGBoost)...")
    model = train_model(X_train_scaled, y_train)
    print("Model training complete.")

    # 6. Model Evaluation
    print("\n6. Evaluating Model...")
    evaluate_model(model, X_test_scaled, y_test)
    print("Model evaluation complete.")

    print("\n--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    run_training_pipeline()