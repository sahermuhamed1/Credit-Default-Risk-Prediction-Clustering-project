import pandas as pd
import joblib
import os
from src.data_cleaning import clean_column_names, select_relevant_columns
from src.feature_engineering import create_financial_ratios, create_delay_features
from src.RFM import calculate_rfm

MODELS_DIR = 'models'

# --- REMOVED HARDCODED FEATURE COLUMNS FROM HERE ---
# Instead, we will load them from 'feature_names.pkl' as intended.

def preprocess_for_prediction(data_dict, feature_columns):
    """
    Applies the same preprocessing and feature engineering steps
    to a single input data point (as a dictionary) for prediction.

    Args:
        data_dict (dict): A dictionary of raw input features.
        feature_columns (list): A list of feature names in the exact order
                                required by the model.

    Returns:
        pd.DataFrame: A single row DataFrame with processed features,
                      ready for scaling and prediction.
    """
    # Create a DataFrame from the input dictionary.
    # It must contain the raw input columns that feature engineering functions expect.
    input_df = pd.DataFrame([data_dict])

    # Apply feature engineering steps in the same order as training
    input_df = create_financial_ratios(input_df)
    input_df = create_delay_features(input_df)
    input_df = calculate_rfm(input_df)

    # Now, ensure the DataFrame has ONLY the columns the model was trained on
    # and in the exact order.
    # The 'id' column *is* expected by the model now, as per your latest error.
    # 'default_payment' is the target and should NOT be in feature_columns.

    # Reindex to ensure all columns are present (NaN if missing) and in the correct order.
    final_features_df = input_df.reindex(columns=feature_columns)

    # Handle any potential NaN values (e.g., from new features, or missing raw inputs)
    final_features_df = final_features_df.fillna(0) # Fill with 0 as a robust default

    return final_features_df


def load_pipeline_assets():
    """
    Loads the trained scaler, model, and feature names list for the prediction pipeline.
    """
    scaler_path = os.path.join(MODELS_DIR, 'robust_scaler.pkl')
    model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl') # Path to saved feature names

    model = None
    scaler = None
    feature_columns = [] # Initialize as empty list

    # Try loading scaler
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at: {scaler_path}")
    else:
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")

    # Try loading model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at: {model_path}")
    else:
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    # Try loading feature names
    if not os.path.exists(feature_names_path):
        print(f"Error: Feature names file NOT FOUND at: {feature_names_path}")
    else:
        try:
            loaded_features = joblib.load(feature_names_path)
            if not loaded_features: # Check if the loaded list is empty
                print(f"Warning: feature_names.pkl loaded but the list is empty at: {feature_names_path}")
            else:
                feature_columns = loaded_features # Assign if loaded successfully and not empty
                print(f"Feature names loaded from {feature_names_path}. Number of features: {len(feature_columns)}")
        except Exception as e:
            print(f"Error loading feature names from {feature_names_path}: {e}")

    # Final check: all three assets must be loaded and feature_columns must not be empty
    if model is None or scaler is None or not feature_columns:
        print("Pipeline initialization failed: Model, Scaler, or Feature Columns are missing after initialization.")
        print("Please ensure 'main.py' has been run successfully to create these assets in the 'models/' directory.")
        return None, None, [] # Return empty list for features if any asset failed
    else:
        return model, scaler, feature_columns


# Global variables to store loaded assets and feature names
global_model = None
global_scaler = None
global_feature_columns = [] # This will be populated by initialize_pipeline()


def initialize_pipeline():
    """Initializes the model, scaler, and feature list for use in the Flask app."""
    global global_model, global_scaler, global_feature_columns
    
    # Attempt to load all necessary assets
    # This will now correctly populate global_model, global_scaler, and global_feature_columns
    global_model, global_scaler, loaded_features = load_pipeline_assets() 
    
    # Assign the loaded features to the global variable if successful
    if loaded_features:
        global_feature_columns = loaded_features
    else:
        # This branch means load_pipeline_assets failed to get features
        print("Error: Global feature columns could not be set in initialize_pipeline. Check load_pipeline_assets logs.")
        global_feature_columns = [] # Ensure it's empty if features not loaded

    # Final check after initialization (for logging purposes)
    if not global_model or not global_scaler or not global_feature_columns:
        print("Pipeline initialization failed (final check in initialize_pipeline): Model, Scaler, or Feature Columns are missing after initialization.")


def predict_single_input(data_dict):
    """
    Takes a dictionary of raw input features, processes it, and returns a prediction.
    """
    # Defensive check: Ensure pipeline assets are loaded. If not, try to initialize.
    if global_model is None or global_scaler is None or not global_feature_columns:
        print("Predict function called before pipeline was fully initialized. Attempting re-initialization.")
        initialize_pipeline() # Try to re-initialize if not loaded
        
        # After re-initialization attempt, check again. If still None/empty, return error.
        if global_model is None or global_scaler is None or not global_feature_columns:
            print("Failed to initialize pipeline for prediction after multiple attempts. Cannot proceed with prediction.")
            return None, None 

    try:
        # Preprocess the input data using the shared function and global feature list
        # global_feature_columns is now populated by initialize_pipeline
        processed_data = preprocess_for_prediction(data_dict, global_feature_columns)
        
        # Validate the processed data before passing to scaler/model
        if processed_data.empty or processed_data.shape[1] != len(global_feature_columns):
            print(f"Error: Processed data has {processed_data.shape[1]} columns, expected {len(global_feature_columns)}.")
            print(f"Processed data columns: {processed_data.columns.tolist()}")
            print(f"Expected feature columns: {global_feature_columns[:5]}... (showing first 5)") 
            # This indicates an issue with feature engineering or the input data not allowing all features to be created.
            return None, None

        # Scale the processed data using the loaded scaler
        scaled_data = global_scaler.transform(processed_data)
        
        # Make prediction using the loaded model
        prediction_proba = global_model.predict_proba(scaled_data)[:, 1]
        prediction_class = (prediction_proba >= 0.5).astype(int)[0] # Using 0.5 as default threshold

        return prediction_class, prediction_proba[0]

    except Exception as e:
        # Catch any other unexpected errors during prediction process
        print(f"Prediction error in predict_single_input (catch-all): {e}")
        return None, None

# Example usage for testing this module in isolation (will not run when imported by Flask)
if __name__ == '__main__':
    print("Running pipeline.py in isolation for testing...")
    
    initialize_pipeline()

    if global_model and global_scaler and global_feature_columns:
        print("\nPipeline initialized for local test.")
        # Example dummy input - This must mimic the structure of your Flask form data
        # with original (renamed) column names, and including 'id' now.
        dummy_input = {
            'id': 1, 'limit_bal': 120000, 'sex': 2, 'education': 2, 'marriage': 1, 'age': 24,
            'status_sep': 2, 'status_aug': 2, 'status_jul': -1, 'status_june': -1, 'status_may': -2, 'status_apr': -2,
            'debt_sep': 3913, 'debt_aug': 3102, 'debt_jul': 689, 'debt_june': 0, 'debt_may': 0, 'debt_apr': 0,
            'pay_sep': 0, 'pay_aug': 689, 'pay_jul': 0, 'pay_june': 0, 'pay_may': 0, 'pay_apr': 0
        }
        
        pred_class, pred_proba = predict_single_input(dummy_input)
        if pred_class is not None:
            print(f"\nLocal Test Prediction: Class {pred_class}, Probability {pred_proba:.4f}")
        else:
            print("\nLocal Test Prediction failed.")
    else:
        print("\nPipeline failed to initialize for local test. Check logs above.")