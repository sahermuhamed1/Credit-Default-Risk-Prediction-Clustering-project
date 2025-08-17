import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def prepare_data_for_model(df):
    """
    Separates features and target, performs train-test split,
    SMOTE oversampling, and robust scaling.
    Returns X_train_scaled, X_test_scaled, y_train, y_test, the fitted scaler,
    and the list of feature names used for training.
    """
    # Based on your latest error, 'id' IS being used as a feature during training
    # Only drop 'default_payment' (the target)
    X = df.drop(columns=["default_payment"]) # <-- MODIFIED: ONLY DROP 'default_payment'
    y = df["default_payment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Original training target distribution:", pd.Series(y_train).value_counts())

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Resampled training target distribution:", pd.Series(y_train_resampled).value_counts())

    scaler = RobustScaler()
    # Fit scaler on resampled training data
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    # Transform test data
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'robust_scaler.pkl'))
    print(f"Scaler saved to {os.path.join(MODELS_DIR, 'robust_scaler.pkl')}")

    # Get the feature names AFTER dropping target, BEFORE splitting/resampling
    feature_names = X.columns.tolist() 
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    print(f"Feature names saved to {os.path.join(MODELS_DIR, 'feature_names.pkl')}")

    # Return as DataFrames to preserve column names (important for sklearn's feature name checks)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_scaled_df, X_test_scaled_df, y_train_resampled, y_test, scaler, feature_names


def train_model(X_train, y_train):
    """
    Trains the XGBoost classifier and saves the model.
    """
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    print(f"XGBoost model saved to {os.path.join(MODELS_DIR, 'xgboost_model.pkl')}")

    return xgb_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints performance metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:\n", pd.DataFrame(pd.crosstab(y_test, y_pred)))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

def load_prediction_assets():
    """
    Loads the trained model and scaler for prediction.
    """
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'robust_scaler.pkl'))
        print("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print(f"Error: Model or scaler not found in {MODELS_DIR}. Please run main.py to train the model first.")
        return None, None