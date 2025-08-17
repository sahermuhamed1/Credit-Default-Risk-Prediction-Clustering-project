from flask import Blueprint, render_template, request, jsonify
import pandas as pd
# Import global_model, global_scaler, global_feature_columns directly
from src.pipeline import predict_single_input, initialize_pipeline, global_model, global_scaler, global_feature_columns 

bp = Blueprint('main', __name__)

# Initialize the pipeline when the blueprint is created
# This happens once when the Flask app starts
initialize_pipeline()

# Debug prints for initial load state
print(f"Routes: global_model is {'NOT None' if global_model else 'None'}")
print(f"Routes: global_scaler is {'NOT None' if global_scaler else 'None'}")
print(f"Routes: global_feature_columns has {len(global_feature_columns)} features. (at blueprint init)")


@bp.route('/')
def index():
    """Renders the main page with the prediction form."""
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            
            # These are the *raw* columns from the form,
            # which will be passed to feature engineering.
            # 'id' is here because the model seems to have been trained with it.
            numeric_fields = [
                'id', 'limit_bal', 'sex', 'education', 'marriage', 'age',
                'status_sep', 'status_aug', 'status_jul', 'status_june', 'status_may', 'status_apr',
                'debt_sep', 'debt_aug', 'debt_jul', 'debt_june', 'debt_may', 'debt_apr',
                'pay_sep', 'pay_aug', 'pay_jul', 'pay_june', 'pay_may', 'pay_apr'
            ]
            
            processed_data_dict = {}
            for k, v in data.items():
                if k in numeric_fields:
                    try:
                        processed_data_dict[k] = float(v)
                    except ValueError:
                        # Assign a default numeric value for invalid or empty inputs
                        processed_data_dict[k] = 0.0
                else:
                    processed_data_dict[k] = v

            # Call the prediction pipeline
            # This function will attempt to re-initialize if assets are None
            prediction_class, prediction_proba = predict_single_input(processed_data_dict)

            # Debug prints after prediction attempt
            print(f"Debug in routes.py /predict: prediction_class = {prediction_class}, type = {type(prediction_class)}")
            print(f"Debug in routes.py /predict: prediction_proba = {prediction_proba}, type = {type(prediction_proba)}")

            # Check if prediction was successful (i.e., not None)
            if prediction_class is None or prediction_proba is None:
                return render_template('index.html', error="Prediction could not be generated. Please check server logs for details.")

            result_text = "Will default next month (Default: Yes)" if prediction_class == 1 else "Will NOT default next month (Default: No)"
            proba_text = f"Probability of default: {prediction_proba:.2f}"

            return render_template('index.html', prediction=result_text, probability=proba_text)

        except Exception as e:
            error_message = f"An unhandled error occurred in /predict route: {e}"
            print(error_message) # Log the error for debugging
            return render_template('index.html', error=error_message)
    
    # This handles GET requests to /predict (unlikely but defensive)
    return render_template('index.html')