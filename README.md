# Credit Default Risk Prediction & Customer Clustering

This project implements a machine learning pipeline to predict credit default risk and includes modules for customer segmentation using clustering. The solution is modularized into a Python package structure and exposed via a Flask web application for real-time predictions.

**Objective:**
Build an end-to-end pipeline on the **UCI Credit Card Default dataset** to:

* Explore data (EDA)
* Engineer financial features
* Perform statistical sampling & hypothesis testing
* Handle imbalance with SMOTE
* Train ML model (with regularization)
* Evaluate with ROC AUC
* Perform clustering (KMeans, DBSCAN) with PCA & t-SNE visualization
* Deploy inference API with Flask

---

**Project Structure**
```bash
project/
├── app/ # Flask web application
│ ├── init.py # Flask app initialization
│ ├── routes.py # Defines web routes (e.g., /predict)
│ └── templates/ # HTML templates for the web interface
│ └── index.html
├── src/ # Core Python modules for data science workflow
│ ├── data_cleaning.py # Handles initial data loading and cleaning
│ ├── feature_engineering.py # Creates financial ratios and delay features
│ ├── RFM.py # Implements Recency, Frequency, Monetary analysis
│ ├── clustering.py # Contains logic for KMeans, DBSCAN, and PCA (for analysis)
│ ├── model.py # Data splitting, SMOTE, scaling, model training (XGBoost), and saving artifacts
│ └── pipeline.py # Orchestrates preprocessing for single predictions in Flask app
├── data/ # Directory for the raw dataset
│ └── my_dataset.csv # The input dataset (e.g., UCI Credit Card Dataset)
├── models/ # Directory to store trained models and scalers
│ ├── xgboost_model.pkl # Trained XGBoost classification model
│ ├── robust_scaler.pkl # Fitted data scaler
│ └── feature_names.pkl # List of feature names used during model training (important for consistent input)
├── main.py # Script to run the full data processing and model training pipeline
├── run.py # Script to start the Flask web application
└── requirements.txt # List of Python dependencies
```

---

**Steps to Implement:**

1. **Data Loading & Cleaning**

   * Load dataset → `pandas.read_csv`
   * Handle missing values, dtypes

2. **EDA**

   * Distribution of `LIMIT_BAL`, `AGE`, etc.
   * Target imbalance check (`default.payment.next.month`)

3. **Feature Engineering**

   * Create `credit_utilization = BILL_AMT1 / LIMIT_BAL`
   * Create `avg_payment = mean(PAY_AMT1..6)`
   * Scale features with `StandardScaler`

4. **Sampling**

   * **Bootstrap** → sample means of `LIMIT_BAL`
   * **Permutation** → shuffle labels, compare mean differences

5. **RFM Analysis**

   * Recency = last bill date
   * Frequency = avg number of timely payments
   * Monetary = total payments

6. **Outlier Removal**

   * Z-score & IQR methods on `AGE`, `LIMIT_BAL`

7. **Statistical Inference**

   * Confidence Interval (CI) on `LIMIT_BAL` mean
   * **z-test / t-test** → compare defaulters vs non-defaulters
   * **Chi-Square** → categorical feature importance
   * **ANOVA** → group comparisons (e.g., education vs default)

8. **Imbalance Handling**

   * Use **SMOTE** to oversample minority (defaults)

9. **ML Modeling**

   * Logistic Regression (baseline)
   * Add regularization (L1/L2) → bias-variance tradeoff
   * Evaluate with **ROC AUC, Confusion Matrix**

10. **Clustering & Dimensionality Reduction**

    * PCA → 2D/3D transformation
    * KMeans & DBSCAN clustering
    * Visualize with **t-SNE**

11. **Hypothesis Testing Context**

    * Define null & alt hypotheses (e.g., credit limit differs by default status)
    * Discuss **Type I (false alarm) vs Type II (missed detection)**

12. **Deployment (Flask API)**

    * Build `/predict` endpoint → takes JSON with client features → returns default probability + cluster label
    * Save model & scaler with `joblib`

---

**Create an Environment and Install Dependencies:**
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Using conda
conda create -n credit_pred_env python=3.9
conda activate credit_pred_env

# Or using venv
python -m venv venv_credit_pred
source venv_credit_pred/bin/activate # On Windows: .\venv_credit_pred\Scripts\activate

pip install -r requirements.txt
```

---

**Train the Machine Learning Model:**

Run the main.py script to perform data cleaning, feature engineering, RFM analysis, and train the XGBoost model. This will also save the trained model, scaler, and a list of feature names into the models/ directory.

```Bash
python main.py
```
*Important: Observe the terminal output during this step. Especially note the Features used for training/prediction: [...] line, as it confirms the exact features the model was trained on.*

---
**Start the Flask Web Application:**

Once the model is trained and saved, you can start the Flask web application.

```Bash
python run.py
```
---

**Contanct Information**
For any questions or issues, please contact me at sahermuhamed176@gmail.com. I'll be happy to assist.