implementing a Credit Default Risk Prediction & Clustering project

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

 That’s the **MVP roadmap**: it touches **every required technique** while keeping it structured for deployment.

Be prepared if I need anything from you ok, this project I do to prepare to my interview at banque misr so its important for making me revision to all statistics, ML, model evaluation before the interview

don't give me the code or anything, If I need anything from you I'll ask. Just be prepared