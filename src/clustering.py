import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

def remove_outliers_iqr(df_numeric):
    """
    Remove outliers from all numeric columns using IQR.
    This function expects a DataFrame containing only numeric columns
    to avoid issues with non-numeric data during quantile calculation.
    """
    for col in df_numeric.columns:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_numeric = df_numeric[(df_numeric[col] >= lower) & (df_numeric[col] <= upper)]
    return df_numeric

def perform_kmeans_clustering(df, features_for_clustering=None, optimal_k=3, remove_outliers=True):
    """
    Performs KMeans clustering on specified features, with optional outlier removal.
    """
    if features_for_clustering is None:
        features_for_clustering = ['limit_bal', 'avg_delay', 'avg_utilization', 'avg_pay_ratio', 'R', 'F', 'M']

    X = df[features_for_clustering].copy()

    if remove_outliers:
        print(f"Original shape before KMeans outlier removal: {X.shape}")
        X = remove_outliers_iqr(X)
        print(f"Shape after KMeans outlier removal: {X.shape}")

    # Scale data for KMeans
    scaler = StandardScaler() # Or RobustScaler, depending on data distribution
    X_scaled = scaler.fit_transform(X)

    # Elbow Method (for analysis, not direct use in function)
    # inertia = []
    # K_range = range(2, 11)
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(X_scaled)
    #     inertia.append(kmeans.inertia_)
    # plt.figure(figsize=(8,4))
    # plt.plot(K_range, inertia, 'bo-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for KMeans')
    # plt.show()

    # Silhouette Analysis (for analysis)
    # silhouette_scores = []
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     labels = kmeans.fit_predict(X_scaled)
    #     if len(np.unique(labels)) > 1: # Silhouette score needs at least 2 clusters
    #         score = silhouette_score(X_scaled, labels)
    #         silhouette_scores.append(score)
    #     else:
    #         silhouette_scores.append(-1) # Placeholder for single cluster case
    # plt.figure(figsize=(8,4))
    # plt.plot(K_range, silhouette_scores, 'ro-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Analysis for KMeans')
    # plt.show()

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    # Note: X.index is used to align clusters with original df
    X['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Merge cluster assignments back to the original (full) DataFrame
    df_with_clusters = df.copy()
    df_with_clusters = df_with_clusters.merge(X[['cluster']], left_index=True, right_index=True, how='left')
    df_with_clusters['cluster'] = df_with_clusters['cluster'].fillna(-1).astype(int) # -1 for rows removed as outliers

    print("\nKMeans Cluster distribution:")
    print(df_with_clusters['cluster'].value_counts())
    return df_with_clusters

def perform_dbscan_clustering(df, features_for_clustering=None, eps=1.5, min_samples=500):
    """
    Performs DBSCAN clustering on specified features after PCA.
    """
    if features_for_clustering is None:
        features_for_clustering = [col for col in df.columns if col not in ['id', 'default_payment']]

    X = df[features_for_clustering].copy().values

    # Scale data for DBSCAN (RobustScaler is often good for DBSCAN as it handles outliers better)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for dimensionality reduction (essential for DBSCAN on high-dim data)
    pca = PCA(n_components=0.95, random_state=42) # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_pca)

    df_with_clusters = df.copy()
    df_with_clusters['dbscan_cluster'] = clusters

    print("DBSCAN Cluster distribution (noisy points are -1):")
    print(Counter(clusters))
    return df_with_clusters

if __name__ == '__main__':
    # Example usage (assuming data_cleaning, feature_engineering, RFM can provide a df)
    from data_cleaning import preprocess_data
    from feature_engineering import engineer_features
    from RFM import add_rfm_features

    df = preprocess_data()
    df_engineered = engineer_features(df.copy())
    df_rfm = add_rfm_features(df_engineered.copy())

    # Example KMeans
    print("\n--- Performing KMeans Clustering ---")
    df_kmeans_clustered = perform_kmeans_clustering(df_rfm.copy(), remove_outliers=True)
    print("KMeans clustered data head:")
    print(df_kmeans_clustered[['R', 'F', 'M', 'cluster']].head())

    # Example DBSCAN
    print("\n--- Performing DBSCAN Clustering ---")
    # For DBSCAN, we'll use all preprocessed and engineered features as in the notebook.
    # Exclude 'id' and 'default_payment' if they exist and are not features.
    features_for_dbscan = [col for col in df_rfm.columns if col not in ['id', 'default_payment']]
    df_dbscan_clustered = perform_dbscan_clustering(df_rfm.copy(), features_for_clustering=features_for_dbscan)
    print("DBSCAN clustered data head:")
    print(df_dbscan_clustered[['dbscan_cluster']].head())