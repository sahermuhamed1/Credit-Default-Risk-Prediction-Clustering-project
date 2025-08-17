import pandas as pd

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary features.
    Assumes 'pay_apr' through 'pay_sep' columns exist.
    """
    months = ['apr', 'may', 'june', 'jul', 'aug', 'sep']
    month_order = {m: i + 1 for i, m in enumerate(months)} # 1 for Apr, 6 for Sep

    # Recency: latest month with nonzero payment (higher value means more recent payment)
    # If no payment, R=0 or a value indicating no recent activity.
    df['R'] = df[[f'pay_{m}' for m in months]].apply(
        lambda row: max([month_order[m] for m in months if row[f'pay_{m}'] > 0], default=0),
        axis=1
    )

    # Frequency: number of transactions (nonzero payments) made in the period
    df['F'] = df[[f'pay_{m}' for m in months]].gt(0).sum(axis=1)

    # Monetary: total payment across Apr–Sep
    df['M'] = df[[f'pay_{m}' for m in months]].sum(axis=1)

    return df

def add_rfm_features(df):
    """
    Adds RFM features to the DataFrame.
    """
    df = calculate_rfm(df)
    return df

if __name__ == '__main__':
    # Example usage (assuming data_cleaning and feature_engineering can provide a df)
    from data_cleaning import preprocess_data
    from feature_engineering import engineer_features

    df = preprocess_data()
    df_engineered = engineer_features(df.copy())
    df_rfm = add_rfm_features(df_engineered.copy())

    print("RFM data head:")
    print(df_rfm[['R', 'F', 'M']].head())
    print("\nR value counts:")
    print(df_rfm['R'].value_counts())
    print("\nF value counts:")
    print(df_rfm['F'].value_counts())