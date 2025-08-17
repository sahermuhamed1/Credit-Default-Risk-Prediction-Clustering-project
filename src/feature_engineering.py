import pandas as pd

def create_financial_ratios(df):
    """
    Creates utilization, payment-to-debt ratios, and their averages/stds.
    """
    months = ['apr', 'may', 'june', 'jul', 'aug', 'sep']

    # Utilization
    for month in months:
        df[f'utilization_{month}'] = df[f'debt_{month}'] / df['limit_bal']
    df['avg_utilization'] = df[[f'utilization_{m}' for m in months]].mean(axis=1)

    # Payment-to-Debt Ratio
    for month in months:
        # Add a small epsilon to avoid division by zero
        df[f'pay_ratio_{month}'] = df[f'pay_{month}'] / (df[f'debt_{month}'] + 1e-6)
    df['avg_pay_ratio'] = df[[f'pay_ratio_{m}' for m in months]].mean(axis=1)
    df['pay_ratio_std'] = df[[f'pay_ratio_{m}' for m in months]].std(axis=1).fillna(0) # Fill NaN from std of single value with 0

    return df

def create_delay_features(df):
    """
    Creates average and maximum delay features.
    """
    months = ['apr', 'may', 'june', 'jul', 'aug', 'sep']
    df['avg_delay'] = df[[f'status_{m}' for m in months]].mean(axis=1)
    df['max_delay'] = df[[f'status_{m}' for m in months]].max(axis=1)
    return df

def engineer_features(df):
    """
    Combines all feature engineering steps.
    """
    df = create_financial_ratios(df)
    df = create_delay_features(df)
    return df

if __name__ == '__main__':
    from data_cleaning import preprocess_data
    df = preprocess_data()
    df_engineered = engineer_features(df.copy())

    print("Engineered data head:")
    print(df_engineered.head())
    print("\nNew engineered columns:")
    new_cols = [col for col in df_engineered.columns if 'utilization' in col or 'pay_ratio' in col or 'delay' in col]
    print(new_cols)