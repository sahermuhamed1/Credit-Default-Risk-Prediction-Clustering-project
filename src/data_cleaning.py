
import pandas as pd

def load_data(filepath='data/my_dataset.csv'):
    """
    Loads the dataset from the specified filepath.
    """
    df = pd.read_csv(filepath)
    return df

def clean_column_names(df):
    """
    Cleans and renames columns for consistency.
    """
    df.columns = df.columns.str.lower()
    df.rename(columns={
        'pay_0': 'status_sep', 'pay_2': 'status_aug', 'pay_3': 'status_jul',
        'pay_4': 'status_june', 'pay_5': 'status_may', 'pay_6': 'status_apr',
        'bill_amt1': 'debt_sep', 'bill_amt2': 'debt_aug', 'bill_amt3': 'debt_jul',
        'bill_amt4': 'debt_june', 'bill_amt5': 'debt_may', 'bill_amt6': 'debt_apr',
        'pay_amt1': 'pay_sep', 'pay_amt2': 'pay_aug', 'pay_amt3': 'pay_jul',
        'pay_amt4': 'pay_june', 'pay_amt5': 'pay_may', 'pay_amt6': 'pay_apr',
        'default.payment.next.month': 'default_payment'
    }, inplace=True)
    return df

def select_relevant_columns(df):
    """
    Selects and reorders relevant columns, placing the target at the end.
    """
    base_cols = ["id", "limit_bal", "sex", "education", "marriage", "age"]
    pay_cols = ['status_sep', 'status_aug', 'status_jul', 'status_june', 'status_may', 'status_apr']
    bill_cols = ['debt_sep', 'debt_aug', 'debt_jul', 'debt_june', 'debt_may', 'debt_apr']
    pay_amt_cols = ['pay_sep', 'pay_aug', 'pay_jul', 'pay_june', 'pay_may', 'pay_apr']
    target_col = ["default_payment"]

    df = df[base_cols + pay_cols + bill_cols + pay_amt_cols + target_col]
    
    # Reorder target to the end if it's not already
    current_cols = df.columns.tolist()
    if 'default_payment' in current_cols:
        current_cols.remove('default_payment')
        df = df[current_cols + ['default_payment']]
    
    return df

def preprocess_data(filepath='data/my_dataset.csv'):
    """
    Combines data loading and initial cleaning steps.
    """
    df = load_data(filepath)
    df = clean_column_names(df)
    df = select_relevant_columns(df)
    return df

if __name__ == '__main__':
    # Example usage:
    df = preprocess_data()
    print("Cleaned data head:")
    print(df.head())
    print("\nCleaned data columns:")
    print(df.columns)
    print("\nDefault payment distribution:")
    print(df['default_payment'].value_counts())