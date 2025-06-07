import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def handle_numerical_data(df, col):
    # Check distribution
    skewness = df[col].skew()
    
    if abs(skewness) < 1:  # Normally distributed
        imputer = SimpleImputer(strategy='mean')
    else:  # Skewed distribution
        imputer = SimpleImputer(strategy='median')
    
    df[col] = imputer.fit_transform(df[[col]])
    
    # Handle outliers using capping based on quantiles
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    
    df[col] = np.clip(df[col], q_low, q_high)
    return df

def handle_categorical_data(df, col):
    # Use most frequent value for imputation
    imputer = SimpleImputer(strategy='most_frequent')
    df[col] = imputer.fit_transform(df[[col]])
    
    # Encode categories into numerical labels
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    return df

def handle_datetime_data(df, col):
    # Fill missing dates using linear interpolation (if sequential)
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='time')
    
    # Extract useful features like year, month, day, etc.
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    return df

def context_aware_preprocessing(df, context_clusters):
    for cluster, cols in context_clusters.items():
        for col in cols:
            if df[col].dtype in [np.float64, np.int64]:
                df = handle_numerical_data(df, col)
            elif df[col].dtype == 'object':
                df = handle_categorical_data(df, col)
            elif np.issubdtype(df[col].dtype, np.datetime64):
                df = handle_datetime_data(df, col)
            else:
                print(f"Skipping unsupported column type: {col}")
    return df
