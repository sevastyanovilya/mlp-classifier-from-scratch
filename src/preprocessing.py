"""
Data Preprocessing Module

This module provides utilities for preparing data for neural network training,
specifically designed for the vehicle purchase quality prediction problem.

Key concepts:
    - Temporal split: Splitting time-series data chronologically to prevent data leakage
    - Feature engineering: Converting categorical and numerical features for ML
    - Missing value handling: Different strategies for different feature types

In time-series classification, temporal split is crucial because:
    1. It simulates real deployment (predicting future from past)
    2. Prevents look-ahead bias where future information leaks into training
    3. Provides a more realistic estimate of model performance

Author: Ilya Sevastyanov
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def temporal_split(
    df: pd.DataFrame,
    date_column: str = 'PurchDate',
    train_ratio: float = 1/3,
    valid_ratio: float = 1/3
) -> tuple:
    """
    Split dataset chronologically for time-series cross-validation.
    
    In many real-world scenarios, data has a temporal component where we want
    to predict future outcomes based on past observations. Using random splits
    would cause data leakage because:
        - Training samples could come after test samples in time
        - The model would implicitly learn from future information
        - Performance estimates would be overly optimistic
    
    Temporal split ensures: train_date < validation_date < test_date
    
    Args:
        df: DataFrame with a date column
        date_column: Name of the datetime column to use for sorting
        train_ratio: Fraction of data for training (default 33%)
        valid_ratio: Fraction of data for validation (default 33%)
        
    Returns:
        Tuple of (df_train, df_valid, df_test)
        
    Example:
        >>> df_train, df_valid, df_test = temporal_split(data, 'PurchDate')
        >>> # df_train contains earliest records
        >>> # df_valid contains middle records
        >>> # df_test contains most recent records
    """
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Ensure date column is datetime type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date for chronological ordering
    df = df.sort_values(by=date_column).reset_index(drop=True)

    # Calculate split indices
    n = len(df)
    split1_idx = int(n * train_ratio)
    split2_idx = int(n * (train_ratio + valid_ratio))

    # Split the data
    df_train = df.iloc[:split1_idx].reset_index(drop=True)
    df_valid = df.iloc[split1_idx:split2_idx].reset_index(drop=True)
    df_test = df.iloc[split2_idx:].reset_index(drop=True)

    # Print split information
    print('Dataset split (train/valid/test):')
    print(f'  Train: {len(df_train)/len(df)*100:.1f}% ({len(df_train)} samples)')
    print(f'  Valid: {len(df_valid)/len(df)*100:.1f}% ({len(df_valid)} samples)')
    print(f'  Test:  {len(df_test)/len(df)*100:.1f}% ({len(df_test)} samples)')
    print(f'\nDate ranges:')
    print(f'  Train: {df_train[date_column].min()} — {df_train[date_column].max()}')
    print(f'  Valid: {df_valid[date_column].min()} — {df_valid[date_column].max()}')
    print(f'  Test:  {df_test[date_column].min()} — {df_test[date_column].max()}')

    return df_train, df_valid, df_test


def prepare_features(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
    target_col: str = 'IsBadBuy',
    date_col: str = 'PurchDate'
) -> tuple:
    """
    Prepare features for neural network training.
    
    This function performs the following preprocessing steps:
    
    1. Extract target variable (y) from each split
    
    2. Handle missing values:
       - Categorical: Replace with 'missing' placeholder
       - Numerical: Impute with median from training set
       
       Using median (not mean) for numerical features because:
       - More robust to outliers
       - Preserves typical values better for skewed distributions
       
    3. Encode categorical features:
       - One-Hot Encoding creates binary columns for each category
       - 'handle_unknown=ignore' handles new categories in test data
       
    4. Scale numerical features:
       - StandardScaler normalizes to mean=0, std=1
       - This is crucial for neural networks because:
         a) Gradient descent converges faster with similar feature scales
         b) Prevents large-valued features from dominating
         
    Important: All transformers (imputer, encoder, scaler) are fit ONLY on
    training data to prevent data leakage from validation/test sets.
    
    Args:
        df_train, df_valid, df_test: DataFrames from temporal_split
        cat_cols: List of categorical column names
        num_cols: List of numerical column names
        target_col: Name of the target variable column
        date_col: Name of the date column (to be excluded from features)
        
    Returns:
        Tuple of (X_train, X_valid, X_test, y_train, y_valid, y_test)
        where X arrays are numpy arrays with processed features
    """
    # Extract target variable
    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values
    y_test = df_test[target_col].values

    # Remove target and date columns from features
    X_train = df_train.drop(columns=[target_col, date_col])
    X_valid = df_valid.drop(columns=[target_col, date_col])
    X_test = df_test.drop(columns=[target_col, date_col])

    # Handle missing values in categorical columns
    # Replace NaN with 'missing' - a simple but effective approach
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('missing').astype(str)
        X_valid[col] = X_valid[col].fillna('missing').astype(str)
        X_test[col] = X_test[col].fillna('missing').astype(str)

    # Handle missing values in numerical columns using median imputation
    # Fit on training data only to prevent leakage
    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_valid[num_cols] = num_imputer.transform(X_valid[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # One-Hot Encoding for categorical features
    # sparse_output=False gives dense arrays (required for numpy operations)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_valid_cat = encoder.transform(X_valid[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    # Scale numerical features to zero mean and unit variance
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_valid_num = scaler.transform(X_valid[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    # Combine numerical and categorical features
    # Order: numerical first, then categorical (by convention)
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_valid_processed = np.hstack([X_valid_num, X_valid_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    print(f'Dimensionality after preprocessing:')
    print(f'  X_train: {X_train_processed.shape}')
    print(f'  X_valid: {X_valid_processed.shape}')
    print(f'  X_test:  {X_test_processed.shape}')

    return X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test


def get_feature_columns(df: pd.DataFrame, target_col: str = 'IsBadBuy') -> tuple:
    """
    Automatically identify categorical and numerical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column to exclude
        
    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target from numerical columns if present
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    print(f'Numerical features ({len(numerical_cols)}): {numerical_cols}')
    print(f'Categorical features ({len(categorical_cols)}): {categorical_cols}')
    
    return categorical_cols, numerical_cols
