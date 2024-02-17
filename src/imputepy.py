import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier


def cols_to_impute(df):
    """
    Identify columns in a DataFrame that contain missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    list: A list of column names that have at least one missing value.
    """
    cols = []
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            cols.append(col)
    return cols


def missing_indices(df):
    """
    Return a dictionary mapping column names to lists of indices where missing values occur.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze for missing values.

    Returns:
    dict: A dictionary where keys are column names with missing values, and values are lists of indices
          with missing values in those columns.
    """
    indices = {}
    for col in cols_to_impute(df):
        indices[col] = df[df[col].isnull()].index.tolist()
    return indices


def find_cat(df, unique_count_limit=15):
    """
    Identify numerical columns in a DataFrame that could be considered categorical based on a threshold of unique values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to search within.
    - unique_count_limit (int, optional): The maximum number of unique values a column can have to be considered categorical. Defaults to 15.

    Returns:
    - list: A list of column names that have fewer than `unique_count_limit` unique values, suggesting they could be treated as categorical.
    """
    possible_cat = []
    for col in df.select_dtypes(include='number').columns:
        unique_count = np.count_nonzero(df[col].unique())
        if unique_count < unique_count_limit:
            possible_cat.append(col)
    return possible_cat

def column_filter(df, cat_cols, filter_upper_limit):
    """
    Filters out categorical columns based on the number of unique values, considering only those below a specified limit.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to filter.
    - cat_cols (list): A list of column names to consider for filtering.
    - filter_upper_limit (int): The upper limit for the number of unique values in a column to be included in the filtered list.

    Returns:
    - list: A list of column names that have a number of unique values below or equal to `filter_upper_limit`.
    """    
    filtered = []
    for col in cat_cols:
        if np.count_nonzero(df[col].unique()) <= filter_upper_limit:
            filtered.append(col)
    return filtered

def LGBMimputer(df, filter=True, exclude=None, filter_upper_limit=50, unique_count_limit=15):
    """
    Imputes missing values in a DataFrame using LightGBM models, with separate handling for categorical and numerical columns.

    This function:
    - Optionally excludes specified columns.
    - Identifies categorical columns based on data type and a uniqueness threshold.
    - Imputes missing values using LightGBM models: LGBMClassifier for categorical columns and LGBMRegressor for numerical columns.
    - Returns the DataFrame with imputed values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process and impute missing values.
    - filter (bool, optional): If True, filters categorical columns to only those with a unique value count below `filter_upper_limit`. Defaults to True.
    - exclude (list, optional): A list of column names to exclude from processing. Defaults to None.
    - filter_upper_limit (int, optional): The maximum number of unique values a column can have to be considered categorical during filtering. Defaults to 50.

    Returns:
    - pd.DataFrame: The DataFrame with missing values imputed.
    """
    if exclude != None:
        df.drop(exclude, axis=1, inplace=True)

    cat_cols = df.select_dtypes(exclude='number').columns.to_list()
    cat_cols += find_cat(df, unique_count_limit)
    
    if filter:
        cat_cols = column_filter(df, cat_cols, filter_upper_limit=filter_upper_limit)

    df[cat_cols] = df[cat_cols].astype('category')


    num_cols = df.select_dtypes(include='number').columns.to_list()
    missing_cols = list(set(cols_to_impute(df)) & set(cat_cols + num_cols)) # intersection of the lists
    print(f'{len(missing_cols)} columns will be imputed: {missing_cols}')
    df = df[missing_cols]

    pred = {}
    for i, target_column in enumerate(missing_cols):
        print(f'target column: {target_column}')

        # select imputer
        if target_column in cat_cols:
            imputer = LGBMClassifier(n_jobs=-1, verbose=-1)
        else:
            imputer = LGBMRegressor(n_jobs=-1, verbose=-1)

        # split trainset testset
        train_df = df.dropna()
        test_df = df[df[target_column].isnull()]
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])

        # fitting
        imputer.fit(X_train, y_train)
        print(f'{i+1}/{len(missing_cols)} columns fitted')

        # prediction
        pred[target_column] = imputer.predict(X_test)

        # fill na
        for i, index in enumerate(missing_indices(df)[target_column]):
            df.loc[index, target_column] = pred[target_column][i]

    return df


if __name__ == '__main__':
    LGBMimputer()
