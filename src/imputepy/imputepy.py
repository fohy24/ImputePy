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


def find_cat(df, unique_count_lim=15):
    """
    Identify numerical columns in a DataFrame that could be considered categorical based on a threshold of unique values.

    Parameters:
    df (pd.DataFrame): The DataFrame to search for potential categorical columns.
    unique_count_lim (int, optional): The maximum number of unique values a column can have to be considered categorical. Defaults to 15.

    Returns:
    list: A list of column names that have fewer than `unique_count_lim` unique values.
    """
    possible_cat = []
    for col in df.select_dtypes(include='number').columns:
        unique_count = np.count_nonzero(df[col].unique())
        if unique_count < unique_count_lim:
            possible_cat.append(col)
    return possible_cat


def LGBMimputer(path, exclude=None):
    """
    Main function to process the DataFrame from a CSV file, impute missing values using LightGBM, and return the processed DataFrame.

    This function performs the following steps:
    - Reads a DataFrame from a specified CSV file.
    - Optionally excludes specified columns.
    - Identifies categorical columns.
    - Imputes missing values using LightGBM models (Classifier or Regressor based on the column data type).
    - Returns the imputed DataFrame.

    Parameters:
    path (str): Path to the CSV file to be processed.
    exclude (list, optional): A list of column names to be excluded from the DataFrame before processing. Defaults to None.

    Returns:
    pd.DataFrame: The DataFrame with missing values imputed.
    """
    df = pd.read_csv(path)
    if exclude != None:
        df.drop(exclude, axis=1, inplace=True)

    cat_cols = df.select_dtypes(exclude='number').columns.to_list()
    cat_cols += find_cat(df)
    df[cat_cols] = df[cat_cols].astype('category')
    missing_cols = cols_to_impute(df)

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
    main()
