import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier


def cols_to_impute(df):
    """Return columns with missing values in a list"""
    cols = []
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            cols.append(col)
    return cols


def missing_indices(df):
    """Return the indices of the missing values in all columns of a dataframe"""
    indices = {}
    for col in cols_to_impute(df):
        indices[col] = df[df[col].isnull()].index.tolist()
    return indices


def find_cat(df, unique_count_lim=15):
    """Return the column names that have less than 15 unique values"""
    possible_cat = []
    for col in df.select_dtypes(include='number').columns:
        unique_count = np.count_nonzero(df[col].unique())
        if unique_count < unique_count_lim:
            possible_cat.append(col)
    return possible_cat


def main(path, exclude=None):
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
