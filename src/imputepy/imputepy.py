import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
# from lightgbm import LGBMRegressor


def read_data(path):
    df = pd.read_csv(path)
    return df


def cols_to_impute(df):
    cols = []
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            cols.append(col)
    return cols


def main(path, method):
    df = read_data(path)
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(exclude='number').columns

    missing_cols = cols_to_impute(df)

    numeric_missing_cols = list(set(missing_cols) & set(numeric_cols))
    categorical_missing_cols = list(
        set(missing_cols) & set(categorical_cols))

    df_numeric = df.select_dtypes(include='number')
    df_categoric = df.select_dtypes(exclude='number')

    train_df_numeric = df_numeric.dropna()
    train_df_categoric = df_categoric.dropna()

    # Define imputer
    if method == 'mean':
        imputer_numeric = SimpleImputer(strategy='mean')
        imputer_categoric = SimpleImputer(strategy='most_frequent')

    elif method == 'median':
        imputer_numeric = SimpleImputer(strategy='median')
        imputer_categoric = SimpleImputer(strategy='most_frequent')

    elif method == 'lgbm':
        imputer_numeric = LGBMRegressor()
        imputer_categoric = LGBMRegressor()

    # imputer_numeric.fit(df_numeric)
    # imputer_categoric.fit(df_categoric)

    df_numeric = pd.DataFrame(imputer_numeric.fit_transform(df_numeric),
                              columns=df_numeric.columns)

    df_categoric = pd.DataFrame(imputer_categoric.fit_transform(df_categoric),
                                columns=df_categoric.columns)

    df_imp = pd.concat([df_numeric, df_categoric], axis=1)
    df_imp = df_imp[df.columns]

    return df_imp.info()


if __name__ == "__main__":
    print(main('data\df.csv', 'mean'))
