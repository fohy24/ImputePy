import unittest
import pandas as pd
import numpy as np
# Adjust the import path as necessary
from imputepy import *


class TestImputationFunctions(unittest.TestCase):
    def test_cols_to_impute_with_missing_values(self):
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, 6], 'C': [np.nan, np.nan, 9]})
        expected = ['A', 'C']
        result = cols_to_impute(df)
        self.assertListEqual(sorted(expected), sorted(result), "Failed to identify all columns with missing values.")

    def test_cols_to_impute_without_missing_values(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        expected = []
        result = cols_to_impute(df)
        self.assertListEqual(expected, result, "Incorrectly identified columns with missing values when there are none.")

    def test_missing_indices_single_column(self):
        df = pd.DataFrame({'A': [1, np.nan, 3]})
        expected = {'A': [1]}
        result = missing_indices(df)
        self.assertDictEqual(expected, result, "Failed to correctly identify indices of missing values in a single column.")

    def test_missing_indices_multiple_columns(self):
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, 3]})
        expected = {'A': [1], 'B': [0]}
        result = missing_indices(df)
        self.assertDictEqual(expected, result, "Failed to correctly identify indices of missing values across multiple columns.")

    def test_find_cat_with_categorical_columns(self):
        df = pd.DataFrame({'Num': [1, 2, 3, 4], 'Cat': pd.Categorical(['a', 'b', 'a', 'b']), 'SmallNum': [1, 1, 2, 2]})
        expected = ['Num', 'SmallNum']  # Should return only numeric columns with less than 15 unique values
        result = find_cat(df)
        self.assertListEqual(expected, result, "Failed to correctly identify numerical columns as categorical based on unique values.")

    def test_find_cat_without_categorical_columns(self):
        df = pd.DataFrame({'Num': [1, 2, 3, 4], 'LargeNum': [10, 20, 30, 40]})
        expected = ['Num', 'LargeNum']  # Both columns should be considered categorical within the default unique_count_limit
        result = find_cat(df)
        self.assertListEqual(expected, result, "Incorrectly identified columns as categorical when they should not be.")

    def test_column_filter_above_limit(self):
        df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3], 'C': [100, 200, 300]})
        expected = []  # No column has fewer unique values than the limit of 2
        result = column_filter(df, ['A', 'B', 'C'], filter_upper_limit=2)
        self.assertListEqual(expected, result, "Incorrectly included columns that exceed the unique value limit.")

    def test_LGBMimputer_imputes_missing_values(self):
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [1, 2, np.nan]})
        # Assuming LGBMimputer is properly configured to impute missing values
        result_df = LGBMimputer(df)
        missing_after_imputation = cols_to_impute(result_df)
        self.assertTrue(len(missing_after_imputation) == 0, "Failed to impute all missing values.")

if __name__ == '__main__':
    unittest.main()
