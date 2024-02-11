# imputepy

Impute missing values using Lightgbm.

## Installation

```bash
pip install imputepy
```

## Features

- **Identify Columns with Missing Values:** Automatically detect columns in a DataFrame that have missing data.
- **Find Missing Indices:** Get the indices of missing values for targeted imputation.
- **Categorical Column Detection:** Identify potential categorical columns based on the count of unique values.
- **Automated Imputation:** Utilize LightGBM models to impute missing values, choosing between regression and classification based on the data type.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`imputepy` was created by Sam Fo. It is licensed under the terms of the MIT license.

## Credits

`imputepy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
