# imputepy

Impute missing values using Lightgbm.

## Installation

```bash
pip install imputepy
```

## Features

- **Automated Imputation:** Utilizes LightGBM models to impute missing values, selecting between regression and classification models based on the column's data type.
- **Flexible Column Exclusion:** Allows specific columns to be excluded from the imputation process.
- **Dynamic Filtering for Categorical Columns:** Filters categorical columns based on a specified upper limit of unique values to enhance efficiency.
- **Customizable Thresholds for Categorical Detection:** Enables setting custom thresholds for unique value counts to refine which columns are considered categorical.
- **Comprehensive Imputation Strategy:** Combines missing value identification, column type determination, and the application of LightGBM models for effective imputation.
- **Direct Imputation into Original DataFrame:** Imputes missing values directly into the original DataFrame, maintaining the data structure for seamless data preprocessing integration.


## Usage

```
from imputepy import LGBMimputer
import pandas as pd
import numpy as np

df = pd.read_csv('data/df.csv')
df_imp = LGBMimputer(df, filter=True, exclude=None, filter_upper_limit=50, unique_count_limit=15)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`imputepy` was created by Sam Fo. It is licensed under the terms of the MIT license.

## Credits

`imputepy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
