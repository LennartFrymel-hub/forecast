import warnings

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.model_selection import TimeSeriesFold
from spotforecast2_safe.model_selection.utils_common import (
    OneStepAheadValidationWarning,
    check_backtesting_input,
    initialize_lags_grid,
    select_n_jobs_backtesting,
)


def test_onestepahead_validation_warning_example():
    """
    >>> import warnings
    >>> from spotforecast2_safe.model_selection.utils_common import OneStepAheadValidationWarning
    >>> warnings.warn(
    ...     "This is a one-step-ahead validation warning.",
    ...     OneStepAheadValidationWarning
    ... )
    This is a one-step-ahead validation warning.
    You can suppress this warning using: warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)
    """
    with pytest.warns(OneStepAheadValidationWarning) as record:
        warnings.warn(
            "This is a one-step-ahead validation warning.",
            OneStepAheadValidationWarning,
        )

    assert len(record) == 1
    assert "This is a one-step-ahead validation warning." in str(record[0].message)
    assert (
        "You can suppress this warning using: warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)"
        in str(record[0].message)
    )


def test_initialize_lags_grid_example():
    """
    >>> from spotforecast2_safe.model_selection.utils_common import initialize_lags_grid
    >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    >>> from sklearn.linear_model import LinearRegression
    >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    >>> lags_grid = [2, 4]
    >>> lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
    >>> print(lags_grid)
    {'2': 2, '4': 4}
    >>> print(lags_label)
    values
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    lags_grid_input = [2, 4]
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid_input)

    assert lags_grid == {"2": 2, "4": 4}
    assert lags_label == "values"


def test_check_backtesting_input_example():
    """
    >>> import pandas as pd
    >>> from spotforecast2_safe.model_selection.utils_common import check_backtesting_input
    >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    >>> from spotforecast2_safe.model_selection import TimeSeriesFold
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.metrics import mean_squared_error
    >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    >>> cv = TimeSeriesFold(
    ...     steps=3,
    ...     initial_train_size=5,
    ...     gap=0,
    ...     refit=False,
    ...     fixed_train_size=False,
    ...     allow_incomplete_fold=True
    ... )
    >>> check_backtesting_input(
    ...     forecaster=forecaster,
    ...     cv=cv,
    ...     metric=mean_squared_error,
    ...     y=y
    ... )
    """
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=5,
        gap=0,
        refit=False,
        fixed_train_size=False,
        allow_incomplete_fold=True,
    )
    # Should not raise any exception
    check_backtesting_input(
        forecaster=forecaster, cv=cv, metric=mean_squared_error, y=y
    )


def test_select_n_jobs_backtesting_example():
    """
    >>> from spotforecast2_safe.model_selection.utils_common import select_n_jobs_backtesting
    >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    >>> from sklearn.linear_model import LinearRegression
    >>> forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    >>> select_n_jobs_backtesting(forecaster, refit=True)
    1
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=2)
    n_jobs = select_n_jobs_backtesting(forecaster, refit=True)
    assert n_jobs == 1
