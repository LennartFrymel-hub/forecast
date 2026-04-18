# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Validation utilities for time series forecasting.

This module provides validation functions for time series data and exogenous variables.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.exceptions import (
    DataTypeWarning,
    MissingValuesWarning,
    UnknownLevelWarning,
)


def check_y(y: Any, series_id: str = "`y`") -> None:
    """
    Validate that y is a pandas Series without missing values.

    This function ensures that the input time series meets the basic requirements
    for forecasting: it must be a pandas Series and must not contain any NaN values.

    Args:
        y: Time series values to validate.
        series_id: Identifier of the series used in error messages. Defaults to "`y`".

    Raises:
        TypeError: If y is not a pandas Series.
        ValueError: If y contains missing (NaN) values.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.utils.validation import check_y
        >>>
        >>> # Valid series
        >>> y = pd.Series([1, 2, 3, 4, 5])
        >>> check_y(y)  # No error
        >>>
        >>> # Invalid: not a Series
        >>> try:
        ...     check_y([1, 2, 3])
        ... except TypeError as e:
        ...     print(f"Error: {e}")
        Error: `y` must be a pandas Series with a DatetimeIndex or a RangeIndex. Found <class 'list'>.
        >>>
        >>> # Invalid: contains NaN
        >>> y_with_nan = pd.Series([1, 2, np.nan, 4])
        >>> try:
        ...     check_y(y_with_nan)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: `y` has missing values.
    """
    if not isinstance(y, pd.Series):
        raise TypeError(
            f"{series_id} must be a pandas Series with a DatetimeIndex or a RangeIndex. "
            f"Found {type(y)}."
        )

    if y.isna().to_numpy().any():
        raise ValueError(f"{series_id} has missing values.")

    return


def check_exog(
    exog: Union[pd.Series, pd.DataFrame],
    allow_nan: bool = True,
    series_id: str = "`exog`",
) -> None:
    """
    Validate that exog is a pandas Series or DataFrame.

    This function ensures that exogenous variables meet basic requirements:
    - Must be a pandas Series or DataFrame
    - If Series, must have a name
    - Optionally warns if NaN values are present

    Args:
        exog: Exogenous variable/s included as predictor/s.
        allow_nan: If True, allows NaN values but issues a warning. If False,
            raises no warning about NaN values. Defaults to True.
        series_id: Identifier of the series used in error messages. Defaults to "`exog`".

    Raises:
        TypeError: If exog is not a pandas Series or DataFrame.
        ValueError: If exog is a Series without a name.

    Warnings:
        MissingValuesWarning: If allow_nan=True and exog contains NaN values.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.utils.validation import check_exog
        >>>
        >>> # Valid DataFrame
        >>> exog_df = pd.DataFrame({"temp": [20, 21, 22], "humidity": [50, 55, 60]})
        >>> check_exog(exog_df)  # No error
        >>>
        >>> # Valid Series with name
        >>> exog_series = pd.Series([1, 2, 3], name="temperature")
        >>> check_exog(exog_series)  # No error
        >>>
        >>> # Invalid: Series without name
        >>> exog_no_name = pd.Series([1, 2, 3])
        >>> try:
        ...     check_exog(exog_no_name)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: When `exog` is a pandas Series, it must have a name.
        >>>
        >>> # Invalid: not a Series/DataFrame
        >>> try:
        ...     check_exog([1, 2, 3])
        ... except TypeError as e:
        ...     print(f"Error: {e}")
        Error: `exog` must be a pandas Series or DataFrame. Got <class 'list'>.
    """
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{series_id} must be a pandas Series or DataFrame. Got {type(exog)}."
        )

    if isinstance(exog, pd.Series) and exog.name is None:
        raise ValueError(f"When {series_id} is a pandas Series, it must have a name.")

    if not allow_nan:
        if exog.isna().to_numpy().any():
            warnings.warn(
                f"{series_id} has missing values. Most machine learning models "
                f"do not allow missing values. Fitting the forecaster may fail.",
                MissingValuesWarning,
            )

    return


def check_exog_dtypes(
    exog: Union[pd.Series, pd.DataFrame],
    call_check_exog: bool = True,
    series_id: str = "`exog`",
) -> None:
    """
    Check that exogenous variables have valid data types (int, float, category).

    This function validates that the exogenous variables (Series or DataFrame)
    contain only supported data types: integer, float, or category. It issues a
    warning if other types (like object/string) are found, as these may cause
    issues with some machine learning estimators.

    It also strictly enforces that categorical columns must have integer categories.

    Args:
        exog: Exogenous variables to check.
        call_check_exog: If True, calls check_exog() first to ensure basic validity.
            Defaults to True.
        series_id: Identifier used in warning/error messages. Defaults to "`exog`".

    Raises:
        TypeError: If categorical columns contain non-integer categories.

    Warnings:
        DataTypeWarning: If columns with unsupported data types (not int, float, category)
            are found.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.utils.validation import check_exog_dtypes
        >>>
        >>> # Valid types (float, int)
        >>> df_valid = pd.DataFrame({
        ...     "a": [1.0, 2.0, 3.0],
        ...     "b": [1, 2, 3]
        ... })
        >>> check_exog_dtypes(df_valid)  # No warning
        >>>
        >>> # Invalid type (object/string)
        >>> df_invalid = pd.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "b": ["x", "y", "z"]
        ... })
        >>> check_exog_dtypes(df_invalid)
        ... # Issues DataTypeWarning about column 'b'
        >>>
        >>> # Valid categorical (with integer categories)
        >>> df_cat = pd.DataFrame({"a": [1, 2, 1]})
        >>> df_cat["a"] = df_cat["a"].astype("category")
        >>> check_exog_dtypes(df_cat)  # No warning
    """
    if call_check_exog:
        check_exog(exog=exog, allow_nan=False, series_id=series_id)

    valid_dtypes = ("int", "Int", "float", "Float", "uint")

    if isinstance(exog, pd.DataFrame):
        unique_dtypes = set(exog.dtypes)
        has_invalid_dtype = False
        for dtype in unique_dtypes:
            if isinstance(dtype, pd.CategoricalDtype):
                try:
                    is_integer = np.issubdtype(dtype.categories.dtype, np.integer)
                except TypeError:
                    # Pandas StringDtype and other non-numpy dtypes will raise TypeError
                    is_integer = False

                if not is_integer:
                    raise TypeError(
                        "Categorical dtypes in exog must contain only integer values. "
                    )
            elif not dtype.name.startswith(valid_dtypes):
                has_invalid_dtype = True

        if has_invalid_dtype:
            warnings.warn(
                f"{series_id} may contain only `int`, `float` or `category` dtypes. "
                f"Most machine learning models do not allow other types of values. "
                f"Fitting the forecaster may fail.",
                DataTypeWarning,
            )

    else:
        dtype_name = str(exog.dtypes)
        if not (dtype_name.startswith(valid_dtypes) or dtype_name == "category"):
            warnings.warn(
                f"{series_id} may contain only `int`, `float` or `category` dtypes. Most "
                f"machine learning models do not allow other types of values. "
                f"Fitting the forecaster may fail.",
                DataTypeWarning,
            )

        if isinstance(exog.dtype, pd.CategoricalDtype):
            if not np.issubdtype(exog.cat.categories.dtype, np.integer):
                raise TypeError(
                    "Categorical dtypes in exog must contain only integer values. "
                )
    return


def get_exog_dtypes(exog: Union[pd.Series, pd.DataFrame]) -> Dict[str, type]:
    """
    Extract and store the data types of exogenous variables.

    This function returns a dictionary mapping column names to their data types.
    For Series, uses the series name as the key. For DataFrames, uses all column names.

    Args:
        exog: Exogenous variable/s (Series or DataFrame).

    Returns:
        Dictionary mapping variable names to their pandas dtypes.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.utils.validation import get_exog_dtypes
        >>>
        >>> # DataFrame with mixed types
        >>> exog_df = pd.DataFrame({
        ...     "temp": pd.Series([20.5, 21.3, 22.1], dtype='float64'),
        ...     "day": pd.Series([1, 2, 3], dtype='int64'),
        ...     "is_weekend": pd.Series([False, False, True], dtype='bool')
        ... })
        >>> dtypes = get_exog_dtypes(exog_df)
        >>> dtypes['temp']
        dtype('float64')
        >>> dtypes['day']
        dtype('int64')
        >>>
        >>> # Series
        >>> exog_series = pd.Series([1.0, 2.0, 3.0], name="temperature", dtype='float64')
        >>> dtypes = get_exog_dtypes(exog_series)
        >>> dtypes
        {'temperature': dtype('float64')}
    """
    if isinstance(exog, pd.Series):
        exog_dtypes = {exog.name: exog.dtypes}
    else:
        exog_dtypes = exog.dtypes.to_dict()

    return exog_dtypes


def check_interval(
    interval: Union[List[float], Tuple[float], None] = None,
    ensure_symmetric_intervals: bool = False,
    quantiles: Union[List[float], Tuple[float], None] = None,
    alpha: Optional[float] = None,
    alpha_literal: Optional[str] = "alpha",
) -> None:
    """
    Validate that a confidence interval specification is valid.

    This function checks that interval values are properly formatted and within
    valid ranges for confidence interval prediction.

    Args:
        interval: Confidence interval percentiles (0-100 inclusive).
            Should be [lower_bound, upper_bound]. Example: [2.5, 97.5] for 95% interval.
        ensure_symmetric_intervals: If True, ensure intervals are symmetric
            (lower + upper = 100).
        quantiles: Sequence of quantiles (0-1 inclusive). Currently not validated,
            reserved for future use.
        alpha: Confidence level (1-alpha). Currently not validated, reserved for future use.
        alpha_literal: Name used in error messages for alpha parameter.

    Raises:
        TypeError: If interval is not a list or tuple.
        ValueError: If interval doesn't have exactly 2 values, values out of range (0-100),
            lower >= upper, or intervals not symmetric when required.

    Examples:
        >>> from spotforecast2_safe.utils.validation import check_interval
        >>>
        >>> # Valid 95% confidence interval
        >>> check_interval(interval=[2.5, 97.5])  # No error
        >>>
        >>> # Valid symmetric interval
        >>> check_interval(interval=[2.5, 97.5], ensure_symmetric_intervals=True)  # No error
        >>>
        >>> # Invalid: not symmetric
        >>> try:
        ...     check_interval(interval=[5, 90], ensure_symmetric_intervals=True)
        ... except ValueError as e:
        ...     print("Error: Interval not symmetric")
        Error: Interval not symmetric
        >>>
        >>> # Invalid: wrong number of values
        >>> try:
        ...     check_interval(interval=[2.5, 50, 97.5])
        ... except ValueError as e:
        ...     print("Error: Must have exactly 2 values")
        Error: Must have exactly 2 values
        >>>
        >>> # Invalid: out of range
        >>> try:
        ...     check_interval(interval=[-5, 105])
        ... except ValueError as e:
        ...     print("Error: Values out of range")
        Error: Values out of range
    """
    if interval is not None:
        if not isinstance(interval, (list, tuple)):
            raise TypeError(
                "`interval` must be a `list` or `tuple`. For example, interval of 95% "
                "should be as `interval = [2.5, 97.5]`."
            )

        if len(interval) != 2:
            raise ValueError(
                "`interval` must contain exactly 2 values, respectively the "
                "lower and upper interval bounds. For example, interval of 95% "
                "should be as `interval = [2.5, 97.5]`."
            )

        if (interval[0] < 0.0) or (interval[0] >= 100.0):
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be >= 0 and < 100."
            )

        if (interval[1] <= 0.0) or (interval[1] > 100.0):
            raise ValueError(
                f"Upper interval bound ({interval[1]}) must be > 0 and <= 100."
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                f"Lower interval bound ({interval[0]}) must be less than the "
                f"upper interval bound ({interval[1]})."
            )

        if ensure_symmetric_intervals and interval[0] + interval[1] != 100:
            raise ValueError(
                f"Interval must be symmetric, the sum of the lower, ({interval[0]}), "
                f"and upper, ({interval[1]}), interval bounds must be equal to "
                f"100. Got {interval[0] + interval[1]}."
            )

    return


def check_predict_input(
    forecaster_name: str,
    steps: Union[int, List[int]],
    is_fitted: bool,
    exog_in_: bool,
    index_type_: type,
    index_freq_: str,
    window_size: int,
    last_window: Optional[Union[pd.Series, pd.DataFrame]],
    last_window_exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    exog: Optional[
        Union[pd.Series, pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]]
    ] = None,
    exog_names_in_: Optional[List[str]] = None,
    interval: Optional[List[float]] = None,
    alpha: Optional[float] = None,
    max_step: Optional[int] = None,
    levels: Optional[Union[str, List[str]]] = None,
    levels_forecaster: Optional[Union[str, List[str]]] = None,
    series_names_in_: Optional[List[str]] = None,
    encoding: Optional[str] = None,
) -> None:
    """
    Check all inputs of predict method. This is a helper function to validate
    that inputs used in predict method match attributes of a forecaster already
    trained.

    Args:
        forecaster_name: str
            Forecaster name.
        steps: int, list
            Number of future steps predicted.
        is_fitted: bool
            Tag to identify if the estimator has been fitted (trained).
        exog_in_: bool
            If the forecaster has been trained using exogenous variable/s.
        index_type_: type
            Type of index of the input used in training.
        index_freq_: str
            Frequency of Index of the input used in training.
        window_size: int
            Size of the window needed to create the predictors. It is equal to
            `max_lag`.
        last_window: pandas Series, pandas DataFrame, None
            Values of the series used to create the predictors (lags) need in the
            first iteration of prediction (t + 1).
        last_window_exog: pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window` in
            ForecasterStats predictions.
        exog: pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        exog_names_in_: list, default None
            Names of the exogenous variables used during training.
        interval: list, tuple, default None
            Confidence of the prediction interval estimated. Sequence of percentiles
            to compute, which must be between 0 and 100 inclusive. For example,
            interval of 95% should be as `interval = [2.5, 97.5]`.
        alpha: float, default None
            The confidence intervals used in ForecasterStats are (1 - alpha) %.
        max_step: int, default None
            Maximum number of steps allowed (`ForecasterDirect` and
            `ForecasterDirectMultiVariate`).
        levels: str, list, default None
            Time series to be predicted (`ForecasterRecursiveMultiSeries`
            and `ForecasterRnn).
        levels_forecaster: str, list, default None
            Time series used as output data of a multiseries problem in a RNN problem
            (`ForecasterRnn`).
        series_names_in_: list, default None
            Names of the columns used during fit (`ForecasterRecursiveMultiSeries`,
            `ForecasterDirectMultiVariate` and `ForecasterRnn`).
        encoding: str, default None
            Encoding used to identify the different series (`ForecasterRecursiveMultiSeries`).

    Returns:
        None
    """

    if not is_fitted:
        raise RuntimeError(
            "This forecaster is not fitted yet. Call `fit` with appropriate "
            "arguments before using `predict`."
        )

    if isinstance(steps, (int, np.integer)) and steps < 1:
        raise ValueError(
            f"`steps` must be an integer greater than or equal to 1. Got {steps}."
        )

    if isinstance(steps, list) and min(steps) < 1:
        raise ValueError(
            f"`steps` must be a list of integers greater than or equal to 1. Got {steps}."
        )

    if max_step is not None:
        if isinstance(steps, (int, np.integer)):
            if steps > max_step:
                raise ValueError(
                    f"The maximum step that can be predicted is {max_step}. "
                    f"Got {steps}."
                )
        elif isinstance(steps, list):
            if max(steps) > max_step:
                raise ValueError(
                    f"The maximum step that can be predicted is {max_step}. "
                    f"Got {max(steps)}."
                )

    if interval is not None or alpha is not None:
        check_interval(interval=interval, alpha=alpha)

    if exog_in_ and exog is None:
        raise ValueError(
            "Forecaster trained with exogenous variable/s. "
            "Same variable/s must be provided when predicting."
        )

    if not exog_in_ and exog is not None:
        raise ValueError(
            "Forecaster trained without exogenous variable/s. "
            "`exog` must be `None` when predicting."
        )

    if exog is not None:
        # If exog is a dictionary, it is assumed that it contains the exogenous
        # variables for each series.
        if isinstance(exog, dict):
            # Check that all series have the exogenous variables
            if levels is None and series_names_in_ is not None:
                levels = series_names_in_

            if isinstance(levels, str):
                levels = [levels]

            if levels is not None:
                for level in levels:
                    if level not in exog:
                        raise ValueError(
                            f"Exogenous variables for series '{level}' are missing."
                        )
                    check_exog(
                        exog=exog[level],
                        allow_nan=False,
                        series_id=f"`exog` for series '{level}'",
                    )
                    check_exog_dtypes(
                        exog=exog[level],
                        call_check_exog=False,
                        series_id=f"`exog` for series '{level}'",
                    )

                    # Check that exogenous variables are the same as used in training
                    # Get the name of columns
                    if isinstance(exog[level], pd.Series):
                        exog_names = [exog[level].name]
                    else:
                        exog_names = exog[level].columns.tolist()

                    col_missing = set(exog_names_in_) - set(exog_names)
                    if col_missing:
                        raise ValueError(
                            f"Missing columns for series '{level}' in `exog`. "
                            f"Expected {exog_names_in_}. Got {exog_names}."
                        )
        else:
            check_exog(exog=exog, allow_nan=False)
            check_exog_dtypes(exog=exog, call_check_exog=False)

            # Check that exogenous variables are the same as used in training
            # Get the name of columns
            if isinstance(exog, pd.Series):
                exog_names = [exog.name]
            else:
                exog_names = exog.columns.tolist()

            col_missing = set(exog_names_in_) - set(exog_names)
            if col_missing:
                raise ValueError(
                    f"Missing columns in `exog`. Expected {exog_names_in_}. "
                    f"Got {exog_names}."
                )

    # Check last_window
    if last_window is not None:
        if isinstance(last_window, pd.DataFrame):
            if last_window.isna().to_numpy().any():
                raise ValueError("`last_window` has missing values.")
        else:
            check_y(last_window, series_id="`last_window`")

    return


def check_residuals_input(
    forecaster_name: str,
    use_in_sample_residuals: bool,
    in_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    out_sample_residuals_: np.ndarray | dict[str, np.ndarray] | None,
    use_binned_residuals: bool,
    in_sample_residuals_by_bin_: (
        dict[str | int, np.ndarray | dict[int, np.ndarray]] | None
    ),
    out_sample_residuals_by_bin_: (
        dict[str | int, np.ndarray | dict[int, np.ndarray]] | None
    ),
    levels: list[str] | None = None,
    encoding: str | None = None,
) -> None:
    """
    Check residuals input arguments in Forecasters.

    Parameters
    ----------
    forecaster_name : str
        Forecaster name.
    use_in_sample_residuals : bool
        Indicates if in sample or out sample residuals are used.
    in_sample_residuals_ : numpy ndarray, dict
        Residuals of the model when predicting training data.
    out_sample_residuals_ : numpy ndarray, dict
        Residuals of the model when predicting non training data.
    use_binned_residuals : bool
        Indicates if residuals are binned.
    in_sample_residuals_by_bin_ : dict
        In sample residuals binned according to the predicted value each residual
        is associated with.
    out_sample_residuals_by_bin_ : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with.
    levels : list, default None
        Names of the series (levels) to be predicted (Forecasters multiseries).
    encoding : str, default None
        Encoding used to identify the different series (ForecasterRecursiveMultiSeries).

    Returns
    -------
    None

    """

    forecasters_multiseries = (
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    )

    if use_in_sample_residuals:
        if use_binned_residuals:
            residuals = in_sample_residuals_by_bin_
            literal = "in_sample_residuals_by_bin_"
        else:
            residuals = in_sample_residuals_
            literal = "in_sample_residuals_"

        # Check if residuals are empty or None
        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`store_in_sample_residuals = True` when fitting the forecaster "
                f"or use the `set_in_sample_residuals()` method before predicting."
            )

        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`, "
                        f"most likely because they were not present in the training data. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels.",
                        UnknownLevelWarning,
                    )
    else:
        if use_binned_residuals:
            residuals = out_sample_residuals_by_bin_
            literal = "out_sample_residuals_by_bin_"
        else:
            residuals = out_sample_residuals_
            literal = "out_sample_residuals_"

        is_empty = (
            residuals is None
            or (isinstance(residuals, dict) and not residuals)
            or (isinstance(residuals, np.ndarray) and residuals.size == 0)
        )
        if is_empty:
            raise ValueError(
                f"`forecaster.{literal}` is either None or empty. Use "
                f"`use_in_sample_residuals = True` or the "
                f"`set_out_sample_residuals()` method before predicting."
            )

        if forecaster_name in forecasters_multiseries:
            if encoding is not None:
                unknown_levels = set(levels) - set(residuals.keys())
                if unknown_levels:
                    warnings.warn(
                        f"`levels` {unknown_levels} are not present in `forecaster.{literal}`. "
                        f"A random sample of the residuals from other levels will be used. "
                        f"This can lead to inaccurate intervals for the unknown levels. "
                        f"Otherwise, Use the `set_out_sample_residuals()` method before "
                        f"predicting to set the residuals for these levels.",
                        UnknownLevelWarning,
                    )

    if forecaster_name in forecasters_multiseries:
        for level in residuals.keys():
            level_residuals = residuals[level]
            if level_residuals is None or len(level_residuals) == 0:
                raise ValueError(
                    f"Residuals for level '{level}' are None. Check `forecaster.{literal}`."
                )

    return


def set_cpu_gpu_device(estimator: object, device: str | None = "cpu") -> str | None:
    """
    Set the device for the estimator to either 'cpu', 'gpu', 'cuda', or None.

    Args:
        estimator: Estimator compatible with the scikit-learn API.
        device: Device to set. Options are 'cpu', 'gpu', 'cuda', or None.
            Defaults to 'cpu'.

    Returns:
        The device that was set on the estimator before the function was called.
    """

    valid_devices = {"gpu", "cpu", "cuda", "GPU", "CPU", None}
    if device not in valid_devices:
        raise ValueError("`device` must be 'gpu', 'cpu', 'cuda', or None.")

    estimator_name = type(estimator).__name__

    supported_estimators = {"XGBRegressor", "LGBMRegressor", "CatBoostRegressor"}
    if estimator_name not in supported_estimators:
        return None

    device_names = {
        "XGBRegressor": "device",
        "LGBMRegressor": "device",
        "CatBoostRegressor": "task_type",
    }
    device_values = {
        "XGBRegressor": {"gpu": "cuda", "cpu": "cpu", "cuda": "cuda"},
        "LGBMRegressor": {"gpu": "gpu", "cpu": "cpu", "cuda": "gpu"},
        "CatBoostRegressor": {
            "gpu": "GPU",
            "cpu": "CPU",
            "cuda": "GPU",
            "GPU": "GPU",
            "CPU": "CPU",
        },
    }

    param_name = device_names[estimator_name]
    original_device = getattr(estimator, param_name, None)

    if device is None:
        return original_device

    new_device = device_values[estimator_name][device]

    if original_device != new_device:
        try:
            estimator.set_params(**{param_name: new_device})
        except Exception as exc:
            warnings.warn(
                f"Failed to set device parameter '{param_name}' to '{new_device}' "
                f"for estimator '{estimator_name}': {exc}",
                UserWarning,
            )

    return original_device
