# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import uuid
import warnings
from copy import copy, deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from spotforecast2_safe.exceptions import (
    IgnoredArgumentWarning,
    InputTypeWarning,
    MissingValuesWarning,
    UnknownLevelWarning,
    set_skforecast_warnings,
)
from spotforecast2_safe.utils import (
    check_exog,
    check_exog_dtypes,
    check_interval,
    check_predict_input,
    check_select_fit_kwargs,
    check_y,
    expand_index,
    get_exog_dtypes,
    initialize_lags,
    initialize_weights,
    input_to_frame,
    transform_dataframe,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None


def check_preprocess_series(
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Index]]:
    """
    Check and preprocess `series` argument in `ForecasterRecursiveMultiSeries` class.

        - If `series` is a wide-format pandas DataFrame, each column represents a
        different time series, and the index must be either a `DatetimeIndex` or
        a `RangeIndex` with frequency or step size, as appropriate
        - If `series` is a long-format pandas DataFrame with a MultiIndex, the
        first level of the index must contain the series IDs, and the second
        level must be a `DatetimeIndex` with the same frequency across all series.
        - If series is a dictionary, each key must be a series ID, and each value
        must be a named pandas Series. All series must have the same index, which
        must be either a `DatetimeIndex` or a `RangeIndex`, and they must share the
        same frequency or step size, as appropriate.

    When `series` is a pandas DataFrame, it is converted to a dictionary of pandas
    Series, where the keys are the series IDs and the values are the Series with
    the same index as the original DataFrame.

    Args:
        series (pd.DataFrame, dict): pandas DataFrame or dictionary of pandas
            Series/DataFrames.

    Returns:
        tuple: A tuple containing:
            - series_dict (dict): Dictionary where keys are series IDs and values
              are pandas Series.
            - series_indexes (dict): Dictionary where keys are series IDs and values
              are the index of each series.
    Raises:
        TypeError:
            If `series` is not a pandas DataFrame or a dictionary of pandas Series/DataFrames.
        TypeError:
            If the index of `series` is not a DatetimeIndex or RangeIndex with frequency/step size.
        ValueError:
            If the series in `series` have different frequencies or step sizes.
        ValueError:
            If all values of any series are NaN.
        UserWarning:
            If `series` is a wide-format DataFrame, only the first column will be used as series values.
        UserWarning:
            If `series` is a DataFrame (either wide or long format), additional internal transformations are required, which can increase computational time.
            It is recommended to use a dictionary of pandas Series instead.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.forecaster.utils import check_preprocess_series
        >>> # Example with wide-format DataFrame
        >>> dates = pd.date_range('2020-01-01', periods=5, freq='D')
        >>> df_wide = pd.DataFrame({
        ...     'series_1': [1, 2, 3, 4, 5],
        ...     'series_2': [5, 4, 3, 2, 1],
        ... }, index=dates)
        >>> series_dict, series_indexes = check_preprocess_series(df_wide)
        UserWarning: `series` DataFrame has multiple columns. Only the values of first column, 'series_1', will be used as series values. All other columns will be ignored.
        UserWarning: Passing a DataFrame (either wide or long format) as `series` requires additional internal transformations, which can increase computational time.
        It is recommended to use a dictionary of pandas Series instead.
        >>> print(series_dict['series_1'])
        2020-01-01    1
        2020-01-02    2
        2020-01-03    3
        2020-01-04    4
        2020-01-05    5
        Name: series_1, dtype: int64
        >>> print(series_indexes['series_1'])
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
                       '2020-01-05'],
                      dtype='datetime64[ns]', freq='D')
        >>> # Example with long-format DataFrame
        >>> df_long = pd.DataFrame({
        ...     'series_id': ['series_1'] * 5 + ['series_2'] * 5,
        ...     'value': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        ... }, index=pd.MultiIndex.from_product([['series_1', 'series_2'], dates], names=['series_id', 'date']))
        >>> series_dict, series_indexes = check_preprocess_series(df_long)
        UserWarning: `series` DataFrame has multiple columns. Only the values of first column, 'value', will be used as series values. All other columns will be ignored.
        UserWarning: Passing a DataFrame (either wide or long format) as `series` requires additional internal transformations, which can increase computational time.
        It is recommended to use a dictionary of pandas Series instead.
        >>> print(series_dict['series_1'])
        2020-01-01    1
        2020-01-02    2
        2020-01-03    3
        2020-01-04    4
        2020-01-05    5
        Name: series_1, dtype: int64
        >>> print(series_indexes['series_1'])
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
                          '2020-01-05'],
                         dtype='datetime64[ns]', freq='D')

        >>> # Example with dictionary of Series
        >>> series_dict_input = {
        ...     'series_1': pd.Series([1, 2, 3, 4, 5], index=dates),
        ...     'series_2': pd.Series([5, 4, 3, 2, 1], index=dates),
        ... }
        >>> series_dict, series_indexes = check_preprocess_series(series_dict_input)
        >>> print(series_dict['series_1'])
        2020-01-01    1
        2020-01-02    2
        2020-01-03    3
        2020-01-04    4
        2020-01-05    5
        Name: series_1, dtype: int64
        >>> print(series_indexes['series_1'])
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
                       '2020-01-05'],
                      dtype='datetime64[ns]', freq='D')
            >>> # Example with dictionary of DataFrames
            >>> df_series_1 = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
            >>> df_series_2 = pd.DataFrame({'value': [5, 4, 3, 2, 1]}, index=dates)
            >>> series_dict_input = {
            ...     'series_1': df_series_1,
            ...     'series_2': df_series_2,
            ... }
            >>> series_dict, series_indexes = check_preprocess_series(series_dict_input)
            >>> print(series_dict['series_1'])
        2020-01-01    1
        2020-01-02    2
        2020-01-03    3
        2020-01-04    4
        2020-01-05    5
        Name: series_1, dtype: int64
        >>> print(series_indexes['series_1'])
        DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
                       '2020-01-05'],
                      dtype='datetime64[ns]', freq='D')
    """
    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
            f"Got {type(series)}."
        )

    if isinstance(series, pd.DataFrame):

        if not isinstance(series.index, pd.MultiIndex):
            _, _ = check_extract_values_and_index(
                data=series, data_label="`series`", return_values=False
            )
            series = series.copy()
            series.index.name = None
            series_dict = series.to_dict(orient="series")
        else:
            if not isinstance(series.index.levels[1], pd.DatetimeIndex):
                raise TypeError(
                    f"The second level of the MultiIndex in `series` must be a "
                    f"pandas DatetimeIndex with the same frequency for each series. "
                    f"Found {type(series.index.levels[1])}."
                )

            first_col = series.columns[0]
            if len(series.columns) != 1:
                warnings.warn(
                    f"`series` DataFrame has multiple columns. Only the values of "
                    f"first column, '{first_col}', will be used as series values. "
                    f"All other columns will be ignored.",
                    IgnoredArgumentWarning,
                )

            series = series.copy()
            series.index = series.index.set_names([series.index.names[0], None])
            series_dict = {
                series_id: series.loc[series_id][first_col].rename(series_id)
                for series_id in series.index.levels[0]
            }

        warnings.warn(
            "Passing a DataFrame (either wide or long format) as `series` requires "
            "additional internal transformations, which can increase computational "
            "time. It is recommended to use a dictionary of pandas Series instead. ",
            InputTypeWarning,
        )

    else:

        not_valid_series = [
            k for k, v in series.items() if not isinstance(v, (pd.Series, pd.DataFrame))
        ]
        if not_valid_series:
            raise TypeError(
                f"If `series` is a dictionary, all series must be a named "
                f"pandas Series or a pandas DataFrame with a single column. "
                f"Review series: {not_valid_series}"
            )

        series_dict = {k: v.copy() for k, v in series.items()}

    not_valid_index = []
    indexes_freq = set()
    series_indexes = {}
    for k, v in series_dict.items():
        if isinstance(v, pd.DataFrame):
            if v.shape[1] != 1:
                raise ValueError(
                    f"If `series` is a dictionary, all series must be a named "
                    f"pandas Series or a pandas DataFrame with a single column. "
                    f"Review series: '{k}'"
                )
            series_dict[k] = v.iloc[:, 0]

        series_dict[k].name = k
        idx = v.index
        if isinstance(idx, pd.DatetimeIndex):
            indexes_freq.add(idx.freq)
        elif isinstance(idx, pd.RangeIndex):
            indexes_freq.add(idx.step)
        else:
            not_valid_index.append(k)

        if v.isna().to_numpy().all():
            raise ValueError(f"All values of series '{k}' are NaN.")

        series_indexes[k] = idx

    if not_valid_index:
        raise TypeError(
            f"If `series` is a dictionary, all series must have a Pandas "
            f"RangeIndex or DatetimeIndex with the same step/frequency. "
            f"Review series: {not_valid_index}"
        )
    if None in indexes_freq:
        raise ValueError(
            "If `series` is a dictionary, all series must have a Pandas "
            "RangeIndex or DatetimeIndex with the same step/frequency. "
            "If it a MultiIndex DataFrame, the second level must be a DatetimeIndex "
            "with the same frequency for each series. Found series with no "
            "frequency or step."
        )
    if not len(indexes_freq) == 1:
        raise ValueError(
            f"If `series` is a dictionary, all series must have a Pandas "
            f"RangeIndex or DatetimeIndex with the same step/frequency. "
            f"If it a MultiIndex DataFrame, the second level must be a DatetimeIndex "
            f"with the same frequency for each series. "
            f"Found frequencies: {sorted(indexes_freq)}"
        )

    return series_dict, series_indexes


def check_preprocess_exog_multiseries(exog):
    pass


def exog_to_direct(
    exog: pd.Series | pd.DataFrame, steps: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms `exog` to a pandas DataFrame with the shape needed for Direct
    forecasting.

    Args:
        exog (pd.Series, pd.DataFrame): Exogenous variables.
        steps (int): Number of steps that will be predicted using exog.

    Returns:
        tuple: A tuple containing:
            - exog_direct (pd.DataFrame): Exogenous variables transformed.
            - exog_direct_names (list): Names of the columns of the exogenous variables transformed.
    """

    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}."
        )

    if isinstance(exog, pd.Series):
        exog = exog.to_frame()

    n_rows = len(exog)
    exog_idx = exog.index
    exog_cols = exog.columns
    exog_direct = []
    for i in range(steps):
        exog_step = exog.iloc[i : n_rows - (steps - 1 - i),]
        exog_step.index = pd.RangeIndex(len(exog_step))
        exog_step.columns = [f"{col}_step_{i + 1}" for col in exog_cols]
        exog_direct.append(exog_step)

    exog_direct = pd.concat(exog_direct, axis=1) if steps > 1 else exog_direct[0]

    exog_direct_names = exog_direct.columns.to_list()
    exog_direct.index = exog_idx[-len(exog_direct) :]

    return exog_direct, exog_direct_names


def exog_to_direct_numpy(
    exog: np.ndarray | pd.Series | pd.DataFrame, steps: int
) -> tuple[np.ndarray, list[str] | None]:
    """
    Transforms `exog` to numpy ndarray with the shape needed for Direct
    forecasting.

    Args:
        exog (np.ndarray, pd.Series, pd.DataFrame): Exogenous variables,
            shape(samples,). If exog is a pandas format, the direct exog names
            are created.
        steps (int): Number of steps that will be predicted using exog.

    Returns:
        tuple: A tuple containing:
            - exog_direct (np.ndarray): Exogenous variables transformed.
            - exog_direct_names (list, None): Names of the columns of the exogenous
              variables transformed. Only created if `exog` is a pandas format.

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import exog_to_direct_numpy
        >>> import numpy as np
        >>> exog = np.array([10, 20, 30, 40, 50])
        >>> steps = 3
        >>> exog_direct, exog_direct_names = exog_to_direct_numpy(exog, steps)
        >>> print(exog_direct)
        [[10 20 30]
         [20 30 40]
         [30 40 50]]
        >>> print(exog_direct_names)
        None
    """

    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog_cols = exog.columns if isinstance(exog, pd.DataFrame) else [exog.name]
        exog_direct_names = [
            f"{col}_step_{i + 1}" for i in range(steps) for col in exog_cols
        ]
        exog = exog.to_numpy()
    else:
        exog_direct_names = None
        if not isinstance(exog, np.ndarray):
            raise TypeError(
                f"`exog` must be a numpy ndarray, pandas Series or DataFrame. "
                f"Got {type(exog)}."
            )

    if exog.ndim == 1:
        exog = np.expand_dims(exog, axis=1)

    n_rows = len(exog)
    exog_direct = [exog[i : n_rows - (steps - 1 - i)] for i in range(steps)]
    exog_direct = np.concatenate(exog_direct, axis=1) if steps > 1 else exog_direct[0]

    return exog_direct, exog_direct_names


def prepare_steps_direct(
    max_step: int | list[int] | np.ndarray, steps: int | list[int] | None = None
) -> list[int]:
    """
    Prepare list of steps to be predicted in Direct Forecasters.

    Args:
        max_step (int, list, np.ndarray): Maximum number of future steps the
            forecaster will predict.
        steps (int, list, None): Predict n steps. The value of `steps` must be
            less than or equal to the value of steps defined when initializing
            the forecaster. Starts at 1. Defaults to `None`.

    Returns:
        list: Steps to be predicted.

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import prepare_steps_direct
        >>> max_step = 5
        >>> steps = 3
        >>> prepare_steps_direct(max_step, steps)
        [1, 2, 3]

        >>> max_step = 5
        >>> steps = [1, 3, 5]
        >>> prepare_steps_direct(max_step, steps)
        [1, 3, 5]

        >>> max_step = 5
        >>> steps = None
        >>> prepare_steps_direct(max_step, steps)
        [1, 2, 3, 4, 5]
    """

    if isinstance(steps, int):
        steps_direct = list(range(1, steps + 1))
    elif steps is None:
        if isinstance(max_step, int):
            steps_direct = list(range(1, max_step + 1))
        else:
            steps_direct = [int(s) for s in max_step]
    elif isinstance(steps, list):
        steps_direct = []
        for step in steps:
            if not isinstance(step, (int, np.integer)):
                raise TypeError(
                    f"`steps` argument must be an int, a list of ints or `None`. "
                    f"Got {type(steps)}."
                )
            steps_direct.append(int(step))

    return steps_direct


def transform_numpy(
    array: np.ndarray,
    transformer: object | None,
    fit: bool = False,
    inverse_transform: bool = False,
) -> np.ndarray:
    """
    Transform raw values of a numpy ndarray with a scikit-learn alike
    transformer, preprocessor or ColumnTransformer. The transformer used must
    have the following methods: fit, transform, fit_transform and
    inverse_transform. ColumnTransformers are not allowed since they do not
    have inverse_transform method.

    Args:
        array (np.ndarray): Array to be transformed.
        transformer (object, None): Scikit-learn alike transformer, preprocessor,
            or ColumnTransformer with methods: fit, transform, fit_transform
            and inverse_transform.
        fit (bool): Train the transformer before applying it. Defaults to `False`.
        inverse_transform (bool): Transform back the data to the original
            representation. This is not available when using transformers of
            class scikit-learn ColumnTransformers. Defaults to `False`.

    Returns:
        np.ndarray: Transformed array.

    Raises:
        TypeError: If `array` is not a numpy ndarray.
        TypeError: If `transformer` is not a scikit-learn alike transformer.
        ValueError: If `inverse_transform` is True and `transformer` is a
            ColumnTransformer.

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import transform_numpy
        >>> from sklearn.preprocessing import StandardScaler
        >>> import numpy as np
        >>> array = np.array([1, 2, 3, 4, 5])
        >>> transformer = StandardScaler()
        >>> array_transformed = transform_numpy(array, transformer, fit=True)
        >>> print(array_transformed)
        [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
        >>> array_inversed = transform_numpy(array_transformed, transformer, inverse_transform=True)
        >>> print(array_inversed)
        [1. 2. 3. 4. 5.]
    """

    if transformer is None:
        return array

    if not isinstance(array, np.ndarray):
        raise TypeError(f"`array` argument must be a numpy ndarray. Got {type(array)}")

    original_ndim = array.ndim
    original_shape = array.shape
    reshaped_for_inverse = False

    if original_ndim == 1:
        array = array.reshape(-1, 1)

    if inverse_transform and isinstance(transformer, ColumnTransformer):
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        if not inverse_transform:
            if fit:
                array_transformed = transformer.fit_transform(array)
            else:
                array_transformed = transformer.transform(array)
        else:
            # Vectorized inverse transformation for 2D arrays with multiple columns.
            # Reshape to single column, transform, and reshape back.
            # This is faster than applying the transformer column by column.
            if array.shape[1] > 1:
                array = array.reshape(-1, 1)
                reshaped_for_inverse = True
            array_transformed = transformer.inverse_transform(array)

    if hasattr(array_transformed, "toarray"):
        # If the returned values are in sparse matrix format, it is converted to dense
        array_transformed = array_transformed.toarray()

    if isinstance(array_transformed, (pd.Series, pd.DataFrame)):
        array_transformed = array_transformed.to_numpy()

    # Reshape back to original shape only if we reshaped for inverse_transform
    if reshaped_for_inverse:
        array_transformed = array_transformed.reshape(original_shape)

    if original_ndim == 1:
        array_transformed = array_transformed.ravel()

    return array_transformed


def select_n_jobs_fit_forecaster(forecaster_name: str, estimator: object) -> int:
    """Select the number of jobs to run in parallel during the fit process.

    This function determines the optimal number of parallel processes for fitting
    the forecaster based on the available system resources. In safety-critical
    environments, this helps manage computational load and ensures system
    predictability.

    Args:
        forecaster_name (str): Name of the forecaster being fitted.
        estimator (object): The estimator object being used by the forecaster.

    Returns:
        int: The number of jobs (CPUs) to use for parallel processing.
    """
    import os

    return os.cpu_count() or 1


__all__ = [
    "initialize_lags",
    "initialize_weights",
    "check_select_fit_kwargs",
    "check_y",
    "check_exog",
    "get_exog_dtypes",
    "check_exog_dtypes",
    "check_predict_input",
    "check_interval",
    "input_to_frame",
    "expand_index",
    "transform_dataframe",
    "check_preprocess_series",
    "check_preprocess_exog_multiseries",
    "set_skforecast_warnings",
    "set_cpu_gpu_device",
    "initialize_window_features",
    "initialize_transformer_series",
    "check_extract_values_and_index",
    "get_style_repr_html",
    "initialize_estimator",
    "check_residuals_input",
    "date_to_index_position",
    "prepare_steps_direct",
    "exog_to_direct",
    "exog_to_direct_numpy",
    "transform_numpy",
    "select_n_jobs_fit_forecaster",
    "select_n_jobs_fit_forecaster",
    "predict_multivariate",
    "align_series_and_exog_multiseries",
    "prepare_levels_multiseries",
    "preprocess_levels_self_last_window_multiseries",
    "initialize_differentiator_multiseries",
]


def initialize_window_features(
    window_features: Any,
) -> Tuple[Optional[List[object]], Optional[List[str]], Optional[int]]:
    """Check window_features argument input and generate the corresponding list.

    This function validates window feature objects and extracts their metadata,
    ensuring they have the required attributes (window_sizes, features_names) and
    methods (transform_batch, transform) for proper forecasting operations.

    Args:
        window_features: Classes used to create window features. Can be a single
            object or a list of objects. Each object must have `window_sizes`,
            `features_names` attributes and `transform_batch`, `transform` methods.

    Returns:
        tuple: A tuple containing:
            - window_features (list or None): List of classes used to create window features.
            - window_features_names (list or None): List with all the features names of the window features.
            - max_size_window_features (int or None): Maximum value of the `window_sizes` attribute of all classes.

    Raises:
        ValueError: If `window_features` is an empty list.
        ValueError: If a window feature is missing required attributes or methods.
        TypeError: If `window_sizes` or `features_names` have incorrect types.

    Examples:
        >>> from spotforecast2_safe.forecaster.preprocessing import RollingFeatures
        >>> wf = RollingFeatures(stats=['mean', 'std'], window_sizes=[7, 14])
        >>> wf_list, names, max_size = initialize_window_features(wf)
        >>> print(f"Max window size: {max_size}")
        Max window size: 14
        >>> print(f"Number of features: {len(names)}")
        Number of features: 4

        Multiple window features:
        >>> class MockWF:
        ...     def __init__(self, names, sizes):
        ...         self.features_names = names
        ...         self.window_sizes = sizes
        ...     def transform_batch(self, X): pass
        ...     def transform(self, X): pass
        >>> wf1 = MockWF(['f1'], 7)
        >>> wf2 = MockWF(['f2', 'f3'], 3)
        >>> wf_list, names, max_size = initialize_window_features([wf1, wf2])
        >>> print(f"Max window size: {max_size}")
        Max window size: 7
    """

    needed_atts = ["window_sizes", "features_names"]
    needed_methods = ["transform_batch", "transform"]

    max_window_sizes = None
    window_features_names = None
    max_size_window_features = None
    if window_features is not None:
        if isinstance(window_features, list) and len(window_features) < 1:
            raise ValueError(
                "Argument `window_features` must contain at least one element."
            )
        if not isinstance(window_features, list):
            window_features = [window_features]

        link_to_docs = (
            "\nVisit the documentation for more information about how to create "
            "custom window features."
        )

        max_window_sizes = []
        window_features_names = []
        needed_atts_set = set(needed_atts)
        needed_methods_set = set(needed_methods)
        for wf in window_features:
            wf_name = type(wf).__name__
            atts_methods = set(dir(wf))
            if not needed_atts_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the attributes: {needed_atts}." + link_to_docs
                )
            if not needed_methods_set.issubset(atts_methods):
                raise ValueError(
                    f"{wf_name} must have the methods: {needed_methods}." + link_to_docs
                )

            window_sizes = wf.window_sizes
            if not isinstance(window_sizes, (int, list)):
                raise TypeError(
                    f"Attribute `window_sizes` of {wf_name} must be an int or a list "
                    f"of ints. Got {type(window_sizes)}." + link_to_docs
                )

            if isinstance(window_sizes, int):
                if window_sizes < 1:
                    raise ValueError(
                        f"If argument `window_sizes` is an integer, it must be equal to or "
                        f"greater than 1. Got {window_sizes} from {wf_name}."
                        + link_to_docs
                    )
                max_window_sizes.append(window_sizes)
            else:
                if not all(isinstance(ws, int) for ws in window_sizes) or not all(
                    ws >= 1 for ws in window_sizes
                ):
                    raise ValueError(
                        f"If argument `window_sizes` is a list, all elements must be integers "
                        f"equal to or greater than 1. Got {window_sizes} from {wf_name}."
                        + link_to_docs
                    )
                max_window_sizes.append(max(window_sizes))

            features_names = wf.features_names
            if not isinstance(features_names, (str, list)):
                raise TypeError(
                    f"Attribute `features_names` of {wf_name} must be a str or "
                    f"a list of strings. Got {type(features_names)}." + link_to_docs
                )
            if isinstance(features_names, str):
                window_features_names.append(features_names)
            else:
                if not all(isinstance(fn, str) for fn in features_names):
                    raise TypeError(
                        f"If argument `features_names` is a list, all elements "
                        f"must be strings. Got {features_names} from {wf_name}."
                        + link_to_docs
                    )
                window_features_names.extend(features_names)

        max_size_window_features = max(max_window_sizes)
        if len(set(window_features_names)) != len(window_features_names):
            raise ValueError(
                f"All window features names must be unique. Got {window_features_names}."
            )

    return window_features, window_features_names, max_size_window_features


def check_extract_values_and_index(
    data: Union[pd.Series, pd.DataFrame],
    data_label: str = "`y`",
    ignore_freq: bool = False,
    return_values: bool = True,
) -> Tuple[Optional[np.ndarray], pd.Index]:
    """Extract values and index from a pandas Series or DataFrame, ensuring they are valid.

    Validates that the input data has a proper DatetimeIndex or RangeIndex and extracts
    its values and index for use in forecasting operations. Optionally checks for
    index frequency consistency.

    Args:
        data: Input data (pandas Series or DataFrame) to extract values and index from.
        data_label: Label used in exception messages for better error reporting.
            Defaults to "`y`".
        ignore_freq: If True, the frequency of the index is not checked.
            Defaults to False.
        return_values: If True, the values of the data are returned.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - values (numpy.ndarray or None): Values of the data as numpy array,
              or None if return_values is False.
            - index (pandas.Index): Index of the data.

    Raises:
        TypeError: If data is not a pandas Series or DataFrame.
        TypeError: If data index is not a DatetimeIndex or RangeIndex.

    Warnings:
        UserWarning: If DatetimeIndex has no frequency (inferred automatically).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> series = pd.Series(np.arange(10), index=dates)
        >>> values, index = check_extract_values_and_index(series)
        >>> print(values.shape)
        (10,)
        >>> print(type(index))
        <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

        Extract index only:
        >>> _, index = check_extract_values_and_index(series, return_values=False)
        >>> print(index[0])
        2020-01-01 00:00:00
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{data_label} must be a pandas Series or DataFrame.")

    if not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex)):
        raise TypeError(f"{data_label} must have a pandas DatetimeIndex or RangeIndex.")

    if isinstance(data.index, pd.DatetimeIndex) and not ignore_freq:
        if data.index.freq is None:
            warnings.warn(
                f"{data_label} has a DatetimeIndex but no frequency. "
                "The frequency has been inferred from the index.",
                UserWarning,
            )

    values = data.to_numpy() if return_values else None

    return values, data.index


def get_style_repr_html(is_fitted: bool = False) -> Tuple[str, str]:
    """Generate CSS style for HTML representation of the Forecaster.

    Creates a unique CSS style block with a container ID for rendering
    forecaster objects in Jupyter notebooks or HTML documents. The styling
    provides a clean, monospace display with a light gray background.

    Args:
        is_fitted: Parameter to indicate if the Forecaster has been fitted.
            Currently not used in styling but reserved for future extensions.

    Returns:
        tuple: A tuple containing:
            - style (str): CSS style block as a string with unique container class.
            - unique_id (str): Unique 8-character ID for the container element.

    Examples:
        >>> style, uid = get_style_repr_html(is_fitted=True)
        >>> print(f"Container ID: {uid}")
        Container ID: a1b2c3d4
        >>> print(f"Style contains CSS: {'container-' in style}")
        Style contains CSS: True

        Using in HTML rendering:
        >>> style, uid = get_style_repr_html(is_fitted=False)
        >>> html = f"{style}<div class='container-{uid}'>Forecaster Info</div>"
        >>> print("background-color" in html)
        True
    """

    unique_id = str(uuid.uuid4())[:8]
    style = f"""
    <style>
        .container-{unique_id} {{
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
    """
    return style, unique_id


def align_series_and_exog_multiseries(
    series_dict: dict[str, pd.Series],
    exog_dict: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame | None]]:
    """
    Align series and exog according to their index.

    Heading and trailing NaNs are removed from all series in `series_dict`.
    If needed, reindexing is applied to `exog_dict`.

    Args:
        series_dict (dict): Dictionary with the series used during training.
        exog_dict (dict, None): Dictionary with the exogenous variable/s used
            during training. Defaults to `None`.

    Returns:
        tuple: A tuple containing:
            - series_dict (dict): Dictionary with the aligned series.
            - exog_dict (dict): Dictionary with the aligned exogenous variables.
    """

    for k in series_dict.keys():
        if np.isnan(series_dict[k].iat[0]) or np.isnan(series_dict[k].iat[-1]):
            first_valid_index = series_dict[k].first_valid_index()
            last_valid_index = series_dict[k].last_valid_index()
            series_dict[k] = series_dict[k].loc[first_valid_index:last_valid_index]
        else:
            first_valid_index = series_dict[k].index[0]
            last_valid_index = series_dict[k].index[-1]

        if exog_dict is not None and exog_dict[k] is not None:
            if not series_dict[k].index.equals(exog_dict[k].index):
                exog_dict[k] = exog_dict[k].loc[first_valid_index:last_valid_index]
                if exog_dict[k].empty:
                    warnings.warn(
                        f"`exog` for series '{k}' is empty after aligning "
                        f"with the series index. Exog values will be NaN.",
                        MissingValuesWarning,
                    )
                    exog_dict[k] = None
                elif len(exog_dict[k]) != len(series_dict[k]):
                    warnings.warn(
                        f"`exog` for series '{k}' doesn't have values for "
                        f"all the dates in the series. Missing values will be "
                        f"filled with NaN.",
                        MissingValuesWarning,
                    )
                    exog_dict[k] = exog_dict[k].reindex(
                        series_dict[k].index, fill_value=np.nan
                    )

    return series_dict, exog_dict


def prepare_levels_multiseries(
    X_train_series_names_in_: list[str], levels: str | list[str] | None = None
) -> tuple[list[str], bool]:
    """
    Prepare list of levels to be predicted in multiseries Forecasters.

    Args:
        X_train_series_names_in_ (list): Names of the series (levels) included
            in the matrix `X_train`.
        levels (str, list, None): Names of the series (levels) to be predicted.
            Defaults to `None`.

    Returns:
        tuple: A tuple containing:
            - levels (list): Names of the series (levels) to be predicted.
            - input_levels_is_list (bool): Indicates if input levels argument is a list.
    """

    input_levels_is_list = False
    if levels is None:
        levels = X_train_series_names_in_
    elif isinstance(levels, str):
        levels = [levels]
    else:
        input_levels_is_list = True

    return levels, input_levels_is_list


def preprocess_levels_self_last_window_multiseries(
    levels: list[str],
    input_levels_is_list: bool,
    last_window_: dict[str, pd.Series],
) -> tuple[list[str], pd.DataFrame]:
    """
    Preprocess `levels` and `last_window` arguments for prediction.

    Only levels whose last window ends at the same datetime index will be
    predicted together.

    Args:
        levels (list): Names of the series (levels) to be predicted.
        input_levels_is_list (bool): Indicates if input levels argument is a list.
        last_window_ (dict): Dictionary with the last window of each series.

    Returns:
        tuple: A tuple containing:
            - levels (list): Names of the series (levels) to be predicted.
            - last_window (pd.DataFrame): Series values used to create predictors.
    """

    available_last_windows = set() if last_window_ is None else set(last_window_.keys())
    not_available_last_window = set(levels) - available_last_windows
    if not_available_last_window:
        levels = [level for level in levels if level not in not_available_last_window]
        if not levels:
            raise ValueError(
                f"No series to predict. None of the series {not_available_last_window} "
                f"are present in `last_window_` attribute. Provide `last_window` "
                f"as argument in predict method."
            )
        else:
            warnings.warn(
                f"Levels {not_available_last_window} are excluded from "
                f"prediction since they were not stored in `last_window_` "
                f"attribute during training. If you don't want to retrain "
                f"the Forecaster, provide `last_window` as argument.",
                IgnoredArgumentWarning,
            )

    last_index_levels = [v.index[-1] for k, v in last_window_.items() if k in levels]
    if len(set(last_index_levels)) > 1:
        max_index_levels = max(last_index_levels)
        selected_levels = [
            k
            for k, v in last_window_.items()
            if k in levels and v.index[-1] == max_index_levels
        ]

        series_excluded_from_last_window = set(levels) - set(selected_levels)
        levels = selected_levels

        if input_levels_is_list and series_excluded_from_last_window:
            warnings.warn(
                f"Only series whose last window ends at the same index "
                f"can be predicted together. Series that do not reach "
                f"the maximum index, '{max_index_levels}', are excluded "
                f"from prediction: {series_excluded_from_last_window}.",
                IgnoredArgumentWarning,
            )

    last_window = pd.DataFrame({k: v for k, v in last_window_.items() if k in levels})

    return levels, last_window


def initialize_differentiator_multiseries(
    series_names_in_: list[str],
    differentiator: object | dict[str, object | None] | None = None,
) -> dict[str, object | None]:
    """
    Initialize `differentiator_` attribute for multiseries forecasters.

    Args:
        series_names_in_ (list): Names of the series (levels) used during training.
        differentiator (object, dict, None): Skforecast object (or dict of objects)
            used to differentiate the time series. Defaults to `None`.

    Returns:
        dict: Dictionary with the `differentiator` for each series.
    """

    series_names_in_ = series_names_in_ + ["_unknown_level"]
    if differentiator is None:
        differentiator_ = {serie: None for serie in series_names_in_}
    elif not isinstance(differentiator, dict):
        differentiator_ = {serie: copy(differentiator) for serie in series_names_in_}
    else:
        differentiator_ = {serie: None for serie in series_names_in_}
        # Only elements already present in differentiator_ are updated
        differentiator_.update(
            {k: deepcopy(v) for k, v in differentiator.items() if k in differentiator_}
        )

        series_not_in_differentiator = set(series_names_in_) - set(
            differentiator.keys()
        )
        if series_not_in_differentiator:
            warnings.warn(
                f"{series_not_in_differentiator} not present in `differentiation`."
                f" No differentiation is applied to these series.",
                IgnoredArgumentWarning,
            )

    return differentiator_


def set_cpu_gpu_device(
    estimator: object,
    device: str | None = "cpu",
) -> str | None:
    """
    Set the device for the estimator.

    Args:
        estimator (object): Estimator object.
        device (str, None): Device to use. One of 'cpu', 'gpu', 'cuda', or None.
            Defaults to 'cpu'.

    Returns:
        str, None: The original device of the estimator.
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

    try:
        new_device = device_values[estimator_name][device]
    except KeyError:
        # Fallback if device mapping fails or model updated
        return original_device

    if original_device != new_device:
        try:
            estimator.set_params(**{param_name: new_device})
        except Exception as exc:
            warnings.warn(
                f"Failed to set device '{new_device}' for estimator "
                f"'{estimator_name}' using parameter '{param_name}': {exc}",
                RuntimeWarning,
            )

    return original_device


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

    Args:
        forecaster_name (str): Forecaster name.
        use_in_sample_residuals (bool): Indicates if in sample or out sample
            residuals are used.
        in_sample_residuals_ (np.ndarray, dict, None): Residuals of the model
            when predicting training data.
        out_sample_residuals_ (np.ndarray, dict, None): Residuals of the model
            when predicting non training data.
        use_binned_residuals (bool): Indicates if residuals are binned.
        in_sample_residuals_by_bin_ (dict, None): In sample residuals binned
            according to the predicted value.
        out_sample_residuals_by_bin_ (dict, None): Out of sample residuals
            binned according to the predicted value.
        levels (list, None): Names of the series (levels) to be predicted.
            Defaults to `None`.
        encoding (str, None): Encoding used to identify the different series.
            Defaults to `None`.

    Returns:
        None

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import check_residuals_input
        >>> import numpy as np
        >>> forecaster_name = "ForecasterRecursiveMultiSeries"
        >>> use_in_sample_residuals = True
        >>> in_sample_residuals_ = {'series_1': np.array([0.1, -0.2]), 'series_2': np.array([0.3, -0.1])}
        >>> out_sample_residuals_ = None
        >>> use_binned_residuals = False
        >>> check_residuals_input(
        ...     forecaster_name,
        ...     use_in_sample_residuals,
        ...     in_sample_residuals_,
        ...     out_sample_residuals_,
        ...     use_binned_residuals,
        ...     in_sample_residuals_by_bin_=None,
        ...     out_sample_residuals_by_bin_=None,
        ...     levels=['series_1', 'series_2'],
        ...     encoding='onehot'
        ... )
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
                f"`set_out_sample_residuals()` method before predicting."
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


def date_to_index_position(
    index: pd.Index,
    date_input: int | str | pd.Timestamp,
    method: str = "prediction",
    date_literal: str = "steps",
    kwargs_pd_to_datetime: dict = {},
) -> int:
    """
    Transform a datetime string or pandas Timestamp to an integer position.

    The integer represents the position of the datetime in the index.

    Args:
        index (pd.Index): Original datetime index.
        date_input (int, str, pd.Timestamp): Datetime to transform.
        method (str): Strategy to use. Options: 'prediction', 'validation'.
            Defaults to `'prediction'`.
        date_literal (str): Variable name used in error messages.
            Defaults to `'steps'`.
        kwargs_pd_to_datetime (dict): Keyword arguments for `pd.to_datetime()`.
            Defaults to `{}`.

    Returns:
        int: `date_input` transformed to integer position in the `index`.

    Raises:
        ValueError: If `method` is not 'prediction' or 'validation'.
        TypeError: If `date_input` is not an int, str, or pandas Timestamp.
        TypeError: If `index` is not a pandas DatetimeIndex when `date_input` is not an int.
        ValueError: If `date_input` is a date and does not meet requirement.

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import date_to_index_position
        >>> import pandas as pd
        >>> index = pd.date_range(start='2020-01-01', periods=10, freq='D')
        >>> # Using an integer input
        >>> date_to_index_position(index, 5)
        5
        >>> # Using a date input for prediction
        >>> date_to_index_position(index, '2020-01-15', method='prediction')
        5
        >>> # Using a date input for validation
        >>> date_to_index_position(index, '2020-01-05', method='validation')
        5
    """

    if method not in ["prediction", "validation"]:
        raise ValueError("`method` must be 'prediction' or 'validation'.")

    # Initialize output; will be set in all valid code paths below
    output: int = 0

    if isinstance(date_input, (str, pd.Timestamp)):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                f"Index must be a pandas DatetimeIndex when `{date_literal}` is "
                f"not an integer. Check input series or last window."
            )

        target_date = pd.to_datetime(date_input, **kwargs_pd_to_datetime)
        last_date = pd.to_datetime(index[-1])

        if method == "prediction":
            if target_date <= last_date:
                raise ValueError(
                    "If `steps` is a date, it must be greater than the last date "
                    "in the index."
                )
            span_index = pd.date_range(
                start=last_date, end=target_date, freq=index.freq
            )
            output = len(span_index) - 1
        elif method == "validation":
            first_date = pd.to_datetime(index[0])
            if target_date < first_date or target_date > last_date:
                raise ValueError(
                    "If `initial_train_size` is a date, it must be greater than "
                    "the first date in the index and less than the last date."
                )
            span_index = pd.date_range(
                start=first_date, end=target_date, freq=index.freq
            )
            output = len(span_index)

    elif isinstance(date_input, (int, np.integer)):
        output = date_input

    else:
        raise TypeError(
            f"`{date_literal}` must be an integer, string, or pandas Timestamp."
        )

    return output


def initialize_estimator(
    estimator: object | None = None, regressor: object | None = None
) -> None:
    """
    Handle the deprecation of 'regressor' in favor of 'estimator'.

    Args:
        estimator (object, None): Estimator or pipeline compatible with the
            scikit-learn API. Defaults to `None`.
        regressor (object, None): Deprecated. Alias for `estimator`.
            Defaults to `None`.

    Returns:
        object: The valid estimator object.

    Raises:
        ValueError: If both `estimator` and `regressor` are provided.

    Examples:
        >>> from spotforecast2_safe.forecaster.utils import initialize_estimator
        >>> from sklearn.linear_model import LinearRegression
        >>> # Using the `estimator` argument
        >>> estimator = LinearRegression()
        >>> initialize_estimator(estimator=estimator)
        LinearRegression()
        >>> # Using the deprecated `regressor` argument
        >>> regressor = LinearRegression()
        >>> initialize_estimator(regressor=regressor)
        LinearRegression()

    """

    if regressor is not None:
        warnings.warn(
            "The `regressor` argument is deprecated and will be removed in a future "
            "version. Please use `estimator` instead.",
            FutureWarning,
            stacklevel=3,  # Important: to point to the user's code
        )
        if estimator is not None:
            raise ValueError(
                "Both `estimator` and `regressor` were provided. Use only `estimator`."
            )
        return regressor

    return estimator


def predict_multivariate(
    forecasters: dict[str, Any],
    steps_ahead: int,
    exog: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Generate multi-output predictions using multiple baseline forecasters.

    Args:
        forecasters (dict): Dictionary of fitted forecaster instances (one per target).
            Keys are target names, values are the fitted forecasters (e.g.,
            ForecasterRecursive, ForecasterEquivalentDate).
        steps_ahead (int): Number of steps to forecast.
        exog (pd.DataFrame, optional): Exogenous variables for prediction.
            If provided, will be passed to each forecaster's predict method.
        show_progress (bool, optional): Show progress bar while predicting
            per target forecaster. Default: False.

    Returns:
        pd.DataFrame: DataFrame with predictions for all targets.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2_safe.forecaster.utils import predict_multivariate
        >>> y1 = pd.Series([1, 2, 3, 4, 5])
        >>> y2 = pd.Series([2, 4, 6, 8, 10])
        >>> f1 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
        >>> f2 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
        >>> f1.fit(y=y1)
        >>> f2.fit(y=y2)
        >>> forecasters = {'target1': f1, 'target2': f2}
        >>> predictions = predict_multivariate(forecasters, steps_ahead=2)
        >>> predictions
           target1  target2
        5      6.0     12.0
        6      7.0     14.0
    """

    if not forecasters:
        return pd.DataFrame()

    predictions = {}

    target_iter = forecasters.items()
    if show_progress and tqdm is not None:
        target_iter = tqdm(
            forecasters.items(),
            desc="Predicting targets",
            unit="model",
        )

    for target, forecaster in target_iter:
        # Generate predictions for this target
        if exog is not None:
            pred = forecaster.predict(steps=steps_ahead, exog=exog)
        else:
            pred = forecaster.predict(steps=steps_ahead)
        predictions[target] = pred

    # Combine into a single DataFrame
    return pd.concat(predictions, axis=1)


def initialize_transformer_series(
    forecaster_name: str,
    series_names_in_: list[str],
    encoding: str | None = None,
    transformer_series: object | dict[str, object | None] | None = None,
) -> dict[str, object | None]:
    """Initialize transformer_series_ attribute for multivariate/multiseries forecasters.

    Creates a dictionary of transformers for each time series in multivariate or
    multiseries forecasting. Handles three cases: no transformation (None), same
    transformer for all series (single object), or different transformers per series
    (dictionary). Clones transformer objects to avoid overwriting.

    Args:
        forecaster_name: Name of the forecaster using this function. Special handling
            is applied for 'ForecasterRecursiveMultiSeries'.
        series_names_in_: Names of the time series (levels) used during training.
            These will be the keys in the returned transformer dictionary.
        encoding: Encoding used to identify different series. Only used for
            ForecasterRecursiveMultiSeries. If None, creates a single '_unknown_level'
            entry. Defaults to None.
        transformer_series: Transformer(s) to apply to series. Can be:
            - None: No transformation applied
            - Single transformer object: Same transformer cloned for all series
            - Dict mapping series names to transformers: Different transformer per series
            Defaults to None.

    Returns:
        dict: Dictionary with series names as keys and transformer objects (or None)
            as values. Transformers are cloned to prevent overwriting.

    Warnings:
        IgnoredArgumentWarning: If transformer_series is a dict and some series_names_in_
            are not present in the dict keys (those series get no transformation).

    Examples:
        No transformation:
        >>> from spotforecast2_safe.forecaster.utils import initialize_transformer_series
        >>> series = ['series1', 'series2', 'series3']
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=series,
        ...     transformer_series=None
        ... )
        >>> print(result)
        {'series1': None, 'series2': None, 'series3': None}

        Same transformer for all series:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=['series1', 'series2'],
        ...     transformer_series=scaler
        ... )
        >>> len(result)
        2
        >>> all(isinstance(v, StandardScaler) for v in result.values())
        True
        >>> result['series1'] is result['series2']  # Different clones
        False

        Different transformer per series:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> transformers = {
        ...     'series1': StandardScaler(),
        ...     'series2': MinMaxScaler()
        ... }
        >>> result = initialize_transformer_series(
        ...     forecaster_name='ForecasterDirectMultiVariate',
        ...     series_names_in_=['series1', 'series2'],
        ...     transformer_series=transformers
        ... )
        >>> isinstance(result['series1'], StandardScaler)
        True
        >>> isinstance(result['series2'], MinMaxScaler)
        True
    """
    from copy import deepcopy

    from sklearn.base import clone

    from spotforecast2_safe.exceptions import IgnoredArgumentWarning

    if forecaster_name == "ForecasterRecursiveMultiSeries":
        if encoding is None:
            series_names_in_ = ["_unknown_level"]
        else:
            series_names_in_ = series_names_in_ + ["_unknown_level"]

    if transformer_series is None:
        transformer_series_ = {serie: None for serie in series_names_in_}
    elif not isinstance(transformer_series, dict):
        transformer_series_ = {
            serie: clone(transformer_series) for serie in series_names_in_
        }
    else:
        transformer_series_ = {serie: None for serie in series_names_in_}
        # Only elements already present in transformer_series_ are updated
        transformer_series_.update(
            {
                k: deepcopy(v)
                for k, v in transformer_series.items()
                if k in transformer_series_
            }
        )

        series_not_in_transformer_series = (
            set(series_names_in_) - set(transformer_series.keys())
        ) - {"_unknown_level"}
        if series_not_in_transformer_series:
            warnings.warn(
                f"{series_not_in_transformer_series} not present in `transformer_series`."
                f" No transformation is applied to these series.",
                IgnoredArgumentWarning,
            )

    return transformer_series_
