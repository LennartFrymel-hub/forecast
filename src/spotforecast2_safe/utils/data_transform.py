# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Data transformation utilities for time series forecasting.

This module provides functions for normalizing and transforming data formats.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def date_to_index_position(
    index: pd.Index,
    date_input: Union[int, str, pd.Timestamp],
    method: str = "prediction",
    date_literal: str = "steps",
    kwargs_pd_to_datetime: Optional[dict] = None,
) -> int:
    """
    Transform a datetime string or pandas Timestamp to an integer. The integer
    represents the position of the datetime in the index.

    Args:
        index:
            Original datetime index (must be a pandas DatetimeIndex if `date_input` is not an int).
        date_input:
            Datetime to transform to integer.
            - If int, returns the same integer.
            - If str or pandas Timestamp, it is converted and expanded into the index.
        method:
            Can be 'prediction' or 'validation'.
            - If 'prediction', the date must be later than the last date in the index.
            - If 'validation', the date must be within the index range.
        date_literal:
            Variable name used in error messages. Defaults to 'steps'.
        kwargs_pd_to_datetime:
            Additional keyword arguments to pass to `pd.to_datetime()`. Defaults to None.

    Returns:
        int:
            `date_input` transformed to integer position in the `index`.
            - If `date_input` is an integer, it returns the same integer.
            - If method is 'prediction', number of steps to predict from the last date in the index.
            - If method is 'validation', position plus one of the date in the index.

    Raises:
        ValueError: If `method` is not 'prediction' or 'validation'.
        TypeError: If `index` is not a DatetimeIndex when `date_input` is not an integer.
        ValueError: If `date_input` (as date) does not meet the method's constraints.
        TypeError: If `date_input` is not an integer, string, or pandas Timestamp.
    """
    # Initialize output to satisfy static analyzers; it will be overwritten
    # on all valid execution paths before being returned.
    output: int = 0

    if kwargs_pd_to_datetime is None:
        kwargs_pd_to_datetime = {}

    if method not in ["prediction", "validation"]:
        raise ValueError("`method` must be 'prediction' or 'validation'.")

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
        output = int(date_input)
    else:
        raise TypeError(
            f"`{date_literal}` must be an integer, string, or pandas Timestamp."
        )

    return output


def input_to_frame(
    data: Union[pd.Series, pd.DataFrame], input_name: str
) -> pd.DataFrame:
    """
    Convert input data to a pandas DataFrame.

    This function ensures consistent DataFrame format for internal processing.
    If data is already a DataFrame, it's returned as-is. If it's a Series,
    it's converted to a single-column DataFrame.

    Args:
        data: Input data as pandas Series or DataFrame.
        input_name: Name of the input data type. Accepted values are:
            - 'y': Target time series
            - 'last_window': Last window for prediction
            - 'exog': Exogenous variables

    Returns:
        DataFrame version of the input data. For Series input, uses the series
        name if available, otherwise uses a default name based on input_name.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.utils.data_transform import input_to_frame
        >>>
        >>> # Series with name
        >>> y = pd.Series([1, 2, 3], name="sales")
        >>> df = input_to_frame(y, input_name="y")
        >>> df.columns.tolist()
        ['sales']
        >>>
        >>> # Series without name (uses default)
        >>> y_no_name = pd.Series([1, 2, 3])
        >>> df = input_to_frame(y_no_name, input_name="y")
        >>> df.columns.tolist()
        ['y']
        >>>
        >>> # DataFrame (returned as-is)
        >>> df_input = pd.DataFrame({"temp": [20, 21], "humidity": [50, 55]})
        >>> df_output = input_to_frame(df_input, input_name="exog")
        >>> df_output.columns.tolist()
        ['temp', 'humidity']
        >>>
        >>> # Exog series without name
        >>> exog = pd.Series([10, 20, 30])
        >>> df_exog = input_to_frame(exog, input_name="exog")
        >>> df_exog.columns.tolist()
        ['exog']
    """
    output_col_name = {"y": "y", "last_window": "y", "exog": "exog"}

    if isinstance(data, pd.Series):
        data = data.to_frame(
            name=data.name if data.name is not None else output_col_name[input_name]
        )

    return data


def expand_index(index: Union[pd.Index, None], steps: int) -> pd.Index:
    """
    Create a new index extending from the end of the original index.

    This function generates future indices for forecasting by extending the time
    series index by a specified number of steps. Handles both DatetimeIndex and
    RangeIndex appropriately.

    Args:
        index: Original pandas Index (DatetimeIndex or RangeIndex). If None,
            creates a RangeIndex starting from 0.
        steps: Number of future steps to generate.

    Returns:
        New pandas Index with `steps` future periods.

    Raises:
        TypeError: If steps is not an integer, or if index is neither DatetimeIndex
            nor RangeIndex.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.utils.data_transform import expand_index
        >>>
        >>> # DatetimeIndex
        >>> dates = pd.date_range("2023-01-01", periods=5, freq="D")
        >>> new_index = expand_index(dates, 3)
        >>> new_index
        DatetimeIndex(['2023-01-06', '2023-01-07', '2023-01-08'], dtype='datetime64[ns]', freq='D')
        >>>
        >>> # RangeIndex
        >>> range_idx = pd.RangeIndex(start=0, stop=10)
        >>> new_index = expand_index(range_idx, 5)
        >>> new_index
        RangeIndex(start=10, stop=15, step=1)
        >>>
        >>> # None index (creates new RangeIndex)
        >>> new_index = expand_index(None, 3)
        >>> new_index
        RangeIndex(start=0, stop=3, step=1)
        >>>
        >>> # Invalid: steps not an integer
        >>> try:
        ...     expand_index(dates, 3.5)
        ... except TypeError as e:
        ...     print("Error: steps must be an integer")
        Error: steps must be an integer
    """
    if not isinstance(steps, (int, np.integer)):
        raise TypeError(f"`steps` must be an integer. Got {type(steps)}.")

    # Convert numpy integer to Python int if needed
    if isinstance(steps, np.integer):
        steps = int(steps)

    if isinstance(index, pd.Index):
        if isinstance(index, pd.DatetimeIndex):
            new_index = pd.date_range(
                start=index[-1] + index.freq, periods=steps, freq=index.freq
            )
        elif isinstance(index, pd.RangeIndex):
            new_index = pd.RangeIndex(start=index[-1] + 1, stop=index[-1] + 1 + steps)
        else:
            raise TypeError(
                "Argument `index` must be a pandas DatetimeIndex or RangeIndex."
            )
    else:
        new_index = pd.RangeIndex(start=0, stop=steps)

    return new_index


def transform_dataframe(
    df: pd.DataFrame,
    transformer: object,
    fit: bool = False,
    inverse_transform: bool = False,
) -> pd.DataFrame:
    """
    Transform raw values of pandas DataFrame with a scikit-learn alike
    transformer, preprocessor or ColumnTransformer.

    The transformer used must have the following methods: fit, transform,
    fit_transform and inverse_transform. ColumnTransformers are not allowed
    since they do not have inverse_transform method.

    Args:
        df: DataFrame to be transformed.
        transformer: Scikit-learn alike transformer, preprocessor, or ColumnTransformer.
            Must implement fit, transform, fit_transform and inverse_transform.
        fit: Train the transformer before applying it. Defaults to False.
        inverse_transform: Transform back the data to the original representation.
            This is not available when using transformers of class
            scikit-learn ColumnTransformers. Defaults to False.

    Returns:
        Transformed DataFrame.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If inverse_transform is requested for ColumnTransformer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` argument must be a pandas DataFrame. Got {type(df)}")

    if transformer is None:
        return df

    # Check for ColumnTransformer by class name to avoid importing sklearn
    is_column_transformer = type(
        transformer
    ).__name__ == "ColumnTransformer" or hasattr(transformer, "transformers")

    if inverse_transform and is_column_transformer:
        raise ValueError(
            "`inverse_transform` is not available when using ColumnTransformers."
        )

    if not inverse_transform:
        if fit:
            values_transformed = transformer.fit_transform(df)
        else:
            values_transformed = transformer.transform(df)
    else:
        values_transformed = transformer.inverse_transform(df)

    if hasattr(values_transformed, "toarray"):
        # If the returned values are in sparse matrix format, it is converted to dense
        values_transformed = values_transformed.toarray()

    if isinstance(values_transformed, pd.DataFrame):
        df_transformed = values_transformed
    else:
        df_transformed = pd.DataFrame(
            values_transformed, index=df.index, columns=df.columns
        )

    return df_transformed
