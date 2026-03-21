# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

import pandas as pd


def remove_duplicate_timestamps(
    df: pd.DataFrame,
    time_col: str = "Time (UTC)",
    agg: str | Callable = "mean",
) -> pd.DataFrame:
    """Resolve duplicate timestamps across all data columns.
    Groups rows that share the same ``time_col`` value and collapses them
    using the chosen aggregation.  All columns except ``time_col`` are
    aggregated.  The resulting frame is sorted chronologically, re-indexed,
    and returned.

    Args:
        df: Input dataframe containing ``time_col`` and one or more data
            columns.
        time_col: Name of the column that holds timestamps.  Defaults to
            ``"Time (UTC)"``.
        agg: Aggregation applied when collapsing duplicate rows.  Accepts
            any string recognised by
            :meth:`pandas.core.groupby.GroupBy.agg` (``"mean"``,
            ``"median"``, ``"min"``, ``"max"``, ``"sum"``, ``"std"``,
            ``"var"``, ``"first"``, ``"last"``) as well as ``"mode"``
            (most frequent value per group) or any custom callable.
            Defaults to ``"mean"``.

    Returns:
        pd.DataFrame: Deduplicated dataframe with unique ``time_col`` rows,
        sorted ascending by timestamp.

    Raises:
        KeyError: If ``time_col`` is not present in ``df``.

    Examples:
        Mean-aggregate two data columns with the default time column:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.curate_data import remove_duplicate_timestamps
        df = pd.DataFrame(
             {
                 "Time (UTC)": [
                     "2026-01-01 00:00:00",
                     "2026-01-01 00:00:00",
                     "2026-01-01 01:00:00",
                 ],
                 "Load A": [100.0, 120.0, 130.0],
                 "Load B": [200.0, 220.0, 210.0],
             }
            )
        out = remove_duplicate_timestamps(df)
        print(f"len(out): {len(out)}")
        print(f"Load A: {float(out.loc[0, 'Load A'])}")
        print(f"Load B: {float(out.loc[0, 'Load B'])}")
        ```

        Median aggregation on a custom time column:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.curate_data import remove_duplicate_timestamps
        df2 = pd.DataFrame(
            {
                "ts": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "value": [10.0, 30.0, 20.0],
            }
        )
        out2 = remove_duplicate_timestamps(
            df2, time_col="ts", agg="median"
        )
        print(f"Value: {float(out2.loc[0, 'value'])}")
        ```
    """
    if time_col not in df.columns:
        raise KeyError(
            f"Time column {time_col!r} not found in dataframe columns: "
            f"{list(df.columns)}"
        )
    agg_fn: str | Callable = (lambda x: x.mode().iloc[0]) if agg == "mode" else agg
    df = df.groupby(time_col, as_index=False).agg(agg_fn)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def get_start_end(
    data: pd.DataFrame,
    forecast_horizon: int,
    verbose: bool = True,
) -> tuple[str, str, str, str]:
    """Get start and end date strings for data and covariate ranges.
    Covariate range is extended by the forecast horizon.

    Args:
        data (pd.DataFrame):
            The dataset with a datetime index.
        forecast_horizon (int):
            The forecast horizon in hours.
        verbose (bool):
            Whether to print the determined date ranges.

    Returns:
        tuple[str, str, str, str]: (data_start, data_end, covariate_start, covariate_end)
            Date strings in the format "YYYY-MM-DDTHH:MM" for data and covariate ranges.

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.curate_data import get_start_end
        import pandas as pd
        date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')
        data = pd.DataFrame(date_rng, columns=['date'])
        data.set_index('date', inplace=True)
        start, end, cov_start, cov_end = get_start_end(data, forecast_horizon=24, verbose=False)
        print(start, end, cov_start, cov_end)
        ```
    """
    FORECAST_HORIZON = forecast_horizon

    START = data.index.min().strftime("%Y-%m-%dT%H:%M")
    END = data.index.max().strftime("%Y-%m-%dT%H:%M")
    if verbose:
        print(f"Data range: {START} to {END}")
    # Define covariate range relative to data range
    COV_START = START
    # Extend end date by forecast horizon to include future covariates
    COV_END = (pd.to_datetime(END) + pd.Timedelta(hours=FORECAST_HORIZON)).strftime(
        "%Y-%m-%dT%H:%M"
    )
    if verbose:
        print(f"Covariate data range: {COV_START} to {COV_END}")
    return START, END, COV_START, COV_END


def curate_holidays(
    holiday_df: pd.DataFrame, data: pd.DataFrame, forecast_horizon: int
):
    """Checks if the holiday dataframe has the correct shape.
    Args:
        holiday_df (pd.DataFrame):
            DataFrame containing holiday information.
        data (pd.DataFrame):
            The main dataset.
        forecast_horizon (int):
            The forecast horizon in hours.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data, fetch_holiday_data
        >>> from spotforecast2_safe.preprocessing.curate_data import get_start_end, curate_holidays
        >>> data = fetch_data()
        >>> START, END, COV_START, COV_END = get_start_end(
        ...     data=data,
        ...     forecast_horizon=24,
        ...     verbose=False
        ... )
        >>> holiday_df = fetch_holiday_data(
        ...     start='2023-01-01T00:00',
        ...     end='2023-01-10T00:00',
        ...     tz='UTC',
        ...     freq='h',
        ...     country_code='DE',
        ...     state='NW'
        ... )
        >>> FORECAST_HORIZON = 24
        >>> curate_holidays(holiday_df, data, forecast_horizon=FORECAST_HORIZON)

    Raises:
        AssertionError:
            If the holiday dataframe does not have the correct number of rows.
    """
    try:
        assert holiday_df.shape[0] == data.shape[0] + forecast_horizon
        print("Holiday dataframe has correct shape.")
    except AssertionError:
        print("Holiday dataframe has wrong shape.")


def curate_weather(weather_df: pd.DataFrame, data: pd.DataFrame, forecast_horizon: int):
    """Checks if the weather dataframe has the correct shape.

    Args:
        weather_df (pd.DataFrame):
            DataFrame containing weather information.
        data (pd.DataFrame):
            The main dataset.
        forecast_horizon (int):
            The forecast horizon in hours.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data, fetch_weather_data
        >>> from spotforecast2_safe.preprocessing.curate_data import get_start_end, curate_weather
        >>> data = fetch_data()
        >>> START, END, COV_START, COV_END = get_start_end(
        ...     data=data,
        ...     forecast_horizon=24,
        ...     verbose=False
        ... )
        >>> weather_df = fetch_weather_data(
        ...     cov_start=COV_START,
        ...     cov_end=COV_END,
        ...     tz='UTC',
        ...     freq='h',
        ...     latitude=51.5136,
        ...     longitude=7.4653
        ... )
        >>> FORECAST_HORIZON = 24
        >>> curate_weather(weather_df, data, forecast_horizon=FORECAST_HORIZON)

    Raises:
        AssertionError:
            If the weather dataframe does not have the correct number of rows.
    """
    try:
        assert weather_df.shape[0] == data.shape[0] + forecast_horizon
        print("Weather dataframe has correct shape.")
    except AssertionError:
        print("Weather dataframe has wrong shape.")


def basic_ts_checks(data: pd.DataFrame, verbose: bool = False) -> bool:
    """Checks if the time series data has a datetime index and is sorted.

    Args:
        data (pd.DataFrame):
            The main dataset.
        verbose (bool):
            Whether to print additional information.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2_safe.preprocessing.curate_data import basic_ts_checks
        >>> data = fetch_data()
        >>> basic_ts_checks(data)

    Raises:
        TypeError:
            If the index is not a datetime index.
        ValueError:
            If the datetime index is not sorted in increasing order or is incomplete.

    Returns:
        bool: True if the datetime index is valid, sorted, and complete.
    """
    # Check if the time series data has a datetime index
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        raise TypeError("The index is not a datetime index.")

    # Check if the datetime index is sorted
    if not data.index.is_monotonic_increasing:
        raise ValueError("The datetime index is not sorted in increasing order.")

    # Check if the index is complete (no missing timestamps)
    start_date = data.index.min()
    end_date = data.index.max()
    complete_date_range = pd.date_range(
        start=start_date, end=end_date, freq=data.index.freq
    )
    is_index_complete = (data.index == complete_date_range).all()

    if not is_index_complete:
        raise ValueError(
            "The datetime index has missing timestamps and is not complete."
        )
    if verbose:
        print(
            "The time series data has a valid datetime index that is sorted and complete."
        )
    return True


def agg_and_resample_data(
    data: pd.DataFrame,
    rule: str = "h",
    closed: str = "left",
    label: str = "left",
    by="mean",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aggregates and resamples the data to (e.g.,hourly) frequency by computing the specified aggregation (e.g. for each hour).

    Args:
        data (pd.DataFrame):
            The dataset with a datetime index.
        rule (str):
            The resample rule (e.g., 'h' for hourly, 'D' for daily).
            Default is 'h' which creates an hourly grid.
        closed (str):
            Which side of bin interval is closed. Default is 'left'.
            Using `closed="left", label="left"` specifies that a time interval
            (e.g., 10:00 to 11:00) is labeled with the start timestamp (10:00).
            For consumption data, a different representation is usually more common:
            `closed="left", label="right"`, so the interval is labeled with the end
            timestamp (11:00), since consumption is typically reported after one hour.
        label (str):
            Which bin edge label to use. Default is 'left'.
            See 'closed' parameter for details on labeling behavior.
        by (str or callable):
            Aggregation method to apply (e.g., 'mean', 'sum', 'median').
            Default is 'mean'.
            The aggregation serves robustness: if the data were more finely resolved
            (e.g., quarter-hourly), asfreq would only pick one value (sampling),
            while .agg("mean") forms the correct average over the hour.
            If the data is already hourly, .agg doesn't change anything but ensures
            that no duplicates exist.
        verbose (bool):
            Whether to print additional information.

    Returns:
        pd.DataFrame: Resampled and aggregated dataframe.

    Notes:
        - resample(rule="h"): Creates an hourly grid
        - closed/label: Control how time intervals are labeled
        - .agg({...: by}): Aggregates values within each time bin

    Examples:
        >>> from spotforecast2_safe.preprocessing.curate_data import agg_and_resample_data
        >>> import pandas as pd
        >>> date_rng = pd.date_range(start='2023-01-01', end='2023-01-02', freq='15min')
        >>> data = pd.DataFrame(date_rng, columns=['date'])
        >>> data.set_index('date', inplace=True)
        >>> data['value'] = range(len(data))
        >>> resampled_data = agg_and_resample_data(data, rule='h', by='mean')
        >>> print(resampled_data.head())
    """
    if verbose:
        print(f"Original data shape: {data.shape}")
    # Create aggregation dictionary for all columns
    agg_dict = {col: by for col in data.columns}

    data = data.resample(rule=rule, closed=closed, label=label).agg(agg_dict)
    if verbose:
        print(
            f"Data resampled with rule='{rule}', closed='{closed}', label='{label}', aggregation='{by}'."
        )
        print(f"Resampled data shape: {data.shape}")
    return data


def reset_index(df: pd.DataFrame, index_name: str = "DateTime") -> pd.DataFrame:
    """Resets the index of the dataframe and assigns a name to the index column.

    Args:
        df (pd.DataFrame): The input dataframe with a datetime index.
        index_name (str): The name to assign to the index column after resetting. Default is "DateTime".

    Returns:
        pd.DataFrame: The dataframe with the reset index.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.curate_data import reset_index
        date_rng = pd.date_range(start='2023-01-01', end='2023-01-02', freq='h')
        data = pd.DataFrame(date_rng, columns=['date'])
        data.set_index('date', inplace=True)
        data['value'] = range(len(data))
        reset_data = reset_index(data, index_name='DateTime')
        print(reset_data.head())
        ```
    """
    df.index.name = index_name
    df = df.reset_index()
    return df
