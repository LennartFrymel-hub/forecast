# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from pathlib import Path
from os import environ
from typing import Optional, Union
from spotforecast2_safe.utils.generate_holiday import create_holiday_df
from spotforecast2_safe.utils.convert_to_utc import convert_to_utc
from pandas import Timestamp
from spotforecast2_safe.weather.weather_client import WeatherService


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return the location where datasets are to be stored.
    By default the data directory is set to a folder named 'spotforecast2_data' in the
    user home folder. Alternatively, it can be set by the 'SPOTFORECAST2_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Args:
        data_home (str or pathlib.Path, optional):
            The path to spotforecast data directory. If `None`, the default path
            is `~/spotforecast2_data`.

    Returns:
        data_home (pathlib.Path):
            The path to the spotforecast data directory.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import get_data_home
        from pathlib import Path
        get_data_home()
        get_data_home(Path('/tmp/spotforecast2_data'))
        ```
    """
    if data_home is None:
        data_home = environ.get(
            "SPOTFORECAST2_DATA", Path.home() / "spotforecast2_data"
        )
    # Ensure data_home is a Path() object pointing to an absolute path
    data_home = Path(data_home).expanduser().absolute()
    # Create data directory if it does not exists.
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def get_package_data_home() -> Path:
    """Return the location of the internal package datasets.

    Returns:
        pathlib.Path:
            The path to the spotforecast package data directory.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import get_package_data_home
        package_data_dir = get_package_data_home()
        print(package_data_dir.name)
        print(package_data_dir.parent.name)
        ```
    """
    return Path(__file__).parent.parent / "datasets" / "csv"


def get_cache_home(
    cache_home: Optional[Union[str, Path]] = None,
    create_dir: bool = True,
) -> Path:
    """Return the location where persistent models are to be cached.

    By default the cache directory is set to a folder named
    ``.spotforecast2_cache`` in the user home folder.  Alternatively, it
    can be set by the ``SPOTFORECAST2_CACHE`` environment variable or
    programmatically by giving an explicit folder path.  The ``~`` symbol
    is expanded to the user home folder.  When ``create_dir`` is ``True``
    (the default) the directory is created automatically if it does not
    already exist.

    This directory is used to store pickled trained models for quick
    reuse across forecasting runs, following scikit-learn model
    persistence conventions.

    Args:
        cache_home: Path to the spotforecast cache directory.  If
            ``None``, the value of the ``SPOTFORECAST2_CACHE`` environment
            variable is used when set, otherwise the default path
            ``~/.spotforecast2_cache`` is used.
        create_dir: Whether to create the cache directory if it does not
            exist.  When ``True`` (the default), the directory and any
            missing parent directories are created automatically.  When
            ``False``, the resolved path is returned without touching the
            filesystem.

    Returns:
        Absolute path to the spotforecast cache directory.

    Raises:
        OSError: If ``create_dir`` is ``True`` and the directory cannot
            be created due to a permissions error or other OS-level
            failure.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import get_cache_home
        cache_dir = get_cache_home()
        print(cache_dir.name)
        print(cache_dir.parent.name)
        ```

        ```{python}
        # Using custom path
        from spotforecast2_safe.data.fetch_data import get_cache_home
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            custom_cache = get_cache_home(Path(tmp) / 'my_cache')
            print(custom_cache.exists())
        ```

        ```{python}
        # Resolve path without creating the directory
        from spotforecast2_safe.data.fetch_data import get_cache_home
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            resolved = get_cache_home(Path(tmp) / 'not_yet', create_dir=False)
            print(resolved.exists())
        ```

        ```{python}
        # Using environment variable
        from spotforecast2_safe.data.fetch_data import get_cache_home
        import os
        os.environ['SPOTFORECAST2_CACHE'] = '/tmp/spotforecast2_cache_env'
        cache_dir = get_cache_home()
        cache_dir.as_posix()
        del os.environ['SPOTFORECAST2_CACHE']
        ```
    """
    if cache_home is None:
        cache_home = environ.get(
            "SPOTFORECAST2_CACHE", Path.home() / ".spotforecast2_cache"
        )
    # Ensure cache_home is a Path() object pointing to an absolute path
    cache_home = Path(cache_home).expanduser().absolute()
    if create_dir:
        # Create cache directory if it does not exist
        cache_home.mkdir(parents=True, exist_ok=True)
    return cache_home


def fetch_data(
    filename: Optional[Union[str, Path]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    columns: Optional[list] = None,
    index_col: int = 0,
    parse_dates: bool = True,
    dayfirst: bool = False,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetches a dataset from a CSV file or processes a DataFrame.

    Args:
        filename (str or Path, optional):
            Full absolute path of the CSV file containing the dataset
            (e.g., ``'/home/data/my_data.csv'``).  Required when
            *dataframe* is ``None``.  Use :func:`get_data_home` or
            :func:`get_package_data_home` to build the path:

            .. code-block:: python

                fetch_data(filename=get_data_home() / "my_data.csv")

        dataframe (pd.DataFrame, optional):
            A pandas DataFrame to process. If provided, it will be processed with
            proper timezone handling. Mutually exclusive with filename.
        columns (list, optional):
            List of columns to be included in the dataset. If None, all columns are included.
            If an empty list is provided, a ValueError is raised. Default: None.
        index_col (int):
            Column index to be used as the index. Default: 0.
        parse_dates (bool):
            Whether to parse dates in the index column. Default: True.
        dayfirst (bool):
            Whether the day comes first in date parsing. Default: False.
        timezone (str):
            Timezone to set for the datetime index. If a DataFrame with naive index is provided,
            it will be localized to this timezone then converted to UTC. Default: "UTC".

    Returns:
        pd.DataFrame: The dataset with UTC timezone.

    Raises:
        ValueError: If columns is an empty list, if both filename and dataframe are provided,
            if neither filename nor dataframe is provided, or if filename is not an absolute path.
        FileNotFoundError: If CSV file does not exist.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        # demo02.csv is included in the package datasets
        path_demo = get_package_data_home() / "demo02.csv"
        df = fetch_data(filename=path_demo)
        df.head()
        ```
    """
    if columns is not None and len(columns) == 0:
        raise ValueError("columns must be specified and cannot be empty.")

    if filename is not None and dataframe is not None:
        raise ValueError(
            "Cannot specify both filename and dataframe. Please provide only one."
        )

    if dataframe is not None:
        df = dataframe.copy()
        df = convert_to_utc(df, timezone)
        if columns is not None:
            df = df[columns].copy()
    else:
        if filename is None:
            raise ValueError(
                "filename must be specified when dataframe is None. "
                "Provide a full absolute path (e.g., get_data_home() / 'my_data.csv') "
                "or a DataFrame."
            )

        csv_path = Path(filename)
        if not csv_path.is_absolute():
            raise ValueError(
                f"filename must be an absolute path, got: '{filename}'. "
                "Use get_data_home() or get_package_data_home() to build the path:\n"
                "    fetch_data(filename=get_data_home() / 'my_data.csv')"
            )

        if not csv_path.is_file():
            raise FileNotFoundError(f"The file {csv_path} does not exist.")

        # Determine which columns to load for efficient reading
        usecols = None
        if columns is not None:
            if isinstance(index_col, int):
                header_df = pd.read_csv(csv_path, nrows=0)
                index_col_name = header_df.columns[index_col]
            else:
                index_col_name = index_col
            usecols = [index_col_name] + columns

        df = pd.read_csv(
            csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
            dayfirst=dayfirst,
            usecols=usecols,
        )
        df = convert_to_utc(df, timezone)

    if df.index.freq is None:
        try:
            df.index.freq = pd.infer_freq(df.index)
        except (ValueError, TypeError):
            pass  # If the frequency cannot be inferred, leave df.index.freq as None.

    return df


def fetch_holiday_data(
    start: str | Timestamp,
    end: str | Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Fetches holiday data for the dataset period.

    Args:
        start (str or pd.Timestamp):
            Start date of the dataset period.
        end (str or pd.Timestamp):
            End date of the dataset period.
        tz (str):
            Timezone for the holiday data.
        freq (str):
            Frequency of the holiday data.
        country_code (str):
            Country code for the holidays.
        state (str):
            State code for the holidays.

    Returns:
        pd.DataFrame: DataFrame containing holiday information.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_holiday_data
        holiday_df = fetch_holiday_data(
            start='2023-01-01T00:00',
            end='2023-01-10T00:00',
            tz='UTC',
            freq='h',
            country_code='DE',
            state='NW'
        )
        holiday_df.head()
        ```
    """
    holiday_df = create_holiday_df(
        start=start, end=end, tz=tz, freq=freq, country_code=country_code, state=state
    )
    return holiday_df


def fetch_weather_data(
    cov_start: str,
    cov_end: str,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    freq: str = "h",
    fallback_on_failure: bool = True,
    cache_home: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Fetch weather data for the dataset period plus forecast horizon.

    Creates a weather DataFrame using the Open-Meteo API with optional
    caching.  Caching is controlled solely by the cache_home argument:
    when a path is provided the service reads from / writes to a parquet
    cache file inside that directory; when None (the default) no caching
    is performed.

    Args:
        cov_start: Start date for covariate data.
        cov_end: End date for covariate data.
        latitude: Latitude of the location for weather data.
            Default is 51.5136 (Dortmund).
        longitude: Longitude of the location for weather data.
            Default is 7.4653 (Dortmund).
        timezone: Timezone for the weather data.
        freq: Frequency of the weather data.
        fallback_on_failure: Whether to use fallback data in case of
            failure.
        cache_home: Optional path to cache directory.  When provided,
            fetched weather data is cached in
            ``<cache_home>/weather_cache.parquet``.  When None (default),
            no caching is performed.

    Returns:
        pd.DataFrame: DataFrame containing weather information.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_weather_data
        weather_df = fetch_weather_data(
            cov_start='2023-01-01T00:00',
            cov_end='2023-01-11T00:00',
            latitude=51.5136,
            longitude=7.4653,
            timezone='UTC',
            freq='h',
            fallback_on_failure=True,
            cache_home='~/.spotforecast2_cache')
        weather_df.head()
        ```
    """
    if cache_home is not None:
        cache_path = get_cache_home(cache_home=cache_home) / "weather_cache.parquet"
    else:
        cache_path = None

    service = WeatherService(
        latitude=latitude, longitude=longitude, cache_path=cache_path
    )

    weather_df = service.get_dataframe(
        start=cov_start,
        end=cov_end,
        timezone=timezone,
        freq=freq,
        fallback_on_failure=fallback_on_failure,
    )
    return weather_df


def load_timeseries(
    data_home: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """Load the actual-load time series from ``interim/energy_load.csv``.
    Reads the ``Actual Load`` column, converts the index to a UTC
    ``DatetimeIndex`` with hourly frequency, and fills any missing
    values with forward/backward fill.

    Args:
        data_home: Root data directory.  If *None*, resolved via
            :func:`get_data_home`.

    Returns:
        pd.Series: Hourly actual-load series indexed by UTC timestamps.

    Raises:
        FileNotFoundError: If ``interim/energy_load.csv`` does not exist.

    Examples:
        >>> import os, tempfile, shutil
        >>> import pandas as pd
        >>> from spotforecast2_safe.data.fetch_data import (
        ...     load_timeseries, get_package_data_home,
        ... )
        >>> tmp = tempfile.mkdtemp()
        >>> os.environ["SPOTFORECAST2_DATA"] = tmp
        >>> interim = os.path.join(tmp, "interim")
        >>> os.makedirs(interim, exist_ok=True)
        >>> demo = get_package_data_home() / "demo01.csv"
        >>> df = pd.read_csv(demo)
        >>> df = df.rename(columns={
        ...     "Time": "Time (UTC)",
        ...     "Actual": "Actual Load",
        ...     "Forecast": "Forecasted Load",
        ... })
        >>> df.to_csv(os.path.join(interim, "energy_load.csv"), index=False)
        >>> y = load_timeseries()
        >>> isinstance(y, pd.Series)
        True
        >>> y.index.tz is not None
        True
        >>> shutil.rmtree(tmp)
        >>> del os.environ["SPOTFORECAST2_DATA"]
    """
    data_dir = get_data_home(data_home)
    csv_path = data_dir / "interim" / "energy_load.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}. "
            "Run the downloader first or place energy_load.csv "
            "in the 'interim' sub-directory."
        )

    df = pd.read_csv(csv_path, parse_dates=["Time (UTC)"])
    df = df.set_index("Time (UTC)")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    df = df.asfreq("h")

    y = df["Actual Load"]
    if y.isna().any():
        y = y.ffill().bfill()
    return y


def load_timeseries_forecast(
    data_home: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """Load the day-ahead forecast time series from ``interim/energy_load.csv``.
    Reads the ``Forecasted Load`` column, converts the index to a UTC
    ``DatetimeIndex`` with hourly frequency, and fills any missing
    values with forward/backward fill.

    Args:
        data_home: Root data directory.  If *None*, resolved via
            :func:`get_data_home`.

    Returns:
        pd.Series: Hourly forecasted-load series indexed by UTC timestamps.

    Raises:
        FileNotFoundError: If ``interim/energy_load.csv`` does not exist.
        KeyError: If ``Forecasted Load`` column is not present.

    Examples:
        >>> import os, tempfile, shutil
        >>> import pandas as pd
        >>> from spotforecast2_safe.data.fetch_data import (
        ...     load_timeseries_forecast, get_package_data_home,
        ... )
        >>> tmp = tempfile.mkdtemp()
        >>> os.environ["SPOTFORECAST2_DATA"] = tmp
        >>> interim = os.path.join(tmp, "interim")
        >>> os.makedirs(interim, exist_ok=True)
        >>> demo = get_package_data_home() / "demo01.csv"
        >>> df = pd.read_csv(demo)
        >>> df = df.rename(columns={
        ...     "Time": "Time (UTC)",
        ...     "Actual": "Actual Load",
        ...     "Forecast": "Forecasted Load",
        ... })
        >>> df.to_csv(os.path.join(interim, "energy_load.csv"), index=False)
        >>> y_f = load_timeseries_forecast()
        >>> isinstance(y_f, pd.Series)
        True
        >>> shutil.rmtree(tmp)
        >>> del os.environ["SPOTFORECAST2_DATA"]
    """
    data_dir = get_data_home(data_home)
    csv_path = data_dir / "interim" / "energy_load.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}. "
            "Run the downloader first or place energy_load.csv "
            "in the 'interim' sub-directory."
        )

    df = pd.read_csv(csv_path, parse_dates=["Time (UTC)"])
    df = df.set_index("Time (UTC)")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    df = df.asfreq("h")

    y = df["Forecasted Load"]
    if y.isna().any():
        y = y.ffill().bfill()
    return y
