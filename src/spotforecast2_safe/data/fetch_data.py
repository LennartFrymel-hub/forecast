# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from spotforecast2_safe.data.data import Data
from pathlib import Path
from os import environ
from typing import Optional, Union
from spotforecast2_safe.utils.generate_holiday import create_holiday_df
from pandas import Timestamp
from spotforecast2_safe.weather.weather_client import WeatherService
import logging


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


def get_cache_home(cache_home: Optional[Union[str, Path]] = None) -> Path:
    """Return the location where persistent models are to be cached.
    By default the cache directory is set to a folder named 'spotforecast2_cache' in the
    user home folder. Alternatively, it can be set by the 'SPOTFORECAST2_CACHE' environment
    variable or programmatically by giving an explicit folder path. The '~' symbol is
    expanded to the user home folder. If the folder does not already exist, it is
    automatically created.

    This directory is used to store pickled trained models for quick reuse across
    forecasting runs, following scikit-learn model persistence conventions.

    Args:
        cache_home (str or pathlib.Path, optional):
            The path to spotforecast cache directory. If `None`, the default path
            is `~/spotforecast2_cache`.

    Returns:
        pathlib.Path:
            The path to the spotforecast cache directory.

    Raises:
        OSError: If the directory cannot be created due to permission issues.

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
        import tempfile
        from pathlib import Path
        custom_cache = get_cache_home(Path('/tmp/my_cache'))
        custom_cache.exists()
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
            "SPOTFORECAST2_CACHE", Path.home() / "spotforecast2_cache"
        )
    # Ensure cache_home is a Path() object pointing to an absolute path
    cache_home = Path(cache_home).expanduser().absolute()
    # Create cache directory if it does not exist
    cache_home.mkdir(parents=True, exist_ok=True)
    return cache_home


unused_logger = logging.getLogger(__name__)


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


def fetch_data(
    filename: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    columns: Optional[list] = None,
    index_col: int = 0,
    parse_dates: bool = True,
    dayfirst: bool = False,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetches the integrated raw dataset from a CSV file or processes a DataFrame.

    Args:
        filename (str, optional):
            Filename of the CSV file containing the dataset. Must be located in the
            data home directory. This is required if dataframe is None.
        dataframe (pd.DataFrame, optional):
            A pandas DataFrame to process. If provided, it will be processed with
            proper timezone handling. Mutually exclusive with filename.
        columns (list, optional):
            List of columns to be included in the dataset. If None, all columns are included.
            If an empty list is provided, a ValueError is raised. Default: None.
        index_col (int):
            Column index to be used as the index (only used when loading from CSV). Default: 0.
        parse_dates (bool):
            Whether to parse dates in the index column (only used when loading from CSV). Default: True.
        dayfirst (bool):
            Whether the day comes first in date parsing (only used when loading from CSV). Default: False.
        timezone (str):
            Timezone to set for the datetime index. If a DataFrame with naive index is provided,
            it will be localized to this timezone then converted to UTC. Default: "UTC".

    Returns:
        pd.DataFrame: The integrated raw dataset with UTC timezone.

    Raises:
        ValueError: If columns is an empty list, if both filename and dataframe are provided,
            or if neither filename nor dataframe is provided.
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

    # Process DataFrame if provided
    if dataframe is not None:
        dataset = Data.from_dataframe(
            df=dataframe,
            timezone=timezone,
            columns=columns,
        )
    else:
        # Load from CSV file
        if filename is None:
            raise ValueError(
                "filename must be specified when dataframe is None. "
                "Explicitly provide a filename (e.g., 'data_in.csv') or a DataFrame."
            )

        # Check if the filename is an absolute path
        csv_path = Path(filename)
        if not csv_path.is_absolute():
            csv_path = get_data_home() / filename

        if not csv_path.is_file():
            raise FileNotFoundError(f"The file {csv_path} does not exist.")

        dataset = Data.from_csv(
            csv_path=csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
            dayfirst=dayfirst,
            timezone=timezone,
            columns=columns,
        )

    return dataset.data


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
    cached=True,
) -> pd.DataFrame:
    """Fetches weather data for the dataset period plus forecast horizon.
        Create weather dataframe using API with optional caching.

    Args:
        cov_start (str):
            Start date for covariate data.
        cov_end (str):
            End date for covariate data.
        latitude (float):
            Latitude of the location for weather data. Default is 51.5136 (Dortmund).
        longitude (float):
            Longitude of the location for weather data. Default is 7.4653 (Dortmund).
        timezone (str):
            Timezone for the weather data.
        freq (str):
            Frequency of the weather data.
        fallback_on_failure (bool):
            Whether to use fallback data in case of failure.
        cached (bool):
            Whether to use cached data.

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
            cached=True)
        weather_df.head()
        ```
    """
    if cached:
        cache_path = get_data_home() / "weather_cache.parquet"
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
