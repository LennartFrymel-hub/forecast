# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calendar, day/night, and holiday feature engineering for forecasting pipelines.

This module exposes three public feature-engineering helpers:

- :func:`get_calendar_features` — extract month, week, day-of-week, and hour
  from a time index.
- :func:`get_day_night_features` — derive sunrise hour, sunset hour, daylight
  hours, and an ``is_daylight`` indicator using the
  `astral <https://astral.readthedocs.io>`_ library.
- :func:`get_holiday_features` — align binary public-holiday indicators to a
  regular time grid using
  :func:`~spotforecast2_safe.data.fetch_data.fetch_holiday_data`.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from astral import LocationInfo

from spotforecast2_safe.data.fetch_data import fetch_holiday_data
from spotforecast2_safe.preprocessing.curate_data import curate_holidays

try:
    from astral.sun import sun
except ImportError:  # pragma: no cover
    raise ImportError("astral is required. Install with: pip install astral")

try:
    from feature_engine.datetime import DatetimeFeatures
except ImportError:  # pragma: no cover
    raise ImportError(
        "feature_engine is required. Install with: pip install feature-engine"
    )


def get_calendar_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    freq: str = "h",
    timezone: str = "UTC",
    features_to_extract: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create calendar-based features for a contiguous time range.

    Uses :class:`~feature_engine.datetime.DatetimeFeatures` to extract
    temporal components from a regularly spaced ``DatetimeIndex``.  The
    resulting DataFrame has the same index as the generated time grid and
    one integer column per requested feature.

    Args:
        start: Start of the time range.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end of the time range.  String values are
            parsed with ``utc=True``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.
        features_to_extract: Calendar components to extract.  Defaults
            to ``["month", "week", "day_of_week", "hour"]``.

    Returns:
        pd.DataFrame: DataFrame with integer columns for each extracted
        calendar feature.  The index is a tz-aware
        :class:`~pandas.DatetimeIndex` with the requested ``freq``.

    Raises:
        ValueError: If ``start`` is later than ``cov_end``.

    Examples:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.exo.calendar import get_calendar_features

        start = pd.Timestamp("2024-01-01", tz="UTC")
        cov_end = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        features = get_calendar_features(
            start=start,
            cov_end=cov_end,
            freq="h",
            timezone="UTC",
        )
        print("shape:", features.shape)
        print("columns:", features.columns.tolist())
        print(features.head(3))
        ```
    """
    if features_to_extract is None:
        features_to_extract = ["month", "week", "day_of_week", "hour"]

    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    calendar_transformer = DatetimeFeatures(
        variables="index",
        features_to_extract=features_to_extract,
        drop_original=True,
    )

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    extended_data = pd.DataFrame(index=extended_index)
    extended_data["dummy"] = 0

    return calendar_transformer.fit_transform(extended_data)[features_to_extract]


def get_day_night_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    location: LocationInfo,
    freq: str = "h",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Create day/night features using astronomical sunrise and sunset times.

    Sunrise and sunset times are computed once per unique calendar date
    (using :func:`astral.sun.sun`) and then broadcast to all timestamps
    in the requested hourly grid, which avoids redundant computation for
    large date ranges.

    The returned DataFrame contains four columns:

    - ``sunrise_hour`` — rounded sunrise hour (0–23).
    - ``sunset_hour`` — rounded sunset hour (0–23).
    - ``daylight_hours`` — ``sunset_hour - sunrise_hour``.
    - ``is_daylight`` — ``1`` if the timestamp is between sunrise and
      sunset, else ``0``.

    Args:
        start: Start of the time range.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end of the time range.  String values are
            parsed with ``utc=True``.
        location: :class:`~astral.LocationInfo` instance describing the
            geographic location (latitude, longitude, timezone).
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        timezone: Timezone label applied to the generated index.
            Defaults to ``"UTC"``.

    Returns:
        pd.DataFrame: DataFrame with columns ``sunrise_hour``,
        ``sunset_hour``, ``daylight_hours``, ``is_daylight``.  The index
        is a tz-aware :class:`~pandas.DatetimeIndex` with the requested
        ``freq``.

    Examples:

        ```{python}
        import pandas as pd
        from astral import LocationInfo
        from spotforecast2_safe.manager.exo.calendar import get_day_night_features

        start = pd.Timestamp("2024-06-01", tz="UTC")
        cov_end = pd.Timestamp("2024-06-07 23:00", tz="UTC")

        location = LocationInfo(
            latitude=51.5136,
            longitude=7.4653,
            timezone="UTC",
        )
        features = get_day_night_features(
            start=start,
            cov_end=cov_end,
            location=location,
            freq="h",
            timezone="UTC",
        )
        print("shape:", features.shape)
        print("columns:", features.columns.tolist())
        print(features.head(3))
        ```
    """
    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)

    # Cache sunrise and sunset times per unique calendar date to avoid
    # recomputing them for every timestamp in the extended_index.
    normalized_dates = extended_index.normalize()
    unique_dates = normalized_dates.unique()

    sunrise_map: dict = {}
    sunset_map: dict = {}
    for d in unique_dates:
        s = sun(location.observer, date=d, tzinfo=location.timezone)
        sunrise_map[d] = s["sunrise"]
        sunset_map[d] = s["sunset"]

    sunrise_series = pd.Series(
        [sunrise_map[d] for d in normalized_dates],
        index=extended_index,
    )
    sunset_series = pd.Series(
        [sunset_map[d] for d in normalized_dates],
        index=extended_index,
    )

    sunrise_hour = sunrise_series.dt.round("h").dt.hour
    sunset_hour = sunset_series.dt.round("h").dt.hour

    sun_light_features = pd.DataFrame(
        {
            "sunrise_hour": sunrise_hour,
            "sunset_hour": sunset_hour,
        }
    )
    sun_light_features["daylight_hours"] = (
        sun_light_features["sunset_hour"] - sun_light_features["sunrise_hour"]
    )
    sun_light_features["is_daylight"] = np.where(
        (extended_index.hour >= sun_light_features["sunrise_hour"])
        & (extended_index.hour < sun_light_features["sunset_hour"]),
        1,
        0,
    )

    return sun_light_features


def get_holiday_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Fetch public-holiday indicators and align them to a regular time grid.

    Downloads holiday data via
    :func:`~spotforecast2_safe.data.fetch_data.fetch_holiday_data`,
    validates coverage with
    :func:`~spotforecast2_safe.preprocessing.curate_data.curate_holidays`,
    and reindexes the result to a full ``[start, cov_end]`` grid with
    ``fill_value=0`` so that non-holiday timestamps are always zero.

    Args:
        data: Reference time series DataFrame used for temporal coverage
            validation inside
            :func:`~spotforecast2_safe.preprocessing.curate_data.curate_holidays`.
        start: Start timestamp.  String values are parsed with
            ``utc=True``.
        cov_end: Inclusive end timestamp (should cover the full forecast
            horizon).  String values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead; passed to
            :func:`~spotforecast2_safe.preprocessing.curate_data.curate_holidays`.
        tz: Timezone applied to the generated index and passed to
            :func:`~spotforecast2_safe.data.fetch_data.fetch_holiday_data`.
            Defaults to ``"UTC"``.
        freq: Pandas-compatible frequency string for the output index.
            Defaults to ``"h"`` (hourly).
        country_code: ISO 3166-1 alpha-2 country code.  Defaults to
            ``"DE"`` (Germany).
        state: Sub-national state/region code.  Defaults to ``"NW"``
            (North Rhine-Westphalia).

    Returns:
        pd.DataFrame: DataFrame with a single integer column
        ``is_holiday``.  The index is a tz-aware
        :class:`~pandas.DatetimeIndex` with the requested ``freq``.

    Examples:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        from spotforecast2_safe.preprocessing.curate_data import agg_and_resample_data
        from spotforecast2_safe.manager.exo.calendar import get_holiday_features

        # Minimal reference DataFrame for validation
        data = fetch_data(filename=str(get_package_data_home() / "demo10.csv"))
        data = agg_and_resample_data(data, verbose=False)

        start = pd.Timestamp("2024-01-01", tz="UTC")
        cov_end = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        holidays = get_holiday_features(
            data=data,
            start=start,
            cov_end=cov_end,
            forecast_horizon=24,
            country_code="DE",
            state="NW",
        )
        print("shape:", holidays.shape)
        print("columns:", holidays.columns.tolist())
        # New Year's Day (Jan 1) is a public holiday in Germany
        print("Jan 1 value:", holidays.loc["2024-01-01 00:00:00+00:00", "is_holiday"])
        ```
    """
    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    holiday_df = fetch_holiday_data(
        start=start,
        end=cov_end,
        tz=tz,
        freq=freq,
        country_code=country_code,
        state=state,
    )

    curate_holidays(holiday_df, data, forecast_horizon=forecast_horizon)

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=tz)
    holiday_features = holiday_df.reindex(extended_index, fill_value=0).astype(int)

    return holiday_features
