# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_weather_data

try:
    from feature_engine.timeseries.forecasting import WindowFeatures
except ImportError:  # pragma: no cover
    raise ImportError(
        "feature_engine is required. Install with: pip install feature-engine"
    )


WeatherLocation = Dict[str, Union[str, float]]


def _safe_location_name(name: str) -> str:
    """Convert location name to a safe column prefix."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")


def _to_timezone(value: Union[str, pd.Timestamp], timezone: str) -> pd.Timestamp:
    """Convert timestamp-like input to the requested timezone."""
    ts = pd.to_datetime(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone)
    else:
        ts = ts.tz_convert(timezone)

    return ts


def _normalize_weather_index(
    weather_df: pd.DataFrame,
    timezone: str,
    location_name: str,
) -> pd.DataFrame:
    """Ensure weather index is sorted, unique and timezone-aware."""
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Weather data for location '{location_name}' must have a DatetimeIndex."
        )

    weather_df = weather_df.copy()

    if weather_df.index.tz is None:
        weather_df.index = weather_df.index.tz_localize(timezone)
    else:
        weather_df.index = weather_df.index.tz_convert(timezone)

    weather_df = weather_df.sort_index()
    weather_df = weather_df[~weather_df.index.duplicated(keep="last")]

    return weather_df

def validate_multi_weather_cache(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    locations: List[WeatherLocation],
    timezone: str = "UTC",
    freq: str = "h",
    cache_home: Optional[Union[str, Path]] = None,
    expected_columns: Optional[List[str]] = None,
    allow_missing_before_fill: bool = True,
    raise_on_error: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Validate that multi-location weather cache covers the required period.

    This function is a preflight check. It should be called before training or
    forecasting starts.

    It checks:
    - cache folder exists per location
    - weather data covers start to cov_end
    - expected weather columns exist
    - NaN values are detectable before and after filling

    Args:
        start: Required start timestamp.
        cov_end: Required end timestamp.
        locations: Weather locations.
        timezone: Timezone, normally "UTC".
        freq: Frequency, normally "h".
        cache_home: Root cache folder. Each location has its own subfolder.
        expected_columns: Required raw weather columns.
        allow_missing_before_fill: If False, raw NaN values fail validation.
            If True, only remaining NaN after ffill/bfill fail validation.
        raise_on_error: Raise ValueError if any location fails.
        verbose: Print validation table.

    Returns:
        DataFrame with one validation row per location.
    """
    if not locations:
        raise ValueError("locations must contain at least one weather location")

    start_ts = _to_timezone(start, timezone)
    cov_end_ts = _to_timezone(cov_end, timezone)

    if cov_end_ts < start_ts:
        raise ValueError(
            f"cov_end must be after start. Got start={start_ts}, cov_end={cov_end_ts}."
        )

    extended_index = pd.date_range(
        start=start_ts,
        end=cov_end_ts,
        freq=freq,
        tz=timezone,
    )

    expected_set = set(expected_columns or [])
    rows = []

    for loc in locations:
        name = _safe_location_name(str(loc["name"]))
        latitude = float(loc["latitude"])
        longitude = float(loc["longitude"])

        location_cache_home = None
        cache_exists = False

        if cache_home is not None:
            location_cache_home = Path(cache_home) / name
            cache_exists = location_cache_home.exists()

        row = {
            "location": name,
            "cache_path": str(location_cache_home) if location_cache_home else "",
            "cache_exists": cache_exists,
            "needed_start": str(extended_index.min()),
            "needed_end": str(extended_index.max()),
            "expected_rows": len(extended_index),
            "weather_start": "",
            "weather_end": "",
            "rows_after_alignment": 0,
            "numeric_columns": 0,
            "missing_columns": "",
            "missing_before_fill": np.nan,
            "missing_after_fill": np.nan,
            "ok": False,
            "error": "",
        }

        try:
            weather_df = fetch_weather_data(
                cov_start=start_ts,
                cov_end=cov_end_ts,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                freq=freq,
                fallback_on_failure=True,
                cache_home=location_cache_home,
                fill_missing=True,
            )

            weather_df = _normalize_weather_index(
                weather_df=weather_df,
                timezone=timezone,
                location_name=name,
            )

            row["weather_start"] = str(weather_df.index.min())
            row["weather_end"] = str(weather_df.index.max())

            if weather_df.empty:
                raise ValueError("Weather dataframe is empty.")

            if weather_df.index.min() > extended_index.min():
                raise ValueError(
                    f"Weather data starts too late. "
                    f"Needed from {extended_index.min()}, got {weather_df.index.min()}."
                )

            if weather_df.index.max() < extended_index.max():
                raise ValueError(
                    f"Weather data ends too early. "
                    f"Needed until {extended_index.max()}, got {weather_df.index.max()}."
                )

            weather_aligned = weather_df.reindex(extended_index)

            numeric_cols = weather_aligned.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            row["rows_after_alignment"] = len(weather_aligned)
            row["numeric_columns"] = len(numeric_cols)

            if not numeric_cols:
                raise ValueError("No numeric weather columns found.")

            missing_columns = sorted(expected_set - set(numeric_cols))
            row["missing_columns"] = ",".join(missing_columns)

            weather_numeric = weather_aligned[numeric_cols].copy()

            missing_before_fill = int(weather_numeric.isna().sum().sum())
            row["missing_before_fill"] = missing_before_fill

            weather_filled = weather_numeric.ffill().bfill()

            missing_after_fill = int(weather_filled.isna().sum().sum())
            row["missing_after_fill"] = missing_after_fill

            has_required_shape = len(weather_filled) == len(extended_index)
            has_required_columns = len(missing_columns) == 0
            has_no_remaining_missing = missing_after_fill == 0
            raw_missing_is_allowed = allow_missing_before_fill or missing_before_fill == 0

            row["ok"] = bool(
                cache_exists
                and has_required_shape
                and has_required_columns
                and has_no_remaining_missing
                and raw_missing_is_allowed
            )

            if not row["ok"]:
                problems = []

                if not cache_exists:
                    problems.append("cache folder missing")

                if not has_required_shape:
                    problems.append("wrong row count after alignment")

                if not has_required_columns:
                    problems.append(f"missing columns: {missing_columns}")

                if not has_no_remaining_missing:
                    problems.append("NaN remains after ffill/bfill")

                if not raw_missing_is_allowed:
                    problems.append("raw NaN values found")

                row["error"] = "; ".join(problems)

        except Exception as exc:
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    result = pd.DataFrame(rows)

    if verbose:
        print()
        print("Weather cache validation:")
        print(result.to_string(index=False))

    if raise_on_error and (result.empty or not result["ok"].all()):
        failed = result.loc[~result["ok"]]
        raise ValueError(
            "Weather cache validation failed:\n"
            + failed.to_string(index=False)
        )

    return result


def get_multi_location_weather_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    locations: List[WeatherLocation],
    timezone: str = "UTC",
    freq: str = "h",
    window_periods: Optional[List[str]] = None,
    window_functions: Optional[List[str]] = None,
    fallback_on_failure: bool = True,
    cache_home: Optional[Union[str, Path]] = None,
    add_weighted_average: bool = True,
    keep_regional_features: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch weather data for multiple locations and build model-ready features.

    Args:
        data: Target time series. Kept for API compatibility.
        start: Start timestamp for covariates.
        cov_end: End timestamp for covariates, including forecast horizon.
        forecast_horizon: Forecast horizon in hours. Kept for API compatibility.
        locations: List of weather locations. Each item needs name, latitude,
            longitude and optionally weight.
        timezone: Timezone for returned weather data.
        freq: Frequency, normally "h".
        window_periods: Rolling windows, defaults to ["1D", "7D"].
        window_functions: Rolling functions, defaults to ["mean", "max", "min"].
        fallback_on_failure: Use cached fallback if API fails.
        cache_home: Optional cache root. Each location gets its own subfolder.
        add_weighted_average: Add weighted aggregate weather columns.
        keep_regional_features: Keep every city's raw weather features.
        verbose: Print progress.

    Returns:
        weather_features: Rolling weather features.
        weather_aligned_all: Raw aligned weather columns.
    """
    _ = data
    _ = forecast_horizon

    if not locations:
        raise ValueError("locations must contain at least one weather location")

    if window_periods is None:
        window_periods = ["1D", "7D"]

    if window_functions is None:
        window_functions = ["mean", "max", "min"]

    start_ts = _to_timezone(start, timezone)
    cov_end_ts = _to_timezone(cov_end, timezone)

    if cov_end_ts < start_ts:
        raise ValueError(
            f"cov_end must be after start. Got start={start_ts}, cov_end={cov_end_ts}."
        )

    extended_index = pd.date_range(
        start=start_ts,
        end=cov_end_ts,
        freq=freq,
        tz=timezone,
    )

    raw_by_location: Dict[str, pd.DataFrame] = {}
    weights: Dict[str, float] = {}
    aligned_frames: List[pd.DataFrame] = []

    for loc in locations:
        name = _safe_location_name(str(loc["name"]))
        latitude = float(loc["latitude"])
        longitude = float(loc["longitude"])
        weight = float(loc.get("weight", 1.0))

        if weight < 0:
            raise ValueError(f"Location weight must be >= 0 for {name}")

        if verbose:
            print(f"Fetching weather for location: {name}")

        location_cache_home = None
        if cache_home is not None:
            location_cache_home = Path(cache_home) / name

        try:
            weather_df = fetch_weather_data(
                cov_start=start_ts,
                cov_end=cov_end_ts,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                freq=freq,
                fallback_on_failure=fallback_on_failure,
                cache_home=location_cache_home,
                fill_missing=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch weather data for location '{name}'. "
                f"Reason: {type(exc).__name__}: {exc}. "
                "If this is an HTTP 429 rate limit, wait and run "
                "scripts\\08_prepare_weather_cache.py first."
            ) from exc

        weather_df = _normalize_weather_index(
            weather_df=weather_df,
            timezone=timezone,
            location_name=name,
        )

        if weather_df.empty:
            raise ValueError(f"Weather data for location '{name}' is empty.")

        if weather_df.index.min() > extended_index.min():
            raise ValueError(
                f"Weather data for location '{name}' starts too late. "
                f"Needed from {extended_index.min()}, got {weather_df.index.min()}."
            )

        if weather_df.index.max() < extended_index.max():
            raise ValueError(
                f"Weather data for location '{name}' ends too early. "
                f"Needed until {extended_index.max()}, got {weather_df.index.max()}. "
                "Extend the weather cache first."
            )

        weather_aligned = weather_df.reindex(extended_index)

        numeric_cols = weather_aligned.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric weather columns found for {name}")

        weather_aligned = weather_aligned[numeric_cols].copy()

        if len(weather_aligned) != len(extended_index):
            raise ValueError(
                f"Weather data for location '{name}' has wrong shape after alignment. "
                f"Expected {len(extended_index)} rows, got {len(weather_aligned)} rows."
            )

        if weather_aligned.isnull().any().any():
            weather_aligned = weather_aligned.ffill().bfill()

        if weather_aligned.isnull().any().any():
            missing_count = int(weather_aligned.isnull().sum().sum())
            raise ValueError(
                f"Missing weather values could not be filled for {name}. "
                f"Missing values: {missing_count}"
            )

        raw_by_location[name] = weather_aligned
        weights[name] = weight

        if keep_regional_features:
            aligned_frames.append(weather_aligned.add_prefix(f"{name}__"))

    if add_weighted_average:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("At least one weather location must have weight > 0")

        common_cols = set.intersection(
            *(set(frame.columns) for frame in raw_by_location.values())
        )

        if not common_cols:
            raise ValueError("No common weather columns found across locations.")

        aggregate_cols = {}
        for col in sorted(common_cols):
            aggregate_cols[f"de_weighted__{col}"] = sum(
                raw_by_location[name][col] * weights[name]
                for name in raw_by_location
            ) / total_weight

        aggregate_weather = pd.DataFrame(aggregate_cols, index=extended_index)
        aligned_frames.append(aggregate_weather)

    if not aligned_frames:
        raise ValueError(
            "No weather columns selected. Enable add_weighted_average "
            "or keep_regional_features."
        )

    weather_aligned_all = pd.concat(aligned_frames, axis=1)

    if weather_aligned_all.isnull().any().any():
        weather_aligned_all = weather_aligned_all.ffill().bfill()

    if weather_aligned_all.isnull().any().any():
        raise ValueError("Missing values in aligned multi-location weather data remain.")

    wf_transformer = WindowFeatures(
        variables=weather_aligned_all.columns.tolist(),
        window=window_periods,
        functions=window_functions,
        freq=freq,
    )

    weather_features = wf_transformer.fit_transform(weather_aligned_all)

    if weather_features.isnull().any().any():
        weather_features = weather_features.ffill().bfill()

    if weather_features.isnull().any().any():
        raise ValueError("Missing values in multi-location weather features remain.")

    if verbose:
        print(f"Multi-location weather raw shape: {weather_aligned_all.shape}")
        print(f"Multi-location weather features shape: {weather_features.shape}")

    return weather_features, weather_aligned_all