# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end recursive forecasting with exogenous covariates.

This module provides a complete pipeline for time series forecasting using
recursive forecasters with exogenous variables (weather, holidays, calendar features).
It handles data preparation, feature engineering, model training, and prediction
in a single integrated function.

Model persistence follows scikit-learn conventions using joblib for efficient
serialization and deserialization of trained forecasters.

Examples:
    Basic usage with default parameters:

    >>> from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    ...     n2n_predict_with_covariates
    ... )
    >>> predictions = n2n_predict_with_covariates(
    ...     forecast_horizon=24,
    ...     verbose=True
    ... )

    With custom parameters:

    >>> predictions = n2n_predict_with_covariates(
    ...     forecast_horizon=48,
    ...     contamination=0.02,
    ...     window_size=100,
    ...     lags=48,
    ...     train_ratio=0.75,
    ...     verbose=True
    ... )

    Using cached models:

    >>> # Load existing models if available, or train new ones
    >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
    ...     forecast_horizon=24,
    ...     force_train=False,
    ...     model_dir="./models",
    ...     verbose=True
    ... )

    Force retraining and update cache:

    >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
    ...     forecast_horizon=24,
    ...     force_train=True,
    ...     model_dir="./models",
    ...     verbose=True
    ... )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astral import LocationInfo
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None

from spotforecast2_safe.data.fetch_data import (
    fetch_data,
    fetch_holiday_data,
    fetch_weather_data,
)
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.forecaster.utils import predict_multivariate
from spotforecast2_safe.preprocessing import RollingFeatures
from spotforecast2_safe.preprocessing.curate_data import (
    agg_and_resample_data,
    basic_ts_checks,
    curate_holidays,
    curate_weather,
    get_start_end,
)
from spotforecast2_safe.preprocessing.imputation import get_missing_weights
from spotforecast2_safe.preprocessing.outlier import mark_outliers
from spotforecast2_safe.preprocessing.split import split_rel_train_val_test
from spotforecast2_safe.manager.persistence import (
    _save_forecasters,
    _load_forecasters,
    _model_directory_exists,
)

try:
    from feature_engine.creation import CyclicalFeatures
    from feature_engine.datetime import DatetimeFeatures
    from feature_engine.timeseries.forecasting import WindowFeatures
except ImportError:
    raise ImportError(
        "feature_engine is required. Install with: pip install feature-engine"
    )

try:
    from astral.sun import sun
except ImportError:
    raise ImportError("astral is required. Install with: pip install astral")


# ============================================================================
# Helper Functions for Feature Engineering
# ============================================================================


def _get_weather_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    freq: str = "h",
    window_periods: Optional[List[str]] = None,
    window_functions: Optional[List[str]] = None,
    fallback_on_failure: bool = True,
    cached: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process weather data with rolling window features.

    Args:
        data: Time series DataFrame for validation.
        start: Start date for weather data.
        cov_end: End date for weather data.
        forecast_horizon: Number of forecast steps.
        latitude: Latitude of location. Default: 51.5136 (Dortmund).
        longitude: Longitude of location. Default: 7.4653 (Dortmund).
        timezone: Timezone for data. Default: "UTC".
        freq: Frequency of time series. Default: "h".
        window_periods: Window periods for rolling features. Default: ["1D", "7D"].
        window_functions: Functions for rolling windows. Default: ["mean", "max", "min"].
        fallback_on_failure: Use fallback if API fails. Default: True.
        cached: Use cached data if available. Default: True.
        verbose: Print progress. Default: False.

    Returns:
        Tuple of (weather_features, weather_aligned).
    """
    if window_periods is None:
        window_periods = ["1D", "7D"]
    if window_functions is None:
        window_functions = ["mean", "max", "min"]

    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    if verbose:
        print("Fetching weather data...")

    weather_df = fetch_weather_data(
        cov_start=start,
        cov_end=cov_end,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        freq=freq,
        fallback_on_failure=fallback_on_failure,
        cached=cached,
    )

    curate_weather(weather_df, data, forecast_horizon=forecast_horizon)

    if verbose:
        print("Processing weather features...")

    extended_index = pd.date_range(start=start, end=cov_end, freq=freq, tz=timezone)
    weather_aligned = weather_df.reindex(extended_index, method="ffill")

    weather_columns = weather_aligned.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(weather_columns) == 0:
        raise ValueError("No numeric weather columns found")

    weather_aligned_filled = weather_aligned[weather_columns].copy()
    if weather_aligned_filled.isnull().any().any():
        weather_aligned_filled = weather_aligned_filled.bfill()
        if weather_aligned_filled.isnull().any().any():
            raise ValueError("Missing values in weather data could not be filled")

    wf_transformer = WindowFeatures(
        variables=weather_columns,
        window=window_periods,
        functions=window_functions,
        freq=freq,
    )

    weather_features = wf_transformer.fit_transform(weather_aligned_filled)

    if weather_features.isnull().any().any():
        weather_features = weather_features.bfill()
        if weather_features.isnull().any().any():
            raise ValueError("Missing values in weather features could not be filled")

    if verbose:
        print(f"Weather features shape: {weather_features.shape}")

    return weather_features, weather_aligned


def _get_calendar_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    freq: str = "h",
    timezone: str = "UTC",
    features_to_extract: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create calendar-based features for a time range.

    Args:
        start: Start date.
        cov_end: End date.
        freq: Frequency. Default: "h".
        timezone: Timezone. Default: "UTC".
        features_to_extract: Features to extract. Default: ["month", "week", "day_of_week", "hour"].

    Returns:
        DataFrame with calendar features.
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


def _get_day_night_features(
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    location: LocationInfo,
    freq: str = "h",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Create day/night features using sunrise and sunset times.

    Args:
        start: Start date.
        cov_end: End date.
        location: Astral LocationInfo object.
        freq: Frequency. Default: "h".
        timezone: Timezone. Default: "UTC".

    Returns:
        DataFrame with sunrise/sunset and daylight features.
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

    sunrise_map = {}
    sunset_map = {}
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


def _get_holiday_features(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Fetch and align holiday features to the extended time index.

    Args:
        data: Target time series for validation.
        start: Start timestamp.
        cov_end: End timestamp.
        forecast_horizon: Number of forecast steps.
        tz: Timezone. Default: "UTC".
        freq: Frequency. Default: "h".
        country_code: Country code. Default: "DE".
        state: State code. Default: "NW".

    Returns:
        DataFrame with holiday features.
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


def _apply_cyclical_encoding(
    data: pd.DataFrame,
    features_to_encode: Optional[List[str]] = None,
    max_values: Optional[Dict[str, int]] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Apply cyclical encoding to selected features.

    Args:
        data: DataFrame with features.
        features_to_encode: Features to encode. Default: calendar and sun features.
        max_values: Max values for features. Default: standard calendar/hour ranges.
        drop_original: Drop original columns. Default: False.

    Returns:
        DataFrame with cyclical encoded features.
    """
    if features_to_encode is None:
        features_to_encode = [
            "month",
            "week",
            "day_of_week",
            "hour",
            "sunrise_hour",
            "sunset_hour",
        ]

    if max_values is None:
        max_values = {
            "month": 12,
            "week": 52,
            "day_of_week": 6,
            "hour": 24,
            "sunrise_hour": 24,
            "sunset_hour": 24,
        }

    # Filter features_to_encode to only those that exist in data
    available_features = [f for f in features_to_encode if f in data.columns]
    available_max_values = {
        k: v for k, v in max_values.items() if k in available_features
    }

    cyclical_encoder = CyclicalFeatures(
        variables=available_features,
        max_values=available_max_values,
        drop_original=drop_original,
    )

    return cyclical_encoder.fit_transform(data)


def _create_interaction_features(
    exogenous_features: pd.DataFrame,
    weather_aligned: pd.DataFrame,
    base_cols: Optional[List[str]] = None,
    weather_window_pattern: str = "_window_",
    include_weather_funcs: Optional[List[str]] = None,
    holiday_col: str = "is_holiday",
    degree: int = 1,
) -> pd.DataFrame:
    """Create interaction features from exogenous features.

    Args:
        exogenous_features: DataFrame with base features.
        weather_aligned: DataFrame with raw weather columns.
        base_cols: Base columns for interactions. Default: day_of_week and hour cyclical features.
        weather_window_pattern: Pattern for weather window features. Default: "_window_".
        include_weather_funcs: Functions to include. Default: ["_mean", "_min", "_max"].
        holiday_col: Holiday column name. Default: "holiday".
        degree: Polynomial degree. Default: 1.

    Returns:
        DataFrame with interaction features appended.
    """
    if base_cols is None:
        base_cols = [
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_sin",
            "hour_cos",
        ]

    if include_weather_funcs is None:
        include_weather_funcs = ["_mean", "_min", "_max"]

    transformer_poly = PolynomialFeatures(
        degree=degree, interaction_only=True, include_bias=False
    )
    transformer_poly = transformer_poly.set_output(transform="pandas")

    weather_window_cols = [
        col
        for col in exogenous_features.columns
        if weather_window_pattern in col
        and any(func in col for func in include_weather_funcs)
    ]

    raw_weather_cols = [
        col
        for col in exogenous_features.columns
        if col in weather_aligned.columns and col not in weather_window_cols
    ]

    poly_cols = list(base_cols)
    poly_cols.extend(weather_window_cols)
    poly_cols.extend(raw_weather_cols)
    if holiday_col in exogenous_features.columns:
        poly_cols.append(holiday_col)

    poly_features = transformer_poly.fit_transform(exogenous_features[poly_cols])
    poly_features = poly_features.drop(columns=poly_cols)
    poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
    poly_features.columns = poly_features.columns.str.replace(" ", "__")

    return pd.concat([exogenous_features, poly_features], axis=1)


def _select_exogenous_features(
    exogenous_features: pd.DataFrame,
    weather_aligned: pd.DataFrame,
    cyclical_regex: str = "_sin$|_cos$",
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
) -> List[str]:
    """Select exogenous feature columns for model training.

    Args:
        exogenous_features: DataFrame with all features.
        weather_aligned: DataFrame with raw weather columns.
        cyclical_regex: Regex for cyclical features. Default: "_sin$|_cos$".
        include_weather_windows: Include weather window features. Default: False.
        include_holiday_features: Include holiday features. Default: False.
        include_poly_features: Include polynomial features. Default: False.

    Returns:
        List of selected feature column names.
    """
    exog_features: List[str] = []

    exog_features.extend(
        exogenous_features.filter(regex=cyclical_regex).columns.tolist()
    )

    if include_weather_windows:
        weather_window_features = [
            col
            for col in exogenous_features.columns
            if "_window_" in col and ("_mean" in col or "_min" in col or "_max" in col)
        ]
        exog_features.extend(weather_window_features)

    raw_weather_features = [
        col for col in exogenous_features.columns if col in weather_aligned.columns
    ]
    exog_features.extend(raw_weather_features)

    if include_holiday_features:
        holiday_related = [
            col for col in exogenous_features.columns if col.startswith("holiday")
        ]
        exog_features.extend(holiday_related)

    if include_poly_features:
        poly_features_list = [
            col for col in exogenous_features.columns if col.startswith("poly_")
        ]
        exog_features.extend(poly_features_list)

    return list(dict.fromkeys(exog_features))


def _merge_data_and_covariates(
    data: pd.DataFrame,
    exogenous_features: pd.DataFrame,
    target_columns: List[str],
    exog_features: List[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    cast_dtype: Optional[str] = "float32",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge target data with exogenous features and build prediction covariates.

    Args:
        data: DataFrame with target variables.
        exogenous_features: DataFrame with exogenous features.
        target_columns: Target column names.
        exog_features: Exogenous feature column names.
        start: Start date.
        end: End date.
        cov_end: Covariate end date.
        forecast_horizon: Number of forecast steps.
        cast_dtype: Data type for merged data. Default: "float32".

    Returns:
        Tuple of (data_with_exog, exo_tmp, exo_pred).
    """
    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(end, str):
        end = pd.to_datetime(end, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    exo_tmp = exogenous_features.loc[start:end].copy()
    exo_pred = exogenous_features.loc[end + pd.Timedelta(hours=1) : cov_end].copy()

    data_with_exog = data[target_columns].merge(
        exo_tmp[exog_features],
        left_index=True,
        right_index=True,
        how="inner",
    )

    if cast_dtype is not None:
        data_with_exog = data_with_exog.astype(cast_dtype)

    return data_with_exog, exo_tmp, exo_pred


# ============================================================================
# Model Persistence Functions
# imported from spotforecast2_safe.manager.persistence
# ============================================================================


# ============================================================================
# Main Function
# ============================================================================


def n2n_predict_with_covariates(
    data: Optional[pd.DataFrame] = None,
    forecast_horizon: int = 24,
    contamination: float = 0.01,
    window_size: int = 72,
    lags: int = 24,
    train_ratio: float = 0.8,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    country_code: str = "DE",
    state: str = "NW",
    estimator: Optional[object] = None,
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
    force_train: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """End-to-end recursive forecasting with exogenous covariates.

    This function implements a complete forecasting pipeline that:
    1. Loads and validates target data
    2. Detects and removes outliers
    3. Imputes missing values with weighted gaps
    4. Creates exogenous features (weather, holidays, calendar, day/night)
    5. Performs feature engineering (cyclical encoding, interactions)
    6. Merges target and exogenous data
    7. Splits into train/validation/test sets
    8. Trains or loads recursive forecasters with sample weighting
    9. Generates multi-step ahead predictions

    Models are persisted to disk following scikit-learn conventions using joblib.
    By default, models are retrained (force_train=True). Set force_train=False to reuse existing cached models.

    Args:
        data: Optional DataFrame with target time series data. If None, fetches data automatically.
            Default: None.
        forecast_horizon: Number of time steps to forecast ahead. Default: 24.
        contamination: Contamination parameter for outlier detection. Default: 0.01.
        window_size: Rolling window size for gap detection. Default: 72.
        lags: Number of lags for recursive forecaster. Default: 24.
        train_ratio: Fraction of data for training. Default: 0.8.
        latitude: Location latitude. Default: 51.5136 (Dortmund).
        longitude: Location longitude. Default: 7.4653 (Dortmund).
        timezone: Timezone for data. Default: "UTC".
        country_code: Country code for holidays. Default: "DE".
        state: State code for holidays. Default: "NW".
        estimator: Base estimator for recursive forecaster.
            If None, uses LGBMRegressor. Default: None.
        include_weather_windows: Include weather window features. Default: False.
        include_holiday_features: Include holiday features. Default: False.
        include_poly_features: Include polynomial interaction features. Default: False.
        force_train: Force retraining of all models, ignoring cached models.
            Default: True.
        model_dir: Directory for saving/loading trained models. If None, uses the
            spotforecast2 cache directory (~/spotforecast2_cache by default, or
            SPOTFORECAST2_CACHE environment variable). Default: None.
        verbose: Print progress messages. Default: True.
        show_progress: Show progress bar during training. Default: False.

    Returns:
        Tuple containing:
        - predictions: DataFrame with forecast values for each target variable.
        - metadata: Dictionary with forecast metadata (index, shapes, etc.).
        - forecasters: Dictionary of trained ForecasterRecursive objects keyed by target.

    Raises:
        ValueError: If data validation fails or required data cannot be retrieved.
        ImportError: If required dependencies are not installed.
        OSError: If models cannot be saved to disk.

    Examples:
        Basic usage with automatic model caching:

        >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
        ...     forecast_horizon=24,
        ...     verbose=True
        ... )
        >>> print(predictions.shape)
        (24, 11)

        Load cached models (if available):

        >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
        ...     forecast_horizon=24,
        ...     force_train=False,
        ...     model_dir="./saved_models"
        ... )

        Force retraining and update cache:

        >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
        ...     forecast_horizon=24,
        ...     force_train=True,
        ...     model_dir="./saved_models"
        ... )

        Custom location and features:

        >>> predictions, metadata, forecasters = n2n_predict_with_covariates(
        ...     forecast_horizon=48,
        ...     latitude=52.5200,  # Berlin
        ...     longitude=13.4050,
        ...     lags=48,
        ...     include_poly_features=True,
        ...     force_train=False,
        ...     verbose=True
        ... )

    Notes:
        - The function uses cached weather data when available.
        - Missing values are handled via forward/backward fill with downweighting
          observations near gaps.
        - Sample weights are passed to the forecaster to penalize observations
          near missing data.
        - Train/validation splits are temporal (80/20 by default).
        - All features are cast to float32 for memory efficiency.
        - Trained models are saved to disk using joblib for fast reuse.
        - When force_train=False, existing models are loaded and prediction
          proceeds without retraining. This significantly speeds up prediction
          for repeated calls with the same configuration.
        - The model_dir directory is created automatically if it doesn't exist.
        - By default, models are cached in ~/spotforecast2_cache, which can be
          customized via the SPOTFORECAST2_CACHE environment variable.

    Performance Notes:
        - First run: Full training
        - Subsequent runs (force_train=False): Model loading only
        - Force retrain (force_train=True): Full training again
    """
    # Set default model_dir if not provided
    if model_dir is None:
        from spotforecast2_safe.data.fetch_data import get_cache_home

        model_dir = get_cache_home() / "forecasters"

    # Input Validation
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
    if not 0 <= contamination <= 0.5:
        raise ValueError(
            f"contamination must be between 0 and 0.5, got {contamination}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if lags <= 0:
        raise ValueError(f"lags must be positive, got {lags}")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if verbose:
        print("=" * 80)
        print("N2N Recursive Forecasting with Exogenous Covariates")
        print("=" * 80)

    # ========================================================================
    # 1. DATA PREPARATION
    # ========================================================================

    if verbose:
        print("\n[1/9] Loading and preparing target data...")

    # Handle data input - fetch_data handles both CSV and DataFrame
    if data is None:
        if verbose:
            print("  Fetching data from CSV...")
        data = fetch_data(filename="data_in.csv", timezone=timezone)
    else:
        if verbose:
            print("  Using provided dataframe...")
        data = fetch_data(dataframe=data, timezone=timezone)

    target_columns = data.columns.tolist()

    if verbose:
        print(f"  Target variables: {target_columns}")

    start, end, cov_start, cov_end = get_start_end(
        data=data,
        forecast_horizon=forecast_horizon,
        verbose=verbose,
    )

    basic_ts_checks(data, verbose=verbose)
    data = agg_and_resample_data(data, verbose=verbose)

    # ========================================================================
    # 2. OUTLIER DETECTION AND REMOVAL
    # ========================================================================

    if verbose:
        print("\n[2/9] Detecting and marking outliers...")

    data, outliers = mark_outliers(
        data,
        contamination=contamination,
        random_state=1234,
        verbose=verbose,
    )

    # ========================================================================
    # 3. MISSING VALUE IMPUTATION WITH WEIGHTING
    # ========================================================================

    if verbose:
        print("\n[3/9] Processing missing values and creating sample weights...")

    imputed_data, weights_series = get_missing_weights(
        data, window_size=window_size, verbose=verbose
    )

    # Create weight function for forecaster
    # Weights are already directly usable: 1.0 (valid/default), 0.0 (near gap)

    # Use WeightFunction class which is picklable (unlike local functions with closures)
    from spotforecast2_safe.preprocessing import WeightFunction

    weight_func = WeightFunction(weights_series)

    # Model persistence enabled: WeightFunction instances can be pickled
    use_model_persistence = True

    # ========================================================================
    # 4. EXOGENOUS FEATURES ENGINEERING
    # ========================================================================

    if verbose:
        print("\n[4/9] Creating exogenous features...")

    # Location for day/night features
    location = LocationInfo(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )

    # Holidays
    holiday_features = _get_holiday_features(
        data=imputed_data,
        start=start,
        cov_end=cov_end,
        forecast_horizon=forecast_horizon,
        tz=timezone,
        freq="h",
        country_code=country_code,
        state=state,
    )

    # Weather
    weather_features, weather_aligned = _get_weather_features(
        data=imputed_data,
        start=start,
        cov_end=cov_end,
        forecast_horizon=forecast_horizon,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        freq="h",
        verbose=verbose,
    )

    # Calendar
    calendar_features = _get_calendar_features(
        start=start,
        cov_end=cov_end,
        freq="h",
        timezone=timezone,
    )

    # Day/night
    sun_light_features = _get_day_night_features(
        start=start,
        cov_end=cov_end,
        location=location,
        freq="h",
        timezone=timezone,
    )

    # ========================================================================
    # 5. COMBINE EXOGENOUS FEATURES
    # ========================================================================

    if verbose:
        print("\n[5/9] Combining and encoding exogenous features...")

    exogenous_features = pd.concat(
        [
            calendar_features,
            sun_light_features,
            weather_features,
            holiday_features,
        ],
        axis=1,
    )

    missing_count = exogenous_features.isnull().sum().sum()
    if missing_count != 0:
        raise ValueError(
            f"Missing values in exogenous features: {missing_count} missing entries"
        )

    # Apply cyclical encoding
    exogenous_features = _apply_cyclical_encoding(
        data=exogenous_features,
        drop_original=False,
    )

    # Create interactions
    exogenous_features = _create_interaction_features(
        exogenous_features=exogenous_features,
        weather_aligned=weather_aligned,
    )

    # ========================================================================
    # 6. SELECT EXOGENOUS FEATURES
    # ========================================================================

    exog_features = _select_exogenous_features(
        exogenous_features=exogenous_features,
        weather_aligned=weather_aligned,
        include_weather_windows=include_weather_windows,
        include_holiday_features=include_holiday_features,
        include_poly_features=include_poly_features,
    )

    if verbose:
        print(f"  Selected {len(exog_features)} exogenous features")

    # ========================================================================
    # 7. MERGE DATA AND COVARIATES
    # ========================================================================

    if verbose:
        print("\n[6/9] Merging target and exogenous data...")

    data_with_exog, exo_tmp, exo_pred = _merge_data_and_covariates(
        data=imputed_data,
        exogenous_features=exogenous_features,
        target_columns=target_columns,
        exog_features=exog_features,
        start=start,
        end=end,
        cov_end=cov_end,
        forecast_horizon=forecast_horizon,
        cast_dtype="float32",
    )

    if verbose:
        print(f"  Merged data shape: {data_with_exog.shape}")
        print(f"  Exogenous prediction shape: {exo_pred.shape}")

    # ========================================================================
    # 8. TRAIN/VALIDATION/TEST SPLIT
    # ========================================================================

    if verbose:
        print("\n[7/9] Splitting data into train/validation/test...")

    perc_val = 1.0 - train_ratio
    data_train, data_val, data_test = split_rel_train_val_test(
        data_with_exog,
        perc_train=train_ratio,
        perc_val=perc_val,
        verbose=verbose,
    )

    # ========================================================================
    # 9. MODEL TRAINING OR LOADING
    # ========================================================================

    if verbose:
        print(
            "\n[8/9] Loading or training recursive forecasters with exogenous variables..."
        )

    if estimator is None:
        estimator = LGBMRegressor(random_state=1234, verbose=-1)

    window_features = RollingFeatures(stats=["mean"], window_sizes=window_size)
    end_validation = pd.concat([data_train, data_val]).index[-1]

    # Attempt to load cached models if force_train=False and persistence is enabled
    recursive_forecasters = {}
    targets_to_train = target_columns

    if use_model_persistence and not force_train and _model_directory_exists(model_dir):
        if verbose:
            print("  Attempting to load cached models...")
        cached_forecasters, missing_targets = _load_forecasters(
            target_columns=target_columns,
            model_dir=model_dir,
            verbose=verbose,
        )
        recursive_forecasters.update(cached_forecasters)
        targets_to_train = missing_targets

        if len(cached_forecasters) == len(target_columns):
            if verbose:
                print(f"  ✓ All {len(target_columns)} forecasters loaded from cache")
        elif len(cached_forecasters) > 0:
            if verbose:
                print(
                    f"  ✓ Loaded {len(cached_forecasters)} forecasters, "
                    f"will train {len(targets_to_train)} new ones"
                )

    # Train missing or forced models
    if len(targets_to_train) > 0:
        if force_train and len(recursive_forecasters) > 0:
            if verbose:
                print(f"  Force retraining all {len(target_columns)} forecasters...")
            targets_to_train = target_columns
            recursive_forecasters.clear()

        target_iter = targets_to_train
        if show_progress and tqdm is not None:
            target_iter = tqdm(
                targets_to_train,
                desc="Training forecasters",
                unit="model",
            )

        for target in target_iter:
            if verbose:
                print(f"  Training forecaster for {target}...")

            forecaster = ForecasterRecursive(
                estimator=estimator,
                lags=lags,
                window_features=window_features,
                weight_func=weight_func,
            )

            forecaster.fit(
                y=data_with_exog[target].loc[:end_validation].squeeze(),
                exog=data_with_exog[exog_features].loc[:end_validation],
            )

            recursive_forecasters[target] = forecaster

            if verbose:
                print(f"    ✓ Forecaster trained for {target}")

        # Save newly trained models to disk (only if persistence is enabled)
        if use_model_persistence:
            if verbose:
                print(
                    f"  Saving {len(targets_to_train)} trained forecasters to disk..."
                )
            _save_forecasters(
                forecasters={t: recursive_forecasters[t] for t in targets_to_train},
                model_dir=model_dir,
                verbose=verbose,
            )

    if verbose:
        print(f"  ✓ Total forecasters available: {len(recursive_forecasters)}")

    # ========================================================================
    # 10. PREDICTION
    # ========================================================================

    if verbose:
        print("\n[9/9] Generating predictions...")

    exo_pred_subset = exo_pred[exog_features]

    predictions = predict_multivariate(
        recursive_forecasters,
        steps_ahead=forecast_horizon,
        exog=exo_pred_subset,
        show_progress=show_progress,
    )

    if verbose:
        print(f"  Predictions shape: {predictions.shape}")
        print("\n" + "=" * 80)
        print("Forecasting completed successfully!")
        print("=" * 80)

    # ========================================================================
    # COMPILE METADATA
    # ========================================================================

    metadata = {
        "forecast_horizon": forecast_horizon,
        "target_columns": target_columns,
        "exog_features": exog_features,
        "n_exog_features": len(exog_features),
        "train_size": len(data_train),
        "val_size": len(data_val),
        "test_size": len(data_test),
        "data_shape_original": data.shape,
        "data_shape_merged": data_with_exog.shape,
        "training_end": end_validation,
        "prediction_start": exo_pred.index[0],
        "prediction_end": exo_pred.index[-1],
        "lags": lags,
        "window_size": window_size,
        "contamination": contamination,
        "n_outliers": (
            outliers.sum() if isinstance(outliers, pd.Series) else len(outliers)
        ),
    }

    return predictions, metadata, recursive_forecasters
