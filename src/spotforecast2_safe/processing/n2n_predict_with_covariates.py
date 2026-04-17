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
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from astral import LocationInfo
from lightgbm import LGBMRegressor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None

from spotforecast2_safe.data.fetch_data import fetch_data
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.forecaster.utils import predict_multivariate
from spotforecast2_safe.manager.exo.calendar import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_features,
)
from spotforecast2_safe.manager.exo.weather import get_weather_features
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    merge_data_and_covariates,
    select_exogenous_features,
)
from spotforecast2_safe.manager.persistence import (
    _load_forecasters,
    _model_directory_exists,
    _save_forecasters,
)
from spotforecast2_safe.preprocessing import RollingFeatures
from spotforecast2_safe.preprocessing.curate_data import (
    agg_and_resample_data,
    basic_ts_checks,
    get_start_end,
)
from spotforecast2_safe.preprocessing.imputation import get_missing_weights
from spotforecast2_safe.preprocessing.outlier import mark_outliers
from spotforecast2_safe.preprocessing.split import split_rel_train_val_test

# ============================================================================
# Helper Functions for Feature Engineering
# (public implementations live in spotforecast2_safe.manager.features)
# ============================================================================

# Private aliases kept for backward compatibility with existing callers
# that import the private names from this module.
_get_weather_features = get_weather_features
_get_calendar_features = get_calendar_features
_get_day_night_features = get_day_night_features
_get_holiday_features = get_holiday_features
_apply_cyclical_encoding = apply_cyclical_encoding
_create_interaction_features = create_interaction_features
_select_exogenous_features = select_exogenous_features
_merge_data_and_covariates = merge_data_and_covariates


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
