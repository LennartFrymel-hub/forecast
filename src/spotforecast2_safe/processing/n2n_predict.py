# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end baseline forecasting using equivalent date method.

This module provides a complete forecasting pipeline using the ForecasterEquivalentDate
baseline model. It handles data preparation, outlier detection, imputation, model
training, and prediction in a single integrated function.

Model persistence follows scikit-learn conventions using joblib for efficient
serialization and deserialization of trained forecasters.

Examples:
    Basic usage with default parameters:

    >>> from spotforecast2_safe.processing.n2n_predict import n2n_predict
    >>> predictions = n2n_predict(forecast_horizon=24, verbose=True)

    Using cached models:

    >>> # Load existing models if available, or train new ones
    >>> predictions = n2n_predict(
    ...     forecast_horizon=24,
    ...     force_train=False,
    ...     model_dir="./models",
    ...     verbose=True
    ... )

    Force retraining and update cache:

    >>> predictions = n2n_predict(
    ...     forecast_horizon=24,
    ...     force_train=True,
    ...     model_dir="./models",
    ...     verbose=True
    ... )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate
from spotforecast2_safe.forecaster.utils import predict_multivariate
from spotforecast2_safe.manager.persistence import (
    _load_forecasters,
    _model_directory_exists,
    _save_forecasters,
)
from spotforecast2_safe.preprocessing.curate_data import (
    agg_and_resample_data,
    basic_ts_checks,
    get_start_end,
)
from spotforecast2_safe.preprocessing.outlier import mark_outliers
from spotforecast2_safe.preprocessing.split import split_rel_train_val_test

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None


# ============================================================================
# Main Function
# ============================================================================


def n2n_predict(
    data: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    forecast_horizon: int = 24,
    contamination: float = 0.01,
    window_size: int = 72,
    force_train: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """End-to-end baseline forecasting using equivalent date method.

    This function implements a complete forecasting pipeline that:
    1. Loads and validates target data
    2. Detects and removes outliers
    3. Imputes missing values
    4. Splits into train/validation/test sets
    5. Trains or loads equivalent date forecasters
    6. Generates multi-step ahead predictions

    Models are persisted to disk following scikit-learn conventions using joblib.
    By default, models are retrained (force_train=True). Set force_train=False to reuse existing cached models.

    Args:
        data: Optional DataFrame with target time series data. If None, fetches data automatically.
            Default: None.
        columns: List of target columns to forecast. If None, uses all available columns.
            Default: None.
        forecast_horizon: Number of time steps to forecast ahead. Default: 24.
        contamination: Contamination parameter for outlier detection. Default: 0.01.
        window_size: Rolling window size for gap detection. Default: 72.
        force_train: Force retraining of all models, ignoring cached models.
            Default: True.
        model_dir: Directory for saving/loading trained models. If None, uses cache directory from get_cache_home(). Default: None (uses ~/spotforecast2_cache/forecasters).
        verbose: Print progress messages. Default: True.
        show_progress: Show progress bar during training and prediction. Default: True.

    Returns:
        Tuple containing:
        - predictions: DataFrame with forecast values for each target variable.
        - forecasters: Dictionary of trained ForecasterEquivalentDate objects keyed by target.

    Raises:
        ValueError: If data validation fails or required data cannot be retrieved.
        ImportError: If required dependencies are not installed.
        OSError: If models cannot be saved to disk.

    Examples:
        Basic usage with automatic model caching:

        >>> predictions, forecasters = n2n_predict(
        ...     forecast_horizon=24,
        ...     verbose=True
        ... )
        >>> print(predictions.shape)
        (24, 11)

        Load cached models (if available):

        >>> predictions, forecasters = n2n_predict(
        ...     forecast_horizon=24,
        ...     force_train=False,
        ...     model_dir="./saved_models",
        ...     verbose=True
        ... )

        Force retraining and update cache:

        >>> predictions, forecasters = n2n_predict(
        ...     forecast_horizon=24,
        ...     force_train=True,
        ...     model_dir="./saved_models",
        ...     verbose=True
        ... )

        With specific target columns:

        >>> predictions, forecasters = n2n_predict(
        ...     columns=["power", "energy"],
        ...     forecast_horizon=48,
        ...     force_train=False,
        ...     verbose=True
        ... )

    Notes:
        - Trained models are saved to disk using joblib for fast reuse.
        - When force_train=False, existing models are loaded and prediction
          proceeds without retraining. This significantly speeds up prediction
          for repeated calls with the same configuration.
        - The model_dir directory is created automatically if it doesn't exist.
        - Default model_dir uses get_cache_home() which respects the
          SPOTFORECAST2_CACHE environment variable.

    Performance Notes:
        - First run: Full training (~2-5 minutes depending on data size)
        - Subsequent runs (force_train=False): Model loading only (~1-2 seconds)
        - Force retrain (force_train=True): Full training again (~2-5 minutes)
    """
    if columns is not None:
        TARGET = columns
    else:
        TARGET = None

    if verbose:
        print("--- Starting n2n_predict ---")

    # Set default model_dir if not provided
    if model_dir is None:
        from spotforecast2_safe.data.fetch_data import get_cache_home

        model_dir = get_cache_home() / "forecasters"

    # Handle data input - fetch_data handles both CSV and DataFrame
    if data is not None:
        if verbose:
            print("Using provided dataframe...")
        data = fetch_data(dataframe=data, columns=TARGET)
    else:
        if verbose:
            print("Fetching data from CSV...")
        data = fetch_data(filename="data_in.csv", columns=TARGET)

    _, _, _, _ = get_start_end(
        data=data,
        forecast_horizon=forecast_horizon,
        verbose=verbose,
    )

    basic_ts_checks(data, verbose=verbose)

    data = agg_and_resample_data(data, verbose=verbose)

    # --- Outlier Handling ---
    if verbose:
        print("Handling outliers...")

    # data_old = data.copy() # kept in notebook, maybe useful for debugging but not used logic-wise here
    data, outliers = mark_outliers(
        data, contamination=contamination, random_state=1234, verbose=verbose
    )

    # --- Missing Data (Imputation) ---
    if verbose:
        print("Imputing missing data...")

    missing_indices = data.index[data.isnull().any(axis=1)]
    if verbose:
        n_missing = len(missing_indices)
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")

    data = data.ffill()
    data = data.bfill()

    # --- Train, Val, Test Split ---
    if verbose:
        print("Splitting data...")
    data_train, data_val, data_test = split_rel_train_val_test(
        data, perc_train=0.8, perc_val=0.2, verbose=verbose
    )

    # --- Model Fit ---
    if verbose:
        print("Fitting models...")

    end_validation = pd.concat([data_train, data_val]).index[-1]

    baseline_forecasters = {}
    targets_to_train = list(data.columns)

    # Attempt to load cached models if force_train=False
    if not force_train and _model_directory_exists(model_dir):
        if verbose:
            print("  Attempting to load cached models...")
        cached_forecasters, missing_targets = _load_forecasters(
            target_columns=list(data.columns),
            model_dir=model_dir,
            verbose=verbose,
        )
        baseline_forecasters.update(cached_forecasters)
        targets_to_train = missing_targets

        if len(cached_forecasters) == len(data.columns):
            if verbose:
                print(f"  ✓ All {len(data.columns)} forecasters loaded from cache")
        elif len(cached_forecasters) > 0:
            if verbose:
                print(
                    f"  ✓ Loaded {len(cached_forecasters)} forecasters, "
                    f"will train {len(targets_to_train)} new ones"
                )

    # Train missing or forced models
    if len(targets_to_train) > 0:
        if force_train and len(baseline_forecasters) > 0:
            if verbose:
                print(f"  Force retraining all {len(data.columns)} forecasters...")
            targets_to_train = list(data.columns)
            baseline_forecasters.clear()

        target_iter = targets_to_train
        if show_progress and tqdm is not None:
            target_iter = tqdm(
                targets_to_train,
                desc="Training forecasters",
                unit="model",
            )

        for target in target_iter:
            forecaster = ForecasterEquivalentDate(
                offset=pd.DateOffset(days=1), n_offsets=1
            )

            forecaster.fit(y=data.loc[:end_validation, target])

            baseline_forecasters[target] = forecaster

        # Save newly trained models to disk
        if verbose:
            print(f"  Saving {len(targets_to_train)} trained forecasters to disk...")
        _save_forecasters(
            forecasters={t: baseline_forecasters[t] for t in targets_to_train},
            model_dir=model_dir,
            verbose=verbose,
        )

    if verbose:
        print(f"  ✓ Total forecasters available: {len(baseline_forecasters)}")

    # --- Predict ---
    if verbose:
        print("Generating predictions...")

    predictions = predict_multivariate(
        baseline_forecasters,
        steps_ahead=forecast_horizon,
        show_progress=show_progress,
    )

    return predictions, baseline_forecasters
