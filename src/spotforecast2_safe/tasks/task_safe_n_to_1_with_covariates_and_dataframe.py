# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
N-to-1 Forecasting with Exogenous Covariates and Prediction Aggregation.

This module implements a complete end-to-end pipeline for multi-step time series
forecasting with exogenous variables (weather, holidays, calendar features),
followed by prediction aggregation using configurable weights.

Logging Mechanism:
    This script implements a production-grade logging system designed for safety-critical
    environments:
    1.  **Console Handler**: Provides real-time progress updates to `stdout`.
    2.  **File Handler**: Automatically persists execution logs to a timestamped file
        in `~/spotforecast2_safe_models/logs/`.

    Log File Location:
        By default, logs are saved to `~/spotforecast2_safe_models/logs/task_safe_n_to_1_YYYYMMDD_HHMMSS.log`.

The pipeline:
    1. Performs multi-output recursive forecasting with exogenous covariates
    2. Aggregates predictions using weighted combinations
    3. Supports flexible model selection (string or object-based)
    4. Allows customization via kwargs for all underlying functions

Key Features:
    - Automatic weather, holiday, and calendar feature generation
    - Cyclical and polynomial feature engineering
    - Configurable recursive forecaster with LGBMRegressor default
    - Weighted prediction aggregation
    - Comprehensive parameter flexibility via **kwargs
    - Detailed logging and progress tracking
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
from spotforecast2_safe.manager.logger import setup_logging
from spotforecast2_safe.manager.tools import _parse_bool
from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)

# Default aggregation weights for the N-to-1 forecasting task.
#
# The position of each value corresponds to a specific forecast component or
# aggregation term used by `agg_predict`. Positive values increase the influence
# of the corresponding component in the final aggregated forecast, whereas
# negative values down-weight or invert the contribution of that component.
#
# NOTE:
# - These defaults are domain-specific and should be updated together with any
#   changes to the aggregation logic or the ordering of components in
#   `agg_predict`.
# - They are defined as a named constant (rather than inline) to make it clear
#   what is being tuned and to avoid unexplained "magic numbers" in the code.
DEFAULT_WEIGHTS: List[float] = [
    1.0,  # Index 0 – weight for forecast component 0
    1.0,  # Index 1 – weight for forecast component 1
    -1.0,  # Index 2 – weight for forecast component 2
    -1.0,  # Index 3 – weight for forecast component 3
    1.0,  # Index 4 – weight for forecast component 4
    -1.0,  # Index 5 – weight for forecast component 5
    1.0,  # Index 6 – weight for forecast component 6
    1.0,  # Index 7 – weight for forecast component 7
    1.0,  # Index 8 – weight for forecast component 8
    -1.0,  # Index 9 – weight for forecast component 9
    1.0,  # Index 10 – weight for forecast component 10
]

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


def _mask_latitude(lat: float) -> str:
    """Mask latitude to 1 decimal place for safe logging (security: CWE-532, CWE-312).

    Geographic precision beyond 1 decimal (~11km) is considered sensitive PII.
    This function rounds coordinates for INFO-level logging.

    Args:
        lat: Latitude value to mask.

    Returns:
        Masked latitude string representation (e.g., "51.5°N").
    """
    masked = round(lat, 1)
    direction = "N" if masked >= 0 else "S"
    return f"{abs(masked)}°{direction}"


def _mask_longitude(lon: float) -> str:
    """Mask longitude to 1 decimal place for safe logging (security: CWE-532, CWE-312).

    Geographic precision beyond 1 decimal (~11km) is considered sensitive PII.
    This function rounds coordinates for INFO-level logging.

    Args:
        lon: Longitude value to mask.

    Returns:
        Masked longitude string representation (e.g., "7.5°E").
    """
    masked = round(lon, 1)
    direction = "E" if masked >= 0 else "W"
    return f"{abs(masked)}°{direction}"


def _mask_estimator(estimator: Any) -> str:
    """Mask estimator details for safe logging (security: CWE-532).

    Logs estimator type but not configuration to avoid exposing model details.

    Args:
        estimator: Estimator object or name.

    Returns:
        Safe string representation of estimator type.
    """
    if estimator is None:
        return "LGBMRegressor (default)"
    if isinstance(estimator, str):
        return estimator
    return type(estimator).__name__


def n_to_1_with_covariates(
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
    estimator: Optional[Union[str, object]] = None,
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
    weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    verbose: bool = True,
    show_progress: bool = True,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict]:
    """Execute N-to-1 forecasting pipeline with exogenous covariates.

    This function performs a complete time series forecasting workflow:
    1. Fetches and preprocesses data
    2. Engineers features (calendar, weather, holidays, cyclical, polynomial)
    3. Trains recursive forecaster on multiple targets
    4. Aggregates predictions using weighted combination

    Security Note:
        Geographic coordinates (latitude/longitude) are considered sensitive PII
        (Personally Identifiable Information) per CWE-312 and CWE-532. This function
        implements data masking for all log output to prevent exposure in production
        monitoring systems, log aggregators, or crash dumps. Raw coordinate values
        are never logged at any log level, including DEBUG.

    Args:
        data (Optional[pd.DataFrame]): Optional DataFrame with target time series data.
            If None, fetches data automatically. Default: None.

        forecast_horizon (int): Number of forecast steps ahead.
            Determines how many time steps to predict into the future.
            Typical values: 24 (1 day), 48 (2 days), 168 (1 week). Default: 24.

        contamination (float): Outlier contamination level for anomaly detection.
            Expected proportion of outliers in the training data [0, 1].
            Higher values detect fewer outliers. Default: 0.01 (1%).

        window_size (int): Rolling window size for feature engineering (hours).
            Size of the rolling window for computing statistics.
            Must be > lags. Typical range: 24-168. Default: 72.

        lags (int): Number of lagged features to create.
            Creates AR(p) features with p=lags.
            Typical values: 12, 24, 48. Default: 24.

        train_ratio (float): Proportion of data for training [0, 1].
            Remaining data (1 - train_ratio) used for validation/testing.
            Typical values: 0.7-0.9. Default: 0.8.

        latitude (float): Geographic latitude for solar features.
            Used to compute sunrise/sunset times for day/night features.
            Default: 51.5136 (Dortmund, Germany).

        longitude (float): Geographic longitude for solar features.
            Used to compute sunrise/sunset times for day/night features.
            Default: 7.4653 (Dortmund, Germany).

        timezone (str): Timezone for time-based features.
            Any timezone recognized by pytz. Default: "UTC".

        country_code (str): ISO 3166-1 alpha-2 country code for holidays.
            Examples: "DE" (Germany), "US" (USA), "GB" (UK). Default: "DE".

        state (str): State/region code for holidays.
            Country-dependent. For Germany: "BW", "BY", "NW", etc.
            Default: "NW" (Nordrhein-Westfalen).

        estimator (Optional[Union[str, object]]): Forecaster model.
            Can be:
            - None: Uses LGBMRegressor(n_estimators=100, verbose=-1).
            - "ForecasterRecursive": References default estimator (same as None).
            - LGBMRegressor(...): Custom pre-configured estimator.
            - Any sklearn-compatible regressor.
            Default: None.

        include_weather_windows (bool): Add rolling weather statistics.
            Creates moving averages, min, max of weather features over
            multiple windows (1D, 7D). Increases feature count significantly.
            Default: False.

        include_holiday_features (bool): Add holiday binary indicators.
            Creates features indicating holidays and special dates.
            Useful for capturing demand patterns around holidays.
            Default: False.

        include_poly_features (bool): Add polynomial interactions.
            Creates 2nd-order interaction terms between selected features.
            Useful for capturing non-linear relationships.
            Default: False.

        weights (Optional[Union[Dict[str, float], List[float], np.ndarray]]):
            Weights for combining multi-output predictions.
            Can be:
            - None: Uses DEFAULT_WEIGHTS (see module-level constant for values)
            - Dict: {"col_name": weight, ...} for specific columns
            - List: [w1, w2, ...] in column order
            - np.ndarray: Same as list
            Default: None (uses DEFAULT_WEIGHTS).

        verbose (bool): Enable progress logging.
            Prints intermediate results and timestamps.
            Default: True.

        show_progress (bool): Show a progress bar for major pipeline steps.
            Default: True.

        **kwargs (Any): Additional parameters for underlying functions.
            These are passed to n2n_predict_with_covariates().
            Examples:
            - freq: Frequency for data resampling. Default: "h" (hourly).
            - columns: Specific columns to forecast. Default: None (all).
            Any parameter accepted by n2n_predict_with_covariates().

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict, Dict]: A tuple containing:
            - predictions (pd.DataFrame): Multi-output forecasts from recursive model.
                Each column represents a target variable.
                Index is datetime matching the forecast period.
            - combined_prediction (pd.Series): Aggregated forecast from weighted combination.
                Single column combining all output predictions.
                Index is datetime matching the forecast period.
            - model_metrics (Dict): Performance metrics from recursive forecaster.
                Keys may include: 'mae', 'rmse', 'mape', etc.
            - feature_info (Dict): Information about engineered features.
                Contains feature counts, types, and engineering details.

    Raises:
        ValueError: If forecast_horizon <= 0 or invalid parameter combinations.
        FileNotFoundError: If data source files cannot be accessed.
        RuntimeError: If model training fails or data processing errors occur.

    Examples:
        Basic usage (uses all defaults):

        >>> predictions, combined, metrics, features = n_to_1_with_covariates()
        >>> print(f"Predictions shape: {predictions.shape}")
        >>> print(f"Combined forecast head:\\n{combined.head()}")

        Custom location and forecast horizon:

        >>> predictions, combined, metrics, features = n_to_1_with_covariates(
        ...     forecast_horizon=48,
        ...     latitude=48.1351,
        ...     longitude=11.5820,
        ...     country_code="DE",
        ...     state="BY",
        ...     verbose=True
        ... )

        With feature engineering enabled:

        >>> predictions, combined, metrics, features = n_to_1_with_covariates(
        ...     forecast_horizon=24,
        ...     include_weather_windows=True,
        ...     include_holiday_features=True,
        ...     include_poly_features=True,
        ...     verbose=True
        ... )

        Custom estimator and weights:

        >>> from lightgbm import LGBMRegressor
        >>> custom_estimator = LGBMRegressor(
        ...     n_estimators=200,
        ...     learning_rate=0.01,
        ...     max_depth=7
        ... )
        >>> custom_weights = [1.0, 1.0, -0.5, -0.5]
        >>> predictions, combined, metrics, features = n_to_1_with_covariates(
        ...     forecast_horizon=24,
        ...     estimator=custom_estimator,
        ...     weights=custom_weights,
        ...     verbose=True
        ... )

        With all advanced options:

        >>> predictions, combined, metrics, features = n_to_1_with_covariates(
        ...     forecast_horizon=72,
        ...     contamination=0.02,
        ...     window_size=168,
        ...     lags=48,
        ...     train_ratio=0.75,
        ...     latitude=50.1109,
        ...     longitude=8.6821,
        ...     timezone="Europe/Berlin",
        ...     country_code="DE",
        ...     state="HE",
        ...     include_weather_windows=True,
        ...     include_holiday_features=True,
        ...     include_poly_features=True,
        ...     weights={"power": 1.0, "demand": 0.8},
        ...     verbose=True,
        ...     freq="h",
        ... )
        >>> print(f"Model Metrics: {metrics}")
        >>> print(f"Feature Info: {features}")
    """
    logger = logging.getLogger(__name__)

    # Security: Mask sensitive coordinates immediately (CWE-532, CWE-312)
    # This prevents raw latitude/longitude from being accessed in logging contexts

    masked_estimator = _mask_estimator(estimator)

    # Default weights if not provided
    if weights is None:
        # Use documented default aggregation weights instead of inline magic numbers.
        # Use a copy to avoid accidental mutation of the module-level default.
        weights = DEFAULT_WEIGHTS.copy()

    if verbose:
        logger.info("=" * 80)
        logger.info("N-to-1 Forecasting with Exogenous Covariates")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Forecast Horizon: {forecast_horizon} steps")
        logger.info(f"  Contamination Level: {contamination}")
        logger.info(f"  Window Size: {window_size}")
        logger.info(f"  Lags: {lags}")
        logger.info(f"  Train Ratio: {train_ratio}")
        # SECURITY: Never log latitude/longitude at all to avoid PII in logs (CWE-312, CWE-532).
        #           Only a fixed redacted placeholder is written to the logs; no masking functions are used.
        logger.info("  Location: [REDACTED]")
        # Log timezone region only, not full timezone (security: CWE-532)
        logger.info(f"  Region: {country_code}-{state}")
        # Log estimator type only, not configuration (security: CWE-532)
        logger.info(f"  Estimator: {masked_estimator}")
        logger.info("  Feature Engineering:")
        logger.info(f"    - Weather Windows: {include_weather_windows}")
        logger.info(f"    - Holiday Features: {include_holiday_features}")
        logger.info(f"    - Polynomial Features: {include_poly_features}")
        logger.info(f"  Weights Type: {type(weights).__name__}")
        logger.info(f"{'=' * 80}")

        # SECURITY CRITICAL (CWE-312, CWE-532): NEVER log the following in clear text:
        # - Raw latitude/longitude values (beyond 1 decimal precision)
        # - Full timezone strings (only log region/country)
        # - Estimator configuration details (only log type)
        # - The forecast_kwargs dictionary (contains unmasked sensitive data)
        # This applies to ALL log levels including DEBUG, and to ALL error messages.
        # Rationale: Prevents PII exposure in log aggregation systems, crash dumps,
        # and production monitoring tools where logs may be retained long-term or
        # accessible to unauthorized parties.

    # --- Step 1: Multi-Output Recursive Forecasting with Covariates ---
    if verbose:
        logger.info("Step 1: Executing multi-output recursive forecasting...")

    # Prepare kwargs for n2n_predict_with_covariates
    # SECURITY: Do NOT log this dict - contains sensitive location data (CWE-532, CWE-312)
    forecast_kwargs = {
        "data": data,
        "forecast_horizon": forecast_horizon,
        "contamination": contamination,
        "window_size": window_size,
        "lags": lags,
        "train_ratio": train_ratio,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "country_code": country_code,
        "state": state,
        "estimator": estimator,
        "include_weather_windows": include_weather_windows,
        "include_holiday_features": include_holiday_features,
        "include_poly_features": include_poly_features,
        "verbose": verbose,
        "show_progress": show_progress,
    }

    # Add any additional kwargs
    forecast_kwargs.update(kwargs)

    # Execute recursive forecasting
    # SECURITY: Wrapped in try-except to prevent sensitive data exposure in tracebacks
    try:
        predictions, model_metrics, feature_info = n2n_predict_with_covariates(
            **forecast_kwargs
        )
    except Exception as e:
        # SECURITY: Do not log any location data (even masked) to avoid CWE-532/CWE-312
        # Include only non-sensitive context information in the error message.
        logger.error(
            "Forecasting failed during N-to-1 forecasting. Exception type: %s. Estimator: %s",
            type(e).__name__,
            masked_estimator,
        )
        raise

    if verbose:
        logger.info(f"Multi-output predictions shape: {predictions.shape}")
        logger.info(f"Output columns: {list(predictions.columns)}")
        logger.info(f"Date range: {predictions.index[0]} to {predictions.index[-1]}")

    # --- Step 2: Prediction Aggregation ---
    if verbose:
        logger.info("Step 2: Aggregating predictions using weighted combination...")

    combined_prediction = agg_predict(predictions, weights=weights)

    if verbose:
        logger.info(f"Combined prediction shape: {combined_prediction.shape}")
        logger.info("Aggregation Summary:")
        logger.info("  Combined Prediction Head:")
        logger.info(f"\n{combined_prediction.head()}")
        logger.info("  Combined Prediction Statistics:")
        logger.info(f"    Mean: {combined_prediction.mean():.4f}")
        logger.info(f"    Std:  {combined_prediction.std():.4f}")
        logger.info(f"    Min:  {combined_prediction.min():.4f}")
        logger.info(f"    Max:  {combined_prediction.max():.4f}")
        logger.info(f"{'=' * 80}")

    return predictions, combined_prediction, model_metrics, feature_info


def main(
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
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
    verbose: bool = False,
    weights: Optional[List[float]] = None,
    log_dir: Optional[Path] = None,
    logging_enabled: bool = False,
) -> None:
    """Execute the complete N-to-1 forecasting pipeline with configurable parameters.

    Args:
        forecast_horizon (int): Number of steps ahead to forecast. Default: 24.
        contamination (float): Outlier contamination parameter [0, 1]. Default: 0.01.
        window_size (int): Rolling window size for features. Default: 72.
        lags (int): Number of lags for recursive model. Default: 24.
        train_ratio (float): Training data split ratio. Default: 0.8.
        latitude (float): Geographic latitude. Default: 51.5136.
        longitude (float): Geographic longitude. Default: 7.4653.
        timezone (str): Data timezone. Default: "UTC".
        country_code (str): Holiday country code. Default: "DE".
        state (str): Holiday state code. Default: "NW".
        include_weather_windows (bool): Toggle weather window features. Default: False.
        include_holiday_features (bool): Toggle holiday features. Default: False.
        include_poly_features (bool): Toggle polynomial features. Default: False.
        verbose (bool): Toggle detailed logging. Default: False.
        weights (Optional[List[float]]): List of weights for prediction aggregation.
            Default: DEFAULT_WEIGHTS.
        log_dir (Optional[Path]): Directory to save log files. If None, uses default path.
        logging_enabled (bool): Toggle overall logging (console and file). Default: False.
    """
    # Use default log directory if none provided
    if log_dir is None:
        log_dir = Path.home() / "spotforecast2_safe_models" / "logs"

    # Setup Logging if enabled
    log_file = None
    if logging_enabled:
        logger, log_file = setup_logging(
            level=logging.INFO if verbose else logging.WARNING, log_dir=log_dir
        )
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    if weights is None:
        # Use a copy to avoid accidental mutation of the module-level default.
        weights = DEFAULT_WEIGHTS.copy()

    data = fetch_data(filename=get_package_data_home() / "demo10.csv")

    logger.info("--- Starting n_to_1_with_covariates using modular functions ---")

    # Execute the forecasting pipeline
    # SECURITY: Wrapped in try-except to prevent sensitive data exposure in tracebacks
    try:
        predictions, combined_prediction, model_metrics, feature_info = (
            n_to_1_with_covariates(
                data=data,
                forecast_horizon=forecast_horizon,
                contamination=contamination,
                window_size=window_size,
                lags=lags,
                train_ratio=train_ratio,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                country_code=country_code,
                state=state,
                estimator=None,
                include_weather_windows=include_weather_windows,
                include_holiday_features=include_holiday_features,
                include_poly_features=include_poly_features,
                weights=weights,
                verbose=verbose,
            )
        )
    except Exception as e:
        # Log error without exposing sensitive parameters (CWE-532, CWE-312)
        logger.error(
            "Pipeline execution failed: %s. Region: %s-%s", str(e), country_code, state
        )
        raise

    # Print results to stdout even if logging is low-level
    print("\nMulti-output predictions head:")
    print(predictions.head())

    print("\nCombined Prediction Head:")
    print(combined_prediction.head())

    if log_file:
        print(f"\nFinalized logging info saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the safety-critical N-to-1 forecasting demo with exogenous covariates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Technical Parameters
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=24,
        help="Number of steps ahead to forecast.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Outlier contamination parameter [0, 1].",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=72,
        help="Rolling window size for feature extraction.",
    )
    parser.add_argument(
        "--lags", type=int, default=24, help="Number of lags for recursive model."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training.",
    )

    # Location Parameters
    parser.add_argument(
        "--latitude",
        type=float,
        default=51.5136,
        help="Location latitude for solar features.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=7.4653,
        help="Location longitude for solar features.",
    )
    parser.add_argument(
        "--timezone", type=str, default="UTC", help="Timezone for data processing."
    )
    parser.add_argument(
        "--country_code",
        type=str,
        default="DE",
        help="Country code for holidays (ISO 3166).",
    )
    parser.add_argument(
        "--state", type=str, default="NW", help="State code for regional holidays."
    )

    # Feature Engineering Flags
    parser.add_argument(
        "--include_weather_windows",
        type=_parse_bool,
        default=False,
        help="Enable rolling weather statistics.",
    )
    parser.add_argument(
        "--include_holiday_features",
        type=_parse_bool,
        default=False,
        help="Enable holiday binary indicators.",
    )
    parser.add_argument(
        "--include_poly_features",
        type=_parse_bool,
        default=False,
        help="Enable polynomial interaction terms.",
    )

    # Execution Controls
    parser.add_argument(
        "--verbose",
        type=_parse_bool,
        default=False,
        help="Enable verbose mission-critical logging.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Space-separated list of weights for prediction aggregation.",
    )
    parser.add_argument(
        "--logging",
        type=_parse_bool,
        default=False,
        help="Enable overall logging (console and file).",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Custom directory for execution logs."
    )

    args = parser.parse_args()

    # Process path
    specified_log_dir = Path(args.log_dir) if args.log_dir else None

    # Filter out log_dir and rename logging to logging_enabled for main kwargs
    kwargs = vars(args)
    if specified_log_dir:
        kwargs["log_dir"] = specified_log_dir

    # Map 'logging' arg to 'logging_enabled' parameter
    kwargs["logging_enabled"] = kwargs.pop("logging")

    try:
        main(**kwargs)
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical failure: {e}")
        sys.exit(1)
