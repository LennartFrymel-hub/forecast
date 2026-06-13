import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)
from spotforecast2_safe.manager.exo.entsoe_forecast import (
    build_entsoe_forecast_exog,
    repair_historical_entsoe_forecast,
)
from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
    n_to_1_with_covariates,
)


# =============================================================================
# Backtest configuration
# =============================================================================

HOLDOUT_HOURS = 24
OUTPUT_TIMEZONE = "Europe/Berlin"

# Simulated forecast creation time on the previous day.
# This mirrors the current live logic of script 07.
FORECAST_ISSUE_HOUR_LOCAL = 21

# At issue time, actual load is assumed to be available only
# a few hours earlier.
DATA_AVAILABILITY_LAG_HOURS = 4

# Used only if no --target-day is given.
EXCLUDE_LATEST_DAYS = 7

MIN_TRAINING_ROWS = 1000

# Fixed training start.
# Change this if you later want a different training history.
TRAIN_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

CONTAMINATION_LEVEL = 0.005

LAGS = [
    1, 2, 3,
    23, 24, 25,
    47, 48,
    167, 168, 169,
    336,
]

ROLLING_WINDOW_SIZES = [72, 168, 720]


# =============================================================================
# Weather locations
# =============================================================================

DE_WEATHER_LOCATIONS = [
    {"name": "hamburg", "latitude": 53.5511, "longitude": 9.9937, "weight": 1.0},
    {"name": "berlin", "latitude": 52.5200, "longitude": 13.4050, "weight": 1.0},
    {"name": "koeln", "latitude": 50.9375, "longitude": 6.9603, "weight": 1.0},
    {"name": "frankfurt", "latitude": 50.1109, "longitude": 8.6821, "weight": 1.0},
    {"name": "stuttgart", "latitude": 48.7758, "longitude": 9.1829, "weight": 1.0},
    {"name": "muenchen", "latitude": 48.1372, "longitude": 11.5756, "weight": 1.0},
    {"name": "leipzig", "latitude": 51.3397, "longitude": 12.3731, "weight": 1.0},
    {"name": "hannover", "latitude": 52.3759, "longitude": 9.7320, "weight": 1.0},
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-day next-calendar-day backtest for German load forecasting. "
            "Example: python scripts\\04_backtest_de_load.py --target-day 2026-04-12"
        )
    )

    parser.add_argument(
        "--target-day",
        default=None,
        help=(
            "Local target calendar day in YYYY-MM-DD format. "
            "The script simulates a forecast created on the previous day "
            f"at {FORECAST_ISSUE_HOUR_LOCAL}:00 {OUTPUT_TIMEZONE}. "
            "If omitted, the script uses the latest complete local day minus "
            f"{EXCLUDE_LATEST_DAYS} days."
        ),
    )

    parser.add_argument(
        "--train-start",
        default=str(TRAIN_START.date()),
        help=(
            "Training start date in YYYY-MM-DD format. "
            "Default: 2023-01-01."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    """Calculate forecast error metrics."""
    df = pd.concat(
        [
            actual.rename("actual"),
            prediction.rename("prediction"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        raise ValueError("No common timestamps between actual and prediction.")

    error = df["prediction"] - df["actual"]
    abs_error = error.abs()

    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())

    valid_mape = df["actual"].abs() > 1e-9
    mape = (
        abs_error[valid_mape] / df.loc[valid_mape, "actual"].abs()
    ).mean() * 100

    denominator = df["actual"].abs() + df["prediction"].abs()
    smape = (2 * abs_error / denominator.replace(0, np.nan)).mean() * 100

    bias = error.mean()
    max_abs_error = abs_error.max()

    return {
        "n_points": len(df),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "Bias": bias,
        "MaxAbsError": max_abs_error,
    }


# =============================================================================
# Day selection
# =============================================================================

def get_latest_complete_local_day_start(actual_raw: pd.Series) -> pd.Timestamp:
    """Return start timestamp of the latest complete local calendar day."""
    last_valid_timestamp = actual_raw.dropna().index.max()

    if pd.isna(last_valid_timestamp):
        raise ValueError("No valid actual load values found.")

    last_valid_local = last_valid_timestamp.tz_convert(OUTPUT_TIMEZONE)

    candidate_day_start_local = last_valid_local.normalize()
    candidate_day_end_local = (
        candidate_day_start_local
        + pd.DateOffset(days=1)
        - pd.Timedelta(hours=1)
    )

    # If the current local day is not complete yet,
    # go back one full local calendar day.
    if last_valid_local < candidate_day_end_local:
        candidate_day_start_local = (
            candidate_day_start_local - pd.DateOffset(days=1)
        )

    return candidate_day_start_local


def resolve_target_day_start_local(
    actual_raw: pd.Series,
    target_day: str | None,
) -> pd.Timestamp:
    """Resolve target day from CLI or fallback logic."""
    if target_day is not None:
        return pd.Timestamp(target_day).tz_localize(OUTPUT_TIMEZONE)

    latest_complete_day_start_local = get_latest_complete_local_day_start(
        actual_raw
    )

    return latest_complete_day_start_local - pd.DateOffset(
        days=EXCLUDE_LATEST_DAYS
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    data_home = Path(
        os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe")
    )
    model_root = Path(
        os.environ.get("SPOTFORECAST2_CACHE", Path.cwd() / "models_cache")
    )

    train_start = pd.Timestamp(args.train_start).tz_localize("UTC")

    print("Single next-calendar-day backtest")
    print(f"Data folder:               {data_home}")
    print(f"Model folder:              {model_root}")
    print(f"Returned forecast horizon: {HOLDOUT_HOURS} hours")
    print(f"Output timezone:           {OUTPUT_TIMEZONE}")
    print(f"Forecast issue hour local: {FORECAST_ISSUE_HOUR_LOCAL}:00")
    print(f"Data availability lag:     {DATA_AVAILABILITY_LAG_HOURS} hours")
    print(f"Target day argument:       {args.target_day}")
    print(f"Training start:            {train_start}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    actual_raw = load_timeseries(
        data_home=data_home,
        on_missing="passthrough",
    )

    actual_filled = load_timeseries(
        data_home=data_home,
        on_missing="ffill_bfill",
    )

    entsoe_forecast = load_timeseries_forecast(
        data_home=data_home,
        on_missing="passthrough",
    )

    weekly_persistence = actual_filled.shift(168)

    # -------------------------------------------------------------------------
    # Select target calendar day
    # -------------------------------------------------------------------------
    target_day_start_local = resolve_target_day_start_local(
        actual_raw=actual_raw,
        target_day=args.target_day,
    )

    target_day_end_local = (
        target_day_start_local
        + pd.DateOffset(days=1)
        - pd.Timedelta(hours=1)
    )

    issue_time_local = (
        target_day_start_local
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=FORECAST_ISSUE_HOUR_LOCAL)
    )

    train_end_local = issue_time_local - pd.Timedelta(
        hours=DATA_AVAILABILITY_LAG_HOURS
    )

    # Convert to UTC for internal pipeline logic.
    test_start = target_day_start_local.tz_convert("UTC")
    test_end = target_day_end_local.tz_convert("UTC")
    train_end = train_end_local.tz_convert("UTC")

    model_forecast_start = train_end + pd.Timedelta(hours=1)

    required_horizon = int(
        (test_end - model_forecast_start) / pd.Timedelta(hours=1)
    ) + 1

    if required_horizon <= 0:
        raise ValueError(
            f"Invalid required_horizon={required_horizon}. "
            f"model_forecast_start={model_forecast_start}, test_end={test_end}"
        )

    expected_test_index = pd.date_range(
        start=test_start,
        end=test_end,
        freq="h",
        tz="UTC",
    )

    # DST days can have 23 or 25 hours.
    # This script is intended to mirror the final 24h project output.
    if len(expected_test_index) != HOLDOUT_HOURS:
        raise ValueError(
            "Selected target day does not contain exactly 24 local hours. "
            f"Target day: {target_day_start_local.date()}, "
            f"hours: {len(expected_test_index)}. "
            "Choose another day."
        )

    print()
    print("Backtest ranges:")
    print(f"Target day local:       {target_day_start_local.date()}")
    print(f"Issue time local:       {issue_time_local}")
    print(f"Training until local:   {train_end_local}")
    print(f"Training until UTC:     {train_end}")
    print(f"Model forecast start:   {model_forecast_start}")
    print(f"Output start local:     {target_day_start_local}")
    print(f"Output end local:       {target_day_end_local}")
    print(f"Output start UTC:       {test_start}")
    print(f"Output end UTC:         {test_end}")
    print(f"Internal horizon:       {required_horizon} hours")
    print(f"Returned forecast:      {HOLDOUT_HOURS} hours")

    # -------------------------------------------------------------------------
    # Build train and test slices
    # -------------------------------------------------------------------------
    train_series = actual_filled.loc[train_start:train_end].copy()
    actual_test = actual_raw.reindex(expected_test_index)
    entsoe_test = entsoe_forecast.reindex(expected_test_index)
    persistence_test = weekly_persistence.reindex(expected_test_index)

    if train_series.empty:
        raise ValueError(
            f"No training data available from {train_start} to {train_end}."
        )

    if len(train_series) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Too little training data: {len(train_series)} rows. "
            "Use more history."
        )

    if actual_test.isna().any():
        missing_actual = actual_test[actual_test.isna()]
        preview = ", ".join(str(ts) for ts in missing_actual.index[:5])
        raise ValueError(
            "Actual load is incomplete in the backtest target window. "
            f"First missing timestamps: {preview}"
        )

    print()
    print("Training data:")
    print(train_series.head())
    print(train_series.tail())
    print(f"Shape: {train_series.shape}")

    print()
    print("Actual test data:")
    print(actual_test.head())
    print(actual_test.tail())
    print(f"Missing actual values in test window: {actual_test.isna().sum()}")

    train_data = train_series.rename("load_de").to_frame()

    # -------------------------------------------------------------------------
    # Build ENTSO-E exogenous forecast feature
    # -------------------------------------------------------------------------
    repaired_historical_entsoe_forecast = repair_historical_entsoe_forecast(
        entsoe_forecast=entsoe_forecast,
        start=train_data.index.min(),
        end=train_end,
    )

    expected_future_entsoe_index = pd.date_range(
        start=model_forecast_start,
        end=test_end,
        freq="h",
        tz="UTC",
    )

    future_entsoe_forecast = entsoe_forecast.reindex(
        expected_future_entsoe_index
    )

    if future_entsoe_forecast.isna().any():
        missing_future = future_entsoe_forecast[
            future_entsoe_forecast.isna()
        ]
        preview = ", ".join(str(ts) for ts in missing_future.index[:5])
        raise ValueError(
            "Historical future ENTSO-E forecast is incomplete for this "
            f"backtest window. First missing timestamps: {preview}"
        )

    entsoe_feature_series = pd.concat(
        [
            repaired_historical_entsoe_forecast,
            future_entsoe_forecast,
        ]
    ).sort_index()

    entsoe_feature_series = entsoe_feature_series[
        ~entsoe_feature_series.index.duplicated(keep="last")
    ]

    entsoe_exog = build_entsoe_forecast_exog(
        entsoe_forecast=entsoe_feature_series,
        start=train_data.index.min(),
        end=test_end,
    )

    # -------------------------------------------------------------------------
    # Train and forecast
    # -------------------------------------------------------------------------
    print()
    print("Start backtest forecast...")

    model_dir_suffix = str(target_day_start_local.date()).replace("-", "_")

    predictions, combined_prediction, model_metrics, feature_info = (
        n_to_1_with_covariates(
            data=train_data,
            forecast_horizon=required_horizon,
            lags=LAGS,
            window_size=ROLLING_WINDOW_SIZES,
            train_ratio=0.85,
            latitude=51.1657,
            longitude=10.4515,
            weather_locations=DE_WEATHER_LOCATIONS,
            add_weighted_weather_average=True,
            keep_regional_weather_features=False,
            weather_cache_home=Path("weather_cache") / "de_multi",
            timezone="UTC",
            country_code="DE",
            state="NW",
            contamination=CONTAMINATION_LEVEL,
            include_weather_windows=True,
            include_holiday_features=True,
            include_poly_features=False,
            weights={"load_de": 1.0},
            force_train=True,
            model_dir=(
                model_root
                / f"de_load_single_calendar_day_entsoe_exog_{model_dir_suffix}"
            ),
            verbose=True,
            show_progress=True,
        )
    )

    # -------------------------------------------------------------------------
    # Trim forecast to the exact local target calendar day
    # -------------------------------------------------------------------------
    predictions = predictions.loc[test_start:test_end]
    combined_prediction = combined_prediction.loc[test_start:test_end]
    combined_prediction = combined_prediction.reindex(expected_test_index)

    if len(combined_prediction) != HOLDOUT_HOURS:
        raise ValueError(
            "Final backtest output does not contain exactly 24 values. "
            f"Expected {HOLDOUT_HOURS}, got {len(combined_prediction)}."
        )

    if combined_prediction.isna().any():
        raise ValueError("Final spotforecast contains missing values.")

    spotforecast_prediction = combined_prediction.rename("spotforecast")

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    comparison = pd.concat(
        [
            actual_test.rename("actual_load"),
            spotforecast_prediction.rename("spotforecast"),
            entsoe_test.rename("entsoe_forecast"),
            persistence_test.rename("weekly_persistence"),
        ],
        axis=1,
    )

    comparison["error_spotforecast"] = (
        comparison["spotforecast"] - comparison["actual_load"]
    )
    comparison["abs_error_spotforecast"] = comparison[
        "error_spotforecast"
    ].abs()

    comparison["error_entsoe"] = (
        comparison["entsoe_forecast"] - comparison["actual_load"]
    )
    comparison["abs_error_entsoe"] = comparison[
        "error_entsoe"
    ].abs()

    comparison["error_weekly_persistence"] = (
        comparison["weekly_persistence"] - comparison["actual_load"]
    )
    comparison["abs_error_weekly_persistence"] = comparison[
        "error_weekly_persistence"
    ].abs()

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    metrics_rows = []

    spotforecast_metrics = calculate_metrics(
        comparison["actual_load"],
        comparison["spotforecast"],
    )
    spotforecast_metrics["model"] = "spotforecast"
    metrics_rows.append(spotforecast_metrics)

    if comparison["entsoe_forecast"].notna().any():
        entsoe_metrics = calculate_metrics(
            comparison["actual_load"],
            comparison["entsoe_forecast"],
        )
        entsoe_metrics["model"] = "entsoe_forecast"
        metrics_rows.append(entsoe_metrics)

    if comparison["weekly_persistence"].notna().any():
        persistence_metrics = calculate_metrics(
            comparison["actual_load"],
            comparison["weekly_persistence"],
        )
        persistence_metrics["model"] = "weekly_persistence"
        metrics_rows.append(persistence_metrics)

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    comparison_output = comparison.copy()
    comparison_output.index = comparison_output.index.tz_convert(
        OUTPUT_TIMEZONE
    )
    comparison_output = comparison_output.rename_axis("timestamp")

    target_day_str = str(target_day_start_local.date())

    comparison_file = (
        output_dir
        / f"backtest_comparison_{target_day_str}_{HOLDOUT_HOURS}h_entsoe_exog.csv"
    )
    metrics_file = (
        output_dir
        / f"backtest_metrics_{target_day_str}_{HOLDOUT_HOURS}h_entsoe_exog.csv"
    )

    comparison_output.to_csv(comparison_file)
    metrics_df.to_csv(metrics_file)

    # -------------------------------------------------------------------------
    # Print result
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("BACKTEST RESULT")
    print("=" * 80)
    print()
    print(metrics_df)

    print()
    print("Actual vs prediction:")
    print(comparison_output.head())
    print()
    print(comparison_output.tail())

    print()
    print("Saved files:")
    print(comparison_file)
    print(metrics_file)


if __name__ == "__main__":
    main()