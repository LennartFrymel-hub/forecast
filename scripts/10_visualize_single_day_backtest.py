import argparse
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
# Configuration
# =============================================================================

OUTPUT_TIMEZONE = "Europe/Berlin"
FORECAST_HORIZON = 24

# Same logic as script 07:
# Forecast is issued on the previous day at 21:00 local time.
FORECAST_ISSUE_HOUR_LOCAL = 21

# Actual load is assumed to be available only until
# a few hours before the issue time.
DATA_AVAILABILITY_LAG_HOURS = 4

MIN_TRAINING_ROWS = 1000

TRAIN_YEARS = 2

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
# Metrics
# =============================================================================

def calculate_metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    df = pd.concat(
        [
            actual.rename("actual"),
            prediction.rename("prediction"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return {
            "n_points": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_percent": np.nan,
            "Bias": np.nan,
            "MaxAbsError": np.nan,
        }

    error = df["prediction"] - df["actual"]
    abs_error = error.abs()

    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())

    valid_mape = df["actual"].abs() > 1e-9
    mape = (
        abs_error[valid_mape] / df.loc[valid_mape, "actual"].abs()
    ).mean() * 100

    bias = error.mean()
    max_abs_error = abs_error.max()

    return {
        "n_points": len(df),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "Bias": bias,
        "MaxAbsError": max_abs_error,
    }


def metrics_for_period(
    comparison: pd.DataFrame,
    period_name: str,
) -> list[dict]:
    rows = []

    models = {
        "spotforecast": comparison["spotforecast"],
        "entsoe_forecast": comparison["entsoe_forecast"],
        "weekly_persistence": comparison["weekly_persistence"],
    }

    for model_name, prediction in models.items():
        metrics = calculate_metrics(
            actual=comparison["actual_load"],
            prediction=prediction,
        )
        metrics["period"] = period_name
        metrics["model"] = model_name
        rows.append(metrics)

    return rows


# =============================================================================
# Plot helpers
# =============================================================================

def format_hourly_time_axis(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.set_xlim(
        index.min(),
        index.max(),
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
    )


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visual single-day next-calendar-day backtest. "
            "Example: --target-day 2026-05-10"
        )
    )
    parser.add_argument(
        "--target-day",
        required=True,
        help="Local target calendar day in YYYY-MM-DD format.",
    )
    return parser.parse_args()


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

    # -------------------------------------------------------------------------
    # Target calendar day
    # -------------------------------------------------------------------------
    target_day_start_local = pd.Timestamp(args.target_day).tz_localize(
        OUTPUT_TIMEZONE
    )
    target_day_end_local = (
        target_day_start_local
        + pd.DateOffset(days=1)
        - pd.Timedelta(hours=1)
    )

    test_start = target_day_start_local.tz_convert("UTC")
    test_end = target_day_end_local.tz_convert("UTC")

    expected_test_index = pd.date_range(
        start=test_start,
        end=test_end,
        freq="h",
        tz="UTC",
    )

    if len(expected_test_index) != FORECAST_HORIZON:
        raise ValueError(
            "The selected local target day does not have exactly 24 hours. "
            f"Target day: {args.target_day}, hours: {len(expected_test_index)}. "
            "Please choose another date."
        )

    # -------------------------------------------------------------------------
    # Simulated forecast issue moment
    # -------------------------------------------------------------------------
    issue_time_local = (
        target_day_start_local
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=FORECAST_ISSUE_HOUR_LOCAL)
    )

    train_end_local = issue_time_local - pd.Timedelta(
        hours=DATA_AVAILABILITY_LAG_HOURS
    )
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

    print("Visual single-day backtest")
    print(f"Data folder:              {data_home}")
    print(f"Model folder:             {model_root}")
    print()
    print("Backtest setup:")
    print(f"Target day local:         {target_day_start_local.date()}")
    print(f"Issue time local:         {issue_time_local}")
    print(f"Training until local:     {train_end_local}")
    print(f"Training until UTC:       {train_end}")
    print(f"Model forecast start UTC: {model_forecast_start}")
    print(f"Output start local:       {target_day_start_local}")
    print(f"Output end local:         {target_day_end_local}")
    print(f"Output start UTC:         {test_start}")
    print(f"Output end UTC:           {test_end}")
    print(f"Internal horizon:         {required_horizon} hours")
    print(f"Returned horizon:         {FORECAST_HORIZON} hours")

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
    # Build data slices
    # -------------------------------------------------------------------------
    train_series = actual_filled.loc[:train_end]
    actual_test = actual_raw.reindex(expected_test_index)
    entsoe_test = entsoe_forecast.reindex(expected_test_index)
    persistence_test = weekly_persistence.reindex(expected_test_index)

    if len(train_series) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Too little training data: {len(train_series)} rows."
        )

    if actual_test.isna().any():
        missing = actual_test[actual_test.isna()]
        preview = ", ".join(str(ts) for ts in missing.index[:5])
        raise ValueError(
            "Actual load is incomplete for the selected target day. "
            f"First missing timestamps: {preview}"
        )

    train_data = train_series.rename("load_de").to_frame()

    # -------------------------------------------------------------------------
    # ENTSO-E exogenous feature
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
        missing = future_entsoe_forecast[future_entsoe_forecast.isna()]
        preview = ", ".join(str(ts) for ts in missing.index[:5])
        raise ValueError(
            "ENTSO-E forecast is incomplete for the selected backtest window. "
            f"First missing timestamps: {preview}"
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
    # Train and predict
    # -------------------------------------------------------------------------
    print()
    print("Start model training and prediction...")

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
            include_weather_windows=True,
            include_holiday_features=True,
            include_poly_features=False,
            weights={"load_de": 1.0},
            force_train=True,
            model_dir=model_root
            / f"visual_single_day_backtest_{args.target_day}",
            verbose=False,
            show_progress=True,
        )
    )

    combined_prediction = combined_prediction.loc[test_start:test_end]
    combined_prediction = combined_prediction.reindex(expected_test_index)

    if len(combined_prediction) != FORECAST_HORIZON:
        raise ValueError(
            f"Expected 24 forecast values, got {len(combined_prediction)}."
        )

    if combined_prediction.isna().any():
        raise ValueError("Spotforecast contains NaN values.")

    spotforecast = combined_prediction.rename("spotforecast")

    # -------------------------------------------------------------------------
    # Build comparison frame
    # -------------------------------------------------------------------------
    comparison = pd.concat(
        [
            actual_test.rename("actual_load"),
            spotforecast,
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

    comparison_local = comparison.copy()
    comparison_local.index = comparison_local.index.tz_convert(OUTPUT_TIMEZONE)
    comparison_local = comparison_local.rename_axis("timestamp")

    # -------------------------------------------------------------------------
    # Metrics: full day and evening 19-23
    # -------------------------------------------------------------------------
    metrics_rows = []

    metrics_rows.extend(
        metrics_for_period(
            comparison=comparison_local,
            period_name="full_day_00_23",
        )
    )

    evening_mask = comparison_local.index.hour >= 19
    comparison_evening_local = comparison_local.loc[evening_mask]

    metrics_rows.extend(
        metrics_for_period(
            comparison=comparison_evening_local,
            period_name="evening_19_23",
        )
    )

    metrics_df = pd.DataFrame(metrics_rows)

    # -------------------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    comparison_file = (
        output_dir / f"single_day_backtest_{args.target_day}_comparison.csv"
    )
    metrics_file = (
        output_dir / f"single_day_backtest_{args.target_day}_metrics.csv"
    )
    curve_plot_file = (
        output_dir / f"single_day_backtest_{args.target_day}_forecast_vs_actual.png"
    )
    error_plot_file = (
        output_dir / f"single_day_backtest_{args.target_day}_absolute_errors.png"
    )

    comparison_local.to_csv(comparison_file)
    metrics_df.to_csv(metrics_file, index=False)

    # -------------------------------------------------------------------------
    # Plot 1: Forecast curves
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        comparison_local.index,
        comparison_local["actual_load"],
        marker="o",
        label="Actual load",
    )
    ax.plot(
        comparison_local.index,
        comparison_local["spotforecast"],
        marker="o",
        label="Spotforecast",
    )
    ax.plot(
        comparison_local.index,
        comparison_local["entsoe_forecast"],
        marker="o",
        label="ENTSO-E forecast",
    )
    ax.plot(
        comparison_local.index,
        comparison_local["weekly_persistence"],
        marker="o",
        label="Weekly persistence",
    )

    ax.axvline(
        pd.Timestamp(f"{args.target_day} 19:00", tz=OUTPUT_TIMEZONE),
        linestyle="--",
        label="19:00 local",
    )

    ax.set_title(f"Forecast comparison for {args.target_day}")
    ax.set_xlabel("Local time Europe/Berlin")
    ax.set_ylabel("Load")

    format_hourly_time_axis(
        ax=ax,
        index=comparison_local.index,
    )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_plot_file, dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: Absolute errors
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        comparison_local.index,
        comparison_local["abs_error_spotforecast"],
        marker="o",
        label="Abs error Spotforecast",
    )
    ax.plot(
        comparison_local.index,
        comparison_local["abs_error_entsoe"],
        marker="o",
        label="Abs error ENTSO-E",
    )
    ax.plot(
        comparison_local.index,
        comparison_local["abs_error_weekly_persistence"],
        marker="o",
        label="Abs error Weekly persistence",
    )

    ax.axvline(
        pd.Timestamp(f"{args.target_day} 19:00", tz=OUTPUT_TIMEZONE),
        linestyle="--",
        label="19:00 local",
    )

    ax.set_title(f"Absolute forecast errors for {args.target_day}")
    ax.set_xlabel("Local time Europe/Berlin")
    ax.set_ylabel("Absolute error")

    format_hourly_time_axis(
        ax=ax,
        index=comparison_local.index,
    )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(error_plot_file, dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Print result
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SINGLE DAY BACKTEST METRICS")
    print("=" * 80)
    print()
    print(
        metrics_df[
            [
                "period",
                "model",
                "n_points",
                "MAE",
                "RMSE",
                "MAPE_percent",
                "Bias",
                "MaxAbsError",
            ]
        ].to_string(index=False)
    )

    print()
    print("Evening rows 19:00-23:00 local:")
    print(
        comparison_evening_local[
            [
                "actual_load",
                "spotforecast",
                "entsoe_forecast",
                "weekly_persistence",
                "abs_error_spotforecast",
                "abs_error_entsoe",
                "abs_error_weekly_persistence",
            ]
        ].to_string()
    )

    print()
    print("Saved files:")
    print(comparison_file)
    print(metrics_file)
    print(curve_plot_file)
    print(error_plot_file)


if __name__ == "__main__":
    main()