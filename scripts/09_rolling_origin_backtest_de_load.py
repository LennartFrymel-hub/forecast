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

FORECAST_HORIZON = 24
OUTPUT_TIMEZONE = "Europe/Berlin"

# Same conceptual setup as script 07:
# We simulate that the forecast is issued on the previous day at 21:00 local time.
FORECAST_ISSUE_HOUR_LOCAL = 21

# At issue time, actual load is assumed to be available only up to
# a few hours earlier. This mirrors the practical delay in the live script.
DATA_AVAILABILITY_LAG_HOURS = 4

# Use weekly folds first. This keeps runtime manageable.
# Later you can set FOLD_STEP_DAYS = 1 for daily folds.
BACKTEST_DAYS = 56
FOLD_STEP_DAYS = 7

# Do not test directly at the latest data edge.
# Latest actual values can still be incomplete.
EXCLUDE_LATEST_DAYS = 7

MIN_TRAINING_ROWS = 1000

TRAIN_YEARS = 2

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
            "sMAPE_percent": np.nan,
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
# Fold construction
# =============================================================================

def get_latest_complete_local_day_start(actual_raw: pd.Series) -> pd.Timestamp:
    """Return the start of the latest complete local calendar day."""
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
    # move one local calendar day back.
    if last_valid_local < candidate_day_end_local:
        candidate_day_start_local = candidate_day_start_local - pd.DateOffset(days=1)

    return candidate_day_start_local


def build_fold_target_days(actual_raw: pd.Series) -> list[pd.Timestamp]:
    """Build target days that represent full local calendar days."""
    latest_complete_day_start_local = get_latest_complete_local_day_start(
        actual_raw
    )

    latest_target_day_start_local = (
        latest_complete_day_start_local
        - pd.DateOffset(days=EXCLUDE_LATEST_DAYS)
    )

    first_target_day_start_local = (
        latest_target_day_start_local
        - pd.DateOffset(days=BACKTEST_DAYS)
    )

    fold_target_days = list(
        pd.date_range(
            start=first_target_day_start_local,
            end=latest_target_day_start_local,
            freq=f"{FOLD_STEP_DAYS}D",
        )
    )

    if not fold_target_days:
        raise ValueError("No fold target days could be created.")

    return fold_target_days


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data_home = Path(
        os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe")
    )
    model_root = Path(
        os.environ.get("SPOTFORECAST2_CACHE", Path.cwd() / "models_cache")
    )

    print("Rolling-origin backtest")
    print(f"Data folder:                  {data_home}")
    print(f"Model folder:                 {model_root}")
    print(f"Returned forecast horizon:    {FORECAST_HORIZON} hours")
    print(f"Output timezone:              {OUTPUT_TIMEZONE}")
    print(f"Forecast issue hour local:    {FORECAST_ISSUE_HOUR_LOCAL}:00")
    print(f"Data availability lag:        {DATA_AVAILABILITY_LAG_HOURS} hours")
    print(f"Backtest days:                {BACKTEST_DAYS}")
    print(f"Fold step days:               {FOLD_STEP_DAYS}")
    print(f"Exclude latest:               {EXCLUDE_LATEST_DAYS} days")

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
    # Build fold target days
    # -------------------------------------------------------------------------
    fold_target_days = build_fold_target_days(actual_raw)

    print()
    print("Fold target days local:")
    for target_day in fold_target_days:
        print(f"  {target_day.date()}")

    all_prediction_rows: list[pd.DataFrame] = []
    all_metric_rows: list[dict] = []

    # -------------------------------------------------------------------------
    # Run folds
    # -------------------------------------------------------------------------
    for fold_id, target_day_start_local in enumerate(fold_target_days, start=1):
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

        test_start = target_day_start_local.tz_convert("UTC")
        test_end = target_day_end_local.tz_convert("UTC")
        train_end = train_end_local.tz_convert("UTC")

        model_forecast_start = train_end + pd.Timedelta(hours=1)

        required_horizon = int(
            (test_end - model_forecast_start) / pd.Timedelta(hours=1)
        ) + 1

        expected_test_index = pd.date_range(
            start=test_start,
            end=test_end,
            freq="h",
            tz="UTC",
        )

        # On DST transition days, the local calendar day can have 23 or 25 hours.
        # Since the project evaluation expects exactly 24 values,
        # skip such days in the backtest.
        if len(expected_test_index) != FORECAST_HORIZON:
            print()
            print("=" * 80)
            print(f"Skip fold {fold_id}: local target day has {len(expected_test_index)} hours.")
            print(f"Target day local: {target_day_start_local.date()}")
            print("=" * 80)
            continue

        if required_horizon <= 0:
            print()
            print("=" * 80)
            print(f"Skip fold {fold_id}: invalid required_horizon={required_horizon}.")
            print("=" * 80)
            continue

        print()
        print("=" * 80)
        print(f"Fold {fold_id}/{len(fold_target_days)}")
        print("=" * 80)
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
        print(f"Returned horizon:       {FORECAST_HORIZON} hours")

        # ---------------------------------------------------------------------
        # Build train and evaluation slices
        # ---------------------------------------------------------------------
        train_series = actual_filled.loc[:train_end]
        actual_test = actual_raw.reindex(expected_test_index)
        entsoe_test = entsoe_forecast.reindex(expected_test_index)
        persistence_test = weekly_persistence.reindex(expected_test_index)

        if len(train_series) < MIN_TRAINING_ROWS:
            print(f"Skip fold {fold_id}: too little training data.")
            continue

        if actual_test.isna().any():
            missing_actual = actual_test[actual_test.isna()]
            preview = ", ".join(str(ts) for ts in missing_actual.index[:5])
            print(
                f"Skip fold {fold_id}: actual load is incomplete. "
                f"First missing timestamps: {preview}"
            )
            continue

        train_data = train_series.rename("load_de").to_frame()

        # ---------------------------------------------------------------------
        # Build ENTSO-E forecast exogenous feature
        # ---------------------------------------------------------------------
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
            missing_future_entsoe = future_entsoe_forecast[
                future_entsoe_forecast.isna()
            ]
            preview = ", ".join(str(ts) for ts in missing_future_entsoe.index[:5])
            print(
                f"Skip fold {fold_id}: ENTSO-E future forecast exog is incomplete. "
                f"First missing timestamps: {preview}"
            )
            continue

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

        # ---------------------------------------------------------------------
        # Train and predict
        # ---------------------------------------------------------------------
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
                contamination = CONTAMINATION_LEVEL,
                include_holiday_features=True,
                include_poly_features=False,
                weights={"load_de": 1.0},
                force_train=True,
                model_dir=model_root
                / f"rolling_backtest_entsoe_exog_fold_{fold_id:03d}",
                verbose=False,
                show_progress=True,
            )
        )

        # ---------------------------------------------------------------------
        # Trim internal forecast to the actual target calendar day
        # ---------------------------------------------------------------------
        predictions = predictions.loc[test_start:test_end]
        combined_prediction = combined_prediction.loc[test_start:test_end]

        combined_prediction = combined_prediction.reindex(expected_test_index)

        if len(combined_prediction) != FORECAST_HORIZON:
            print(
                f"Skip fold {fold_id}: final prediction has "
                f"{len(combined_prediction)} rows, expected "
                f"{FORECAST_HORIZON}."
            )
            continue

        if combined_prediction.isna().any():
            print(f"Skip fold {fold_id}: final spotforecast contains NaN values.")
            continue

        spotforecast = combined_prediction.rename("spotforecast")

        # ---------------------------------------------------------------------
        # Comparison table
        # ---------------------------------------------------------------------
        comparison = pd.concat(
            [
                actual_test.rename("actual_load"),
                spotforecast.rename("spotforecast"),
                entsoe_test.rename("entsoe_forecast"),
                persistence_test.rename("weekly_persistence"),
            ],
            axis=1,
        )

        comparison["fold_id"] = fold_id
        comparison["target_day_local"] = str(target_day_start_local.date())
        comparison["issue_time_local"] = str(issue_time_local)
        comparison["train_end_utc"] = train_end
        comparison["test_start_utc"] = test_start
        comparison["test_end_utc"] = test_end
        comparison["required_horizon"] = required_horizon

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

        all_prediction_rows.append(comparison)

        # ---------------------------------------------------------------------
        # Metrics per fold
        # ---------------------------------------------------------------------
        models_to_score = {
            "spotforecast": comparison["spotforecast"],
            "entsoe_forecast": comparison["entsoe_forecast"],
            "weekly_persistence": comparison["weekly_persistence"],
        }

        for model_name, prediction_series in models_to_score.items():
            metrics = calculate_metrics(
                comparison["actual_load"],
                prediction_series,
            )
            metrics["fold_id"] = fold_id
            metrics["model"] = model_name
            metrics["target_day_local"] = str(target_day_start_local.date())
            metrics["issue_time_local"] = str(issue_time_local)
            metrics["train_end_utc"] = train_end
            metrics["test_start_utc"] = test_start
            metrics["test_end_utc"] = test_end
            metrics["required_horizon"] = required_horizon
            all_metric_rows.append(metrics)

        fold_metrics = pd.DataFrame(
            [row for row in all_metric_rows if row["fold_id"] == fold_id]
        )

        print()
        print("Fold metrics:")
        print(
            fold_metrics[
                [
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

    # -------------------------------------------------------------------------
    # Final outputs
    # -------------------------------------------------------------------------
    if not all_prediction_rows:
        raise ValueError("No folds were completed.")

    predictions_df = pd.concat(all_prediction_rows).sort_index()
    metrics_by_fold = pd.DataFrame(all_metric_rows)

    predictions_hourly = predictions_df.copy()
    predictions_hourly.index = predictions_hourly.index.tz_convert(OUTPUT_TIMEZONE)
    predictions_hourly["local_hour"] = predictions_hourly.index.hour

    hourly_metric_rows = []

    for model_name in [
        "spotforecast",
        "weekly_persistence",
    ]:
        pred_col = model_name

        for hour, group in predictions_hourly.groupby("local_hour"):
            valid = group.dropna(subset=["actual_load", pred_col])

            if valid.empty:
                continue

            error = valid[pred_col] - valid["actual_load"]
            abs_error = error.abs()

            hourly_metric_rows.append(
                {
                    "model": model_name,
                    "local_hour": int(hour),
                    "n_points": len(valid),
                    "MAE": abs_error.mean(),
                    "RMSE": np.sqrt((error**2).mean()),
                    "MAPE_percent": (
                        abs_error / valid["actual_load"].abs()
                ).mean() * 100,
                "Bias": error.mean(),
                "MedianError": error.median(),
                "MaxAbsError": abs_error.max(),
            }
        )

    hourly_metrics_df = pd.DataFrame(hourly_metric_rows)

    summary_rows = []

    for model_name, group in metrics_by_fold.groupby("model"):
        valid = group.dropna(subset=["MAPE_percent"])

        if valid.empty:
            continue

        best_row = valid.loc[valid["MAPE_percent"].idxmin()]
        worst_row = valid.loc[valid["MAPE_percent"].idxmax()]

        summary_rows.append(
            {
                "model": model_name,
                "n_folds": len(valid),
                "mean_MAE": valid["MAE"].mean(),
                "median_MAE": valid["MAE"].median(),
                "best_MAE": valid["MAE"].min(),
                "worst_MAE": valid["MAE"].max(),
                "mean_RMSE": valid["RMSE"].mean(),
                "median_RMSE": valid["RMSE"].median(),
                "mean_MAPE_percent": valid["MAPE_percent"].mean(),
                "median_MAPE_percent": valid["MAPE_percent"].median(),
                "best_MAPE_percent": valid["MAPE_percent"].min(),
                "worst_MAPE_percent": valid["MAPE_percent"].max(),
                "mean_Bias": valid["Bias"].mean(),
                "median_Bias": valid["Bias"].median(),
                "best_fold_id": int(best_row["fold_id"]),
                "best_target_day_local": best_row["target_day_local"],
                "worst_fold_id": int(worst_row["fold_id"]),
                "worst_target_day_local": worst_row["target_day_local"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    predictions_file = output_dir / "rolling_backtest_predictions_entsoe_exog.csv"
    metrics_file = output_dir / "rolling_backtest_metrics_by_fold_entsoe_exog.csv"
    summary_file = output_dir / "rolling_backtest_summary_entsoe_exog.csv"
    hourly_metrics_file = output_dir / "rolling_backtest_metrics_by_local_hour.csv"
    hourly_metrics_df.to_csv(hourly_metrics_file, index=False)

    # Save prediction timestamps in local Europe/Berlin time.
    predictions_output = predictions_df.copy()
    predictions_output.index = predictions_output.index.tz_convert(OUTPUT_TIMEZONE)
    predictions_output = predictions_output.rename_axis("timestamp")

    predictions_output.to_csv(predictions_file)
    metrics_by_fold.to_csv(metrics_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print()
    print("=" * 80)
    print("ROLLING BACKTEST SUMMARY")
    print("=" * 80)
    print()
    print(summary_df.to_string(index=False))

    print()
    print("Saved files:")
    print(predictions_file)
    print(metrics_file)
    print(summary_file)
    print(hourly_metrics_file)


if __name__ == "__main__":
    main()