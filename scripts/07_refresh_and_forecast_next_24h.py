import logging
import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    get_data_home,
    load_timeseries,
    load_timeseries_forecast,
)
from spotforecast2_safe.manager.exo.entsoe_forecast import (
    build_entsoe_forecast_exog,
    fetch_entsoe_future_forecast,
    repair_historical_entsoe_forecast,
)
from spotforecast2_safe.downloader.entsoe import (
    download_new_data,
    merge_build_manual,
)
from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
    n_to_1_with_covariates,
)
from spotforecast2_safe.manager.exo.multi_weather import validate_multi_weather_cache


FORECAST_HORIZON = 24
FULL_REFRESH_START = "2022-01-01 00:00"
INCREMENTAL_BUFFER_DAYS = 7
ENTSOE_SAFETY_DELAY_HOURS = 3

LAGS = [
    1, 2, 3,
    23, 24, 25,
    47, 48,
    167, 168, 169,
    336,
]

ROLLING_WINDOW_SIZES = [72, 168, 720]

CONTAMINATION_LEVEL = 0.005


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

TRAIN_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")


EXPECTED_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


def month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    current = start

    while current < end:
        next_month = current + pd.offsets.MonthBegin(1)
        chunk_end = min(next_month, end)
        yield current, chunk_end
        current = chunk_end

def get_last_timestamp_from_energy_file(energy_file: Path) -> pd.Timestamp | None:
    """Return the latest timestamp from an existing energy_load.csv file."""
    if not energy_file.exists():
        return None

    df = pd.read_csv(energy_file, usecols=["Time (UTC)"])

    if df.empty:
        return None

    timestamps = pd.to_datetime(df["Time (UTC)"], utc=True, errors="coerce")
    timestamps = timestamps.dropna()

    if timestamps.empty:
        return None

    return timestamps.max()




def get_last_complete_actual_hour_from_energy_file(
    energy_file: Path,
    min_samples_per_hour: int = 4,
) -> pd.Timestamp | None:
    """Return latest complete actual-load hour from 15-minute ENTSO-E CSV."""
    if not energy_file.exists():
        return None

    df = pd.read_csv(energy_file)

    if df.empty:
        return None

    time_col = "Time (UTC)" if "Time (UTC)" in df.columns else df.columns[0]

    actual_candidates = [
        col for col in df.columns if "Actual" in col and "Load" in col
    ]

    if not actual_candidates:
        return None

    actual_col = actual_candidates[0]

    timestamps = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    values = pd.to_numeric(df[actual_col], errors="coerce")

    tmp = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": values,
        }
    ).dropna(subset=["timestamp", "actual"])

    if tmp.empty:
        return None

    tmp["hour"] = tmp["timestamp"].dt.floor("h")

    counts = tmp.groupby("hour")["actual"].count()
    complete_hours = counts[counts >= min_samples_per_hour]

    if complete_hours.empty:
        return None

    return pd.Timestamp(complete_hours.index.max()).tz_convert("UTC")


def refresh_entsoe_data() -> Path:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY is missing. Set it in PowerShell with:\n"
            '$env:ENTSOE_API_KEY="YOUR_TOKEN"'
        )

    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))
    os.environ["SPOTFORECAST2_DATA"] = str(data_home)

    country_code = os.environ.get("ENTSOE_COUNTRY_CODE", "DE")

    raw_dir = data_home / "raw"
    interim_dir = data_home / "interim"

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    energy_file = interim_dir / "energy_load.csv"

    last_timestamp = get_last_timestamp_from_energy_file(energy_file)

    end = (
        pd.Timestamp.now(tz="Europe/Berlin").floor("h")
        - pd.Timedelta(hours=ENTSOE_SAFETY_DELAY_HOURS)
    )

    if last_timestamp is None:
        start = pd.Timestamp(FULL_REFRESH_START, tz="Europe/Berlin")
        refresh_mode = "full"
    else:
        start = (
            last_timestamp
            .tz_convert("Europe/Berlin")
            .floor("h")
            - pd.Timedelta(days=INCREMENTAL_BUFFER_DAYS)
        )
        refresh_mode = "incremental"

    if start >= end:
        print()
        print("ENTSO-E refresh skipped")
        print(f"Reason: start >= end")
        print(f"Start:  {start}")
        print(f"End:    {end}")
        print(f"Existing file: {energy_file}")

        if not energy_file.exists():
            raise FileNotFoundError(
                f"No energy_load.csv exists at {energy_file}, "
                "but refresh was skipped."
            )

        return energy_file

    print()
    print("Refresh ENTSO-E data")
    print(f"Mode:         {refresh_mode}")
    print(f"Country code: {country_code}")
    print(f"Start:        {start}")
    print(f"End:          {end}")
    print(f"Data folder:  {data_home}")
    print(f"Energy file:  {energy_file}")

    if last_timestamp is not None:
        print(f"Last existing timestamp: {last_timestamp}")
        print(f"Buffer days:             {INCREMENTAL_BUFFER_DAYS}")

    print()

    for chunk_start, chunk_end in month_ranges(start, end):
        print(f"Download block: {chunk_start} to {chunk_end}")

        download_new_data(
            api_key=api_key,
            country_code=country_code,
            start=chunk_start,
            end=chunk_end,
            force=True,
        )

    print()
    print("Merge raw ENTSO-E files into energy_load.csv")
    merge_build_manual(output_file="energy_load.csv")

    energy_file = get_data_home() / "interim" / "energy_load.csv"

    if not energy_file.exists():
        raise FileNotFoundError(
            f"energy_load.csv was not created. Expected path: {energy_file}"
        )

    refreshed_last_timestamp = get_last_timestamp_from_energy_file(energy_file)

    print()
    print("Data refresh finished")
    print(f"Energy file: {energy_file}")
    print(f"Latest timestamp after refresh: {refreshed_last_timestamp}")

    return energy_file


def forecast_next_24h(energy_file: Path | None = None) -> None:
    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))
    model_root = Path(os.environ.get("SPOTFORECAST2_CACHE", Path.cwd() / "models_cache"))

    print()
    print("Load actual load data")
    print(f"Data folder:  {data_home}")
    print(f"Model folder: {model_root}")

    actual_raw = load_timeseries(
        data_home=data_home,
        on_missing="passthrough",
    )

    actual_filled = load_timeseries(
        data_home=data_home,
        on_missing="ffill_bfill",
    )

    historical_entsoe_forecast = load_timeseries_forecast(
        data_home=data_home,
        on_missing="passthrough",
    )

    last_actual_time_raw = actual_raw.dropna().index.max()

    if pd.isna(last_actual_time_raw):
        raise ValueError("No valid actual load values found.")

    last_actual_time_raw = last_actual_time_raw.floor("h")

    last_complete_hour = None
    if energy_file is not None:
        last_complete_hour = get_last_complete_actual_hour_from_energy_file(
            energy_file=energy_file,
            min_samples_per_hour=4,
        )

    if last_complete_hour is not None:
        last_actual_time = min(last_actual_time_raw, last_complete_hour)
    else:
        last_actual_time = last_actual_time_raw

    print()
    print("Actual-load frontier")
    print(f"Last raw actual hour:       {last_actual_time_raw}")
    print(f"Last complete actual hour:  {last_complete_hour}")
    print(f"Used last actual hour:      {last_actual_time}")

    train_series = actual_filled.loc[TRAIN_START:last_actual_time].copy()

    if train_series.empty:
        raise ValueError(
            f"No training data available from {TRAIN_START} "
            f"to {last_actual_time}."
        )

    if len(train_series) < 500:
        raise ValueError(
            f"Too little training data: {len(train_series)} rows. "
            "Use more history."
        )

    train_data = train_series.rename("load_de").to_frame()

    print()
    print("Training window")
    print(f"Training start: {train_data.index.min()}")
    print(f"Training end:   {train_data.index.max()}")
    print(f"Training rows:  {len(train_data)}")

    model_forecast_start = last_actual_time + pd.Timedelta(hours=1)

    # Ziel: exakt der naechste Kalendertag in deutscher Ortszeit
    now_local = pd.Timestamp.now(tz="Europe/Berlin")
    next_day_start_local = now_local.normalize() + pd.Timedelta(days=1)
    next_day_end_local = next_day_start_local + pd.Timedelta(hours=23)

    # Intern arbeitet die Pipeline mit UTC
    forecast_start = next_day_start_local.tz_convert("UTC")
    forecast_end = next_day_end_local.tz_convert("UTC")

    # Das Modell muss intern ab dem letzten Messwert bis zum Ende
    # des naechsten Kalendertages prognostizieren.
    required_horizon = int(
        (forecast_end - model_forecast_start) / pd.Timedelta(hours=1)
    ) + 1

    entsoe_api_key = os.environ.get("ENTSOE_API_KEY")
    entsoe_country_code = os.environ.get("ENTSOE_COUNTRY_CODE", "DE")

    try:
        future_entsoe_forecast = fetch_entsoe_future_forecast(
            api_key=entsoe_api_key,
            country_code=entsoe_country_code,
            start=model_forecast_start,
            end=forecast_end,
        )
    except Exception as exc:
        raise RuntimeError(
            "ENTSO-E forecast for the required future period is not fully "
            "available. Run the script later or use the version without "
            "ENTSO-E exogenous features."
        ) from exc

    

    # -------------------------------------------------------------------------
    # Historical ENTSO-E forecast coverage check
    # -------------------------------------------------------------------------
    # The model can only use ENTSO-E forecast as an exogenous feature if this
    # feature is complete for the entire training period. If the historical
    # forecast series has gaps, we move the training start to the first hour
    # after the last missing historical ENTSO-E forecast value.
    repaired_historical_entsoe_forecast = repair_historical_entsoe_forecast(
        entsoe_forecast=historical_entsoe_forecast,
        start=train_data.index.min(),
        end=last_actual_time,
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
        end=forecast_end,
    )

    if required_horizon <= 0:
        raise ValueError(
            f"Invalid required_horizon: {required_horizon}. "
            f"model_forecast_start={model_forecast_start}, "
            f"forecast_end={forecast_end}"
        )

    print()
    print("Forecast setup")
    print(f"Last actual timestamp:     {last_actual_time}")
    print(f"Model forecast start:      {model_forecast_start}")
    print(f"Output day local:          {next_day_start_local.date()}")
    print(f"Output start UTC:          {forecast_start}")
    print(f"Output end UTC:            {forecast_end}")
    print(f"Internal model horizon:    {required_horizon} hours")
    print(f"Returned output horizon:   {FORECAST_HORIZON} hours")
    print(f"Training rows:             {len(train_data)}")

    weather_cache_home = Path("weather_cache") / "de_multi"

    print()
    print("Validate weather cache before training")

    validate_multi_weather_cache(
        start=train_data.index.min(),
        cov_end=forecast_end,
        locations=DE_WEATHER_LOCATIONS,
        timezone="UTC",
        freq="h",
        cache_home=weather_cache_home,
        expected_columns=EXPECTED_WEATHER_COLUMNS,
        allow_missing_before_fill=True,
        raise_on_error=True,
        verbose=True,
    )
    print()
    print("Start training and prediction")

    predictions, combined_prediction, model_metrics, feature_info = n_to_1_with_covariates(
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
        weather_cache_home=weather_cache_home,
        timezone="UTC",
        country_code="DE",
        state="NW",
        contamination=CONTAMINATION_LEVEL,
        include_weather_windows=True,
        include_holiday_features=True,
        include_poly_features=False,
        weights={"load_de": 1.0},
        force_train=True,
        model_dir=model_root / "de_load_next_calendar_day_entsoe_exog",
        verbose=True,
        show_progress=True,
    )

    # Nur die 24 Stunden des gewuenschten naechsten Kalendertages ausgeben
    predictions = predictions.loc[forecast_start:forecast_end]
    combined_prediction = combined_prediction.loc[forecast_start:forecast_end]

    if len(combined_prediction) != FORECAST_HORIZON:
        raise ValueError(
            "Final output does not contain exactly 24 hourly forecast values. "
            f"Expected {FORECAST_HORIZON}, got {len(combined_prediction)}."
        )

    # Convert final outputs from UTC to German local time for easier
    # comparison with the ENTSO-E website.
    output_timezone = "Europe/Berlin"

    predictions_output = predictions.copy()
    predictions_output.index = predictions_output.index.tz_convert(output_timezone)

    combined_prediction_output = combined_prediction.copy()
    combined_prediction_output.index = combined_prediction_output.index.tz_convert(
        output_timezone
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    predictions_file = output_dir / "de_load_predictions_next_24h.csv"
    forecast_file = output_dir / "de_load_forecast_next_24h.csv"

    predictions_output.to_csv(predictions_file)
    combined_prediction_output.rename("forecast_load_de").to_csv(forecast_file)

    print()
    print("=" * 80)
    print("NEXT CALENDAR DAY FORECAST - LOCAL TIME EUROPE/BERLIN")
    print("=" * 80)
    print()
    print(combined_prediction_output.rename("forecast_load_de").to_string())

    print()
    print("Saved files:")
    print(predictions_file)
    print(forecast_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    energy_file = refresh_entsoe_data()
    forecast_next_24h(energy_file=energy_file)


if __name__ == "__main__":
    main()