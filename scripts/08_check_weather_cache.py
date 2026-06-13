import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.data.fetch_data import load_timeseries
from spotforecast2_safe.manager.exo.multi_weather import validate_multi_weather_cache


FORECAST_HORIZON = 24

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


def main() -> None:
    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))
    weather_cache_home = Path("weather_cache") / "de_multi"

    print("Weather cache preflight check")
    print(f"Data folder:    {data_home}")
    print(f"Weather cache:  {weather_cache_home}")

    actual_raw = load_timeseries(
        data_home=data_home,
        on_missing="passthrough",
    )

    last_actual_time = actual_raw.dropna().index.max()

    if pd.isna(last_actual_time):
        raise ValueError("No valid actual load values found.")

    last_actual_time = last_actual_time.floor("h")

    required_start = actual_raw.index.min().floor("h")
    required_end = last_actual_time + pd.Timedelta(hours=FORECAST_HORIZON)

    print()
    print("Required weather coverage:")
    print(f"Start: {required_start}")
    print(f"End:   {required_end}")

    result = validate_multi_weather_cache(
        start=required_start,
        cov_end=required_end,
        locations=DE_WEATHER_LOCATIONS,
        timezone="UTC",
        freq="h",
        cache_home=weather_cache_home,
        expected_columns=EXPECTED_WEATHER_COLUMNS,
        allow_missing_before_fill=True,
        raise_on_error=True,
        verbose=True,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "weather_cache_validation.csv"
    result.to_csv(output_file, index=False)

    print()
    print("Weather cache is valid.")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()