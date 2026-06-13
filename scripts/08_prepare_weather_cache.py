import time
from pathlib import Path

import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_weather_data


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
    cache_root = Path("weather_cache") / "de_multi"

    cov_start = pd.Timestamp("2022-01-01 00:00", tz="UTC")
    cov_end = pd.Timestamp.now(tz="UTC").ceil("h") + pd.Timedelta(hours=48)

    print("Prepare weather cache")
    print(f"Start: {cov_start}")
    print(f"End:   {cov_end}")
    print(f"Cache: {cache_root}")

    for index, location in enumerate(DE_WEATHER_LOCATIONS, start=1):
        name = location["name"]
        location_cache = cache_root / name
        location_cache.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 80)
        print(f"{index}/{len(DE_WEATHER_LOCATIONS)} Fetch weather for {name}")
        print("=" * 80)

        try:
            weather = fetch_weather_data(
                cov_start=cov_start,
                cov_end=cov_end,
                latitude=location["latitude"],
                longitude=location["longitude"],
                timezone="UTC",
                freq="h",
                fallback_on_failure=True,
                cache_home=location_cache,
                fill_missing=True,
            )

            print(weather.head())
            print(weather.tail())
            print(f"Shape: {weather.shape}")

        except Exception as exc:
            print(f"FAILED for {name}: {type(exc).__name__}: {exc}")
            print("Stop now. Wait 30 to 60 minutes and run again.")
            raise

        print("Sleep 20 seconds to avoid API rate limit")
        time.sleep(20)

    print()
    print("Weather cache finished")


if __name__ == "__main__":
    main()