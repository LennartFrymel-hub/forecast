import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)


def main() -> None:
    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))

    print(f"Datenordner: {data_home}")

    actual = load_timeseries(
        data_home=data_home,
        on_missing="ffill_bfill",
    )

    forecast = load_timeseries_forecast(
        data_home=data_home,
        on_missing="ffill_bfill",
    )

    print()
    print("Actual Load:")
    print(actual.head())
    print(actual.tail())
    print(actual.describe())

    print()
    print("Forecasted Load:")
    print(forecast.head())
    print(forecast.tail())
    print(forecast.describe())

    print()
    print("Zeitraum Actual Load:")
    print(actual.index.min(), "bis", actual.index.max())

    print()
    print("Fehlende Werte Actual Load:")
    print(actual.isna().sum())

    print()
    print("Fehlende Werte Forecasted Load:")
    print(forecast.isna().sum())


if __name__ == "__main__":
    main()