import os
from pathlib import Path

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)


def calc_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    df = pd.concat(
        [
            actual.rename("actual"),
            pred.rename("prediction"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_percent": np.nan,
            "Bias": np.nan,
            "MaxAbsError": np.nan,
        }

    error = df["prediction"] - df["actual"]
    abs_error = error.abs()

    return {
        "n": len(df),
        "MAE": abs_error.mean(),
        "RMSE": np.sqrt((error**2).mean()),
        "MAPE_percent": (abs_error / df["actual"].abs()).mean() * 100,
        "Bias": error.mean(),
        "MaxAbsError": abs_error.max(),
    }


def main() -> None:
    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))

    print("ENTSO-E alignment diagnosis")
    print(f"Data folder: {data_home}")

    actual = load_timeseries(
        data_home=data_home,
        on_missing="passthrough",
    ).rename("actual_load")

    forecast = load_timeseries_forecast(
        data_home=data_home,
        on_missing="passthrough",
    ).rename("entsoe_forecast")

    df = pd.concat([actual, forecast], axis=1).sort_index()

    print()
    print("Data range:")
    print(df.index.min(), "to", df.index.max())
    print("Rows:", len(df))
    print("Missing actual:", int(df["actual_load"].isna().sum()))
    print("Missing forecast:", int(df["entsoe_forecast"].isna().sum()))

    print()
    print("Original ENTSO-E metrics, no shift:")
    print(calc_metrics(df["actual_load"], df["entsoe_forecast"]))

    print()
    print("Shift test")
    print("Meaning:")
    print("shift_hours = +1 means forecast is moved 1 hour later")
    print("shift_hours = -1 means forecast is moved 1 hour earlier")
    print()

    rows = []
    for shift_hours in range(-12, 13):
        shifted_forecast = df["entsoe_forecast"].shift(
            periods=shift_hours,
            freq="h",
        )

        metrics = calc_metrics(df["actual_load"], shifted_forecast)
        metrics["shift_hours"] = shift_hours
        rows.append(metrics)

    result = pd.DataFrame(rows)
    result = result[
        [
            "shift_hours",
            "n",
            "MAE",
            "RMSE",
            "MAPE_percent",
            "Bias",
            "MaxAbsError",
        ]
    ].sort_values("MAE")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "entsoe_alignment_shift_test.csv"
    result.to_csv(output_file, index=False)

    print(result.head(15).to_string(index=False))

    print()
    print("Best shift:")
    print(result.head(1).to_string(index=False))

    print()
    print("Saved:")
    print(output_file)


if __name__ == "__main__":
    main()