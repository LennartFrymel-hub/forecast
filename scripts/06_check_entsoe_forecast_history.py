import os
from pathlib import Path

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)


EXCLUDE_LATEST_DAYS = 7
HISTORY_DAYS = 90


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

    print("ENTSO-E forecast history check")
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

    latest = df.dropna(subset=["actual_load"]).index.max()

    if pd.isna(latest):
        raise ValueError("No valid actual load values found.")

    # Do not evaluate the latest edge. It can contain incomplete actual data.
    end = latest.floor("h") - pd.Timedelta(days=EXCLUDE_LATEST_DAYS)
    start = end - pd.Timedelta(days=HISTORY_DAYS)

    df_window = df.loc[start:end]

    print()
    print("Evaluation window:")
    print(start, "to", end)
    print("Rows:", len(df_window))

    print()
    print("Overall metrics in window:")
    overall_metrics = calc_metrics(
        df_window["actual_load"],
        df_window["entsoe_forecast"],
    )
    print(overall_metrics)

    daily_rows = []

    for day, group in df_window.groupby(df_window.index.date):
        metrics = calc_metrics(
            group["actual_load"],
            group["entsoe_forecast"],
        )

        if metrics["n"] > 0:
            metrics["date"] = str(day)
            daily_rows.append(metrics)

    daily_df = pd.DataFrame(daily_rows)

    if daily_df.empty:
        raise ValueError("No daily metrics could be calculated.")

    daily_df = daily_df[
        [
            "date",
            "n",
            "MAE",
            "RMSE",
            "MAPE_percent",
            "Bias",
            "MaxAbsError",
        ]
    ]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    daily_file = output_dir / "entsoe_daily_metrics_90d.csv"
    summary_file = output_dir / "entsoe_history_summary_90d.csv"

    daily_df.to_csv(daily_file, index=False)

    summary = daily_df[["MAE", "RMSE", "MAPE_percent", "Bias", "MaxAbsError"]].describe()
    summary.to_csv(summary_file)

    print()
    print("Daily metrics, last 20 days in window:")
    print(daily_df.tail(20).to_string(index=False))

    print()
    print("MAPE percent summary:")
    print(daily_df["MAPE_percent"].describe())

    print()
    print("Worst days by MAPE:")
    print(
        daily_df.sort_values("MAPE_percent", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(daily_file)
    print(summary_file)


if __name__ == "__main__":
    main()