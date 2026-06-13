import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)


def build_gap_spans(missing_index: pd.DatetimeIndex) -> list[dict]:
    if len(missing_index) == 0:
        return []

    rows = []
    start = missing_index[0]
    previous = missing_index[0]

    for ts in missing_index[1:]:
        if ts - previous != pd.Timedelta(hours=1):
            rows.append(
                {
                    "gap_start": start,
                    "gap_end": previous,
                    "n_missing_hours": int(
                        (previous - start) / pd.Timedelta(hours=1)
                    ) + 1,
                }
            )
            start = ts

        previous = ts

    rows.append(
        {
            "gap_start": start,
            "gap_end": previous,
            "n_missing_hours": int(
                (previous - start) / pd.Timedelta(hours=1)
            ) + 1,
        }
    )

    return rows


def main() -> None:
    data_home = Path(
        os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe")
    )

    actual = load_timeseries(
        data_home=data_home,
        on_missing="passthrough",
    )

    forecast = load_timeseries_forecast(
        data_home=data_home,
        on_missing="passthrough",
    )

    missing_forecast = forecast[forecast.isna()]
    missing_index = pd.DatetimeIndex(missing_forecast.index)

    spans = build_gap_spans(missing_index)
    spans_df = pd.DataFrame(spans)

    if not spans_df.empty:
        spans_df["actual_missing_too"] = spans_df.apply(
            lambda row: actual.loc[row["gap_start"] : row["gap_end"]]
            .isna()
            .any(),
            axis=1,
        )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "entsoe_forecast_gap_spans.csv"
    spans_df.to_csv(output_file, index=False)

    print()
    print("ENTSO-E forecast gap diagnosis")
    print(f"Data folder: {data_home}")
    print(f"Missing hourly forecast values: {len(missing_index)}")
    print()

    if spans_df.empty:
        print("No forecast gaps found.")
    else:
        print(spans_df.to_string(index=False))

    print()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()