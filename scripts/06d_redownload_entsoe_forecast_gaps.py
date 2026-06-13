import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.downloader.entsoe import (
    download_new_data,
    merge_build_manual,
)


BUFFER_HOURS = 24


def main() -> None:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is missing.")

    country_code = os.environ.get("ENTSOE_COUNTRY_CODE", "DE_LU")

    gaps_file = Path("outputs") / "entsoe_forecast_gap_spans.csv"

    if not gaps_file.exists():
        raise FileNotFoundError(
            f"Gap file not found: {gaps_file}. "
            "Run 06c_diagnose_entsoe_forecast_gaps.py first."
        )

    gaps = pd.read_csv(
        gaps_file,
        parse_dates=["gap_start", "gap_end"],
    )

    if gaps.empty:
        print("No gaps to redownload.")
        return

    print()
    print("Redownload ENTSO-E forecast gap windows")
    print(f"Country code: {country_code}")
    print()

    for _, row in gaps.iterrows():
        gap_start = pd.Timestamp(row["gap_start"])
        gap_end = pd.Timestamp(row["gap_end"])

        if gap_start.tzinfo is None:
            gap_start = gap_start.tz_localize("UTC")
        else:
            gap_start = gap_start.tz_convert("UTC")

        if gap_end.tzinfo is None:
            gap_end = gap_end.tz_localize("UTC")
        else:
            gap_end = gap_end.tz_convert("UTC")

        download_start = gap_start - pd.Timedelta(hours=BUFFER_HOURS)
        download_end = gap_end + pd.Timedelta(hours=BUFFER_HOURS + 1)

        print(
            f"Redownload window: {download_start} to {download_end} "
            f"(gap: {gap_start} to {gap_end})"
        )

        download_new_data(
            api_key=api_key,
            country_code=country_code,
            start=download_start,
            end=download_end,
            force=True,
        )

    print()
    print("Merge refreshed raw files...")
    merge_build_manual(output_file="energy_load.csv")

    print()
    print("Finished redownload and merge.")


if __name__ == "__main__":
    main()