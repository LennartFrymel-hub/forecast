# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Union

import pandas as pd


ENTSOE_FORECAST_FEATURE = "entsoe_forecast_load"


def _to_utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Convert timestamp-like input to UTC."""
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _extract_hourly_forecast_series(
    data: Union[pd.Series, pd.DataFrame],
) -> pd.Series:
    """Extract and normalize ENTSO-E forecast series to hourly UTC."""
    if isinstance(data, pd.Series):
        series = data.copy()

    elif isinstance(data, pd.DataFrame):
        if "Forecasted Load" in data.columns:
            series = data["Forecasted Load"].copy()
        elif data.shape[1] == 1:
            series = data.iloc[:, 0].copy()
        else:
            raise ValueError(
                "Could not identify the ENTSO-E forecast column. "
                f"Available columns: {list(data.columns)}"
            )
    else:
        raise TypeError(
            "ENTSO-E forecast data must be a pandas Series or DataFrame."
        )

    series.index = pd.to_datetime(series.index, utc=True)
    series = pd.to_numeric(series, errors="coerce")
    series = series[~series.index.duplicated(keep="last")].sort_index()

    # Your current project also reduces the ENTSO-E data to hourly values.
    series = series.asfreq("h")

    return series


def build_entsoe_forecast_exog(
    entsoe_forecast: pd.Series,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    column_name: str = ENTSOE_FORECAST_FEATURE,
) -> pd.DataFrame:
    """Build a complete hourly ENTSO-E forecast feature frame."""
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)

    if end_ts < start_ts:
        raise ValueError(
            f"end must be >= start. Got start={start_ts}, end={end_ts}."
        )

    expected_index = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq="h",
        tz="UTC",
    )

    series = _extract_hourly_forecast_series(entsoe_forecast)
    series = series.reindex(expected_index)

    missing = series[series.isna()]
    if not missing.empty:
        preview = ", ".join(str(ts) for ts in missing.index[:5])
        raise ValueError(
            "ENTSO-E future forecast is not fully available for the requested "
            f"period. First missing timestamps: {preview}"
        )

    return series.rename(column_name).to_frame()


def fetch_entsoe_future_forecast(
    api_key: str,
    country_code: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    column_name: str = ENTSOE_FORECAST_FEATURE,
) -> pd.Series:
    """Fetch ENTSO-E load forecast for a future prediction window."""
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is missing.")

    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError(
            "The 'entsoe-py' package is required for future ENTSO-E forecasts."
        ) from exc

    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)

    if end_ts < start_ts:
        raise ValueError(
            f"end must be >= start. Got start={start_ts}, end={end_ts}."
        )

    # ENTSO-E interval ends are safer when requested one hour beyond
    # the desired final timestamp and then sliced back.
    request_end = end_ts + pd.Timedelta(hours=1)

    client = EntsoePandasClient(api_key=api_key)

    raw_forecast = client.query_load_forecast(
        country_code=country_code,
        start=start_ts,
        end=request_end,
    )

    series = _extract_hourly_forecast_series(raw_forecast)

    expected_index = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq="h",
        tz="UTC",
    )

    series = series.reindex(expected_index)

    missing = series[series.isna()]
    if not missing.empty:
        preview = ", ".join(str(ts) for ts in missing.index[:5])
        raise ValueError(
            "ENTSO-E future forecast is not fully available for the requested "
            f"period. First missing timestamps: {preview}"
        )

    return series.rename(column_name)

def repair_historical_entsoe_forecast(
    entsoe_forecast: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Repair sparse historical gaps in the ENTSO-E forecast feature.

    Repair strategy:
    1. Keep all original values.
    2. Fill missing values with the same hour one week earlier.
    3. Fill remaining gaps with the same hour one week later.
    4. Fill very short remaining interior gaps by time interpolation.
    5. Fail if values are still missing.
    """
    expected_index = pd.date_range(
        start=start,
        end=end,
        freq="h",
        tz="UTC",
    )

    series = entsoe_forecast.copy().sort_index()

    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    series = series.reindex(expected_index)
    original_missing = int(series.isna().sum())

    if original_missing == 0:
        return series

    repaired = series.copy()

    missing_before_prev_week = int(repaired.isna().sum())
    repaired = repaired.fillna(series.shift(168))
    filled_from_prev_week = missing_before_prev_week - int(repaired.isna().sum())

    missing_before_next_week = int(repaired.isna().sum())
    repaired = repaired.fillna(series.shift(-168))
    filled_from_next_week = missing_before_next_week - int(repaired.isna().sum())

    missing_before_interp = int(repaired.isna().sum())
    repaired = repaired.interpolate(
        method="time",
        limit=6,
        limit_direction="both",
    )
    filled_by_interpolation = missing_before_interp - int(repaired.isna().sum())

    remaining_missing = int(repaired.isna().sum())

    print()
    print("Repair historical ENTSO-E forecast feature")
    print(f"Original missing hours:        {original_missing}")
    print(f"Filled from previous week:     {filled_from_prev_week}")
    print(f"Filled from next week:         {filled_from_next_week}")
    print(f"Filled by short interpolation: {filled_by_interpolation}")
    print(f"Remaining missing hours:       {remaining_missing}")

    if remaining_missing != 0:
        remaining = repaired[repaired.isna()]
        preview = ", ".join(str(ts) for ts in remaining.index[:5])
        raise ValueError(
            "Historical ENTSO-E forecast could not be fully repaired. "
            f"First remaining missing timestamps: {preview}"
        )

    return repaired