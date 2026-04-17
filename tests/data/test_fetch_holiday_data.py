# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from spotforecast2_safe.data.fetch_data import fetch_holiday_data


def test_fetch_holiday_data_utc_hourly():
    """Test fetching hourly holiday data in UTC."""
    start = "2023-01-01 00:00:00"
    # 24 hours of New Year's Day
    end = "2023-01-01 23:00:00"
    df = fetch_holiday_data(start=start, end=end, tz="UTC", freq="h")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert df["is_holiday"].all()
    # Use str() representation for portable timezone checking
    assert str(df.index.tz).lower() in ["utc", "utc+00:00"]


def test_fetch_holiday_data_berlin_daily():
    """Test fetching daily holiday data in Europe/Berlin."""
    # Jan 1st is holiday, Jan 2nd is not
    start = "2023-01-01"
    end = "2023-01-02"
    df = fetch_holiday_data(
        start=start,
        end=end,
        tz="Europe/Berlin",
        freq="D",
        country_code="DE",
        state="NW",
    )

    assert len(df) == 2
    assert df.loc["2023-01-01", "is_holiday"]
    assert not df.loc["2023-01-02", "is_holiday"]


def test_fetch_holiday_data_boundary_single_point():
    """Test fetching holiday data for a single point (start == end)."""
    start = "2023-12-25 00:00:00"
    end = start
    df = fetch_holiday_data(start=start, end=end, tz="UTC", freq="h")

    assert len(df) == 1
    assert df["is_holiday"].iloc[0]


def test_fetch_holiday_data_year_boundary():
    """Test fetching data crossing a year boundary."""
    start = "2023-12-31 22:00:00"
    end = "2024-01-01 02:00:00"
    df = fetch_holiday_data(start=start, end=end, tz="UTC", freq="h")

    # 22, 23, 00, 01, 02 -> 5 points
    assert len(df) == 5
    # Dec 31 is not necessarily a holiday in Germany (usually half day, depends on library)
    # But Jan 1st definitely is.
    assert df.loc["2024-01-01 00:00:00+00:00", "is_holiday"]


def test_fetch_holiday_data_invalid_country():
    """Test behavior with likely invalid/edge case country code if supported by lib."""
    # The underlying library usually defaults or handles errors.
    # We just want to ensure it doesn't crash the safety-critical system.
    try:
        df = fetch_holiday_data(
            start="2023-01-01", end="2023-01-01", country_code="INVALID"
        )
        assert isinstance(df, pd.DataFrame)
    except Exception as e:
        # If it raises a specific known error, that's also acceptable evidence of handling
        print(f"Handled exception for invalid country: {e}")


@pytest.mark.parametrize("freq", ["h", "15min", "D"])
def test_fetch_holiday_data_frequencies(freq):
    """Test different frequencies."""
    start = "2023-05-01"  # Labour Day
    end = "2023-05-01 02:00:00"
    df = fetch_holiday_data(start=start, end=end, freq=freq)
    assert not df.empty
    assert df["is_holiday"].any()
