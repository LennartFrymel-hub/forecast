import pandas as pd

from spotforecast2_safe.utils.generate_holiday import create_holiday_df


def test_create_holiday_df_christmas():
    """Test standard Christmas holidays in Germany."""
    start = "2023-12-24"
    end = "2023-12-26"
    df = create_holiday_df(start, end, freq="D", country_code="DE", state="NW")

    # 24th is usually not a full public holiday in DE/NW (Heiligabend),
    # but 25th and 26th are.
    # checking holidays library behavior for DE:
    # 24th is not a holiday by default in python-holidays for DE unless specifically added or custom.
    # 25th (1st Xmas Day) -> 1
    # 26th (2nd Xmas Day) -> 1

    assert df.loc["2023-12-25", "is_holiday"] == 1
    assert df.loc["2023-12-26", "is_holiday"] == 1
    # Note: Depending on holidays version, 24th might be treated differently,
    # but typically it's 0 or 0.5 if handled as half-day.
    # Assuming 0 based on original test.
    if "2023-12-24" in df.index:
        assert df.loc["2023-12-24", "is_holiday"] == 0


def test_hourly_frequency():
    """Test that hourly frequency works and fills the whole day."""
    start = "2023-12-25 00:00"
    end = "2023-12-25 23:00"
    df = create_holiday_df(start, end, freq="h", country_code="DE", state="NW")

    assert len(df) == 24
    assert df["is_holiday"].sum() == 24  # All hours should be holidays


def test_timezone_handling():
    """Test explicit timezone."""
    start = "2023-12-25"
    end = "2023-12-25"
    df = create_holiday_df(start, end, freq="D", tz="Europe/Berlin", country_code="DE")

    assert str(df.index.tz) == "Europe/Berlin"
    assert df.iloc[0]["is_holiday"] == 1


def test_inferred_timezone():
    """Test inferred timezone from Timestamp."""
    start = pd.Timestamp("2023-12-25", tz="US/Pacific")
    end = pd.Timestamp("2023-12-26", tz="US/Pacific")
    df = create_holiday_df(start, end, freq="D", country_code="US", state="CA")

    assert str(df.index.tz) == "US/Pacific"
    assert df.iloc[0]["is_holiday"] == 1  # Xmas


def test_empty_range():
    """Test behavior with single point or empty range if needed, though pandas handles this."""
    start = "2023-01-01"
    end = "2023-01-01"
    df = create_holiday_df(start, end, freq="D", country_code="DE")  # New Year
    assert len(df) == 1
    assert df.iloc[0]["is_holiday"] == 1
