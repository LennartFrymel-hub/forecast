from datetime import datetime

import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.curate_data import (
    get_start_end,
    remove_duplicate_timestamps,
)


class TestGetStartEnd:
    """Test suite for the get_start_end function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with datetime index."""
        date_rng = pd.date_range(start="2023-01-01", end="2023-01-10", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"
        return data

    @pytest.fixture
    def single_row_data(self):
        """Create sample data with single row."""
        date_rng = pd.date_range(start="2023-01-01", periods=1, freq="h")
        data = pd.DataFrame({"value": [1]}, index=date_rng)
        data.index.name = "date"
        return data

    @pytest.fixture
    def long_range_data(self):
        """Create sample data with long date range."""
        date_rng = pd.date_range(start="2020-01-01", end="2023-12-31", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"
        return data

    def test_get_start_end_returns_tuple(self, sample_data):
        """Test that get_start_end returns a tuple of 4 elements."""
        result = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_get_start_end_returns_strings(self, sample_data):
        """Test that get_start_end returns strings."""
        start, end, cov_start, cov_end = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        assert isinstance(start, str)
        assert isinstance(end, str)
        assert isinstance(cov_start, str)
        assert isinstance(cov_end, str)

    def test_get_start_end_datetime_format(self, sample_data):
        """Test that returned dates are in correct format YYYY-MM-DDTHH:MM."""
        start, end, cov_start, cov_end = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        # Check format by trying to parse
        datetime.strptime(start, "%Y-%m-%dT%H:%M")
        datetime.strptime(end, "%Y-%m-%dT%H:%M")
        datetime.strptime(cov_start, "%Y-%m-%dT%H:%M")
        datetime.strptime(cov_end, "%Y-%m-%dT%H:%M")

    def test_get_start_end_start_equals_data_min(self, sample_data):
        """Test that start equals minimum date in data."""
        start, _, _, _ = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        expected_start = sample_data.index.min().strftime("%Y-%m-%dT%H:%M")
        assert start == expected_start

    def test_get_start_end_end_equals_data_max(self, sample_data):
        """Test that end equals maximum date in data."""
        _, end, _, _ = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        expected_end = sample_data.index.max().strftime("%Y-%m-%dT%H:%M")
        assert end == expected_end

    def test_get_start_end_cov_start_equals_data_start(self, sample_data):
        """Test that covariate start equals data start."""
        start, _, cov_start, _ = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        assert start == cov_start

    def test_get_start_end_cov_end_extended_by_horizon(self, sample_data):
        """Test that covariate end is extended by forecast horizon."""
        _, end, _, cov_end = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        end_dt = pd.to_datetime(end)
        cov_end_dt = pd.to_datetime(cov_end)
        expected_cov_end = end_dt + pd.Timedelta(hours=24)
        assert cov_end_dt == expected_cov_end

    def test_get_start_end_different_forecast_horizons(self, sample_data):
        """Test get_start_end with different forecast horizons."""
        _, end, _, cov_end_24 = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        _, _, _, cov_end_48 = get_start_end(
            sample_data, forecast_horizon=48, verbose=False
        )
        _, _, _, cov_end_12 = get_start_end(
            sample_data, forecast_horizon=12, verbose=False
        )

        end_dt = pd.to_datetime(end)
        cov_end_24_dt = pd.to_datetime(cov_end_24)
        cov_end_48_dt = pd.to_datetime(cov_end_48)
        cov_end_12_dt = pd.to_datetime(cov_end_12)

        assert cov_end_24_dt == end_dt + pd.Timedelta(hours=24)
        assert cov_end_48_dt == end_dt + pd.Timedelta(hours=48)
        assert cov_end_12_dt == end_dt + pd.Timedelta(hours=12)

    def test_get_start_end_single_row(self, single_row_data):
        """Test get_start_end with single row dataframe."""
        start, end, cov_start, cov_end = get_start_end(
            single_row_data, forecast_horizon=24, verbose=False
        )
        assert start == end
        assert cov_start == start
        # cov_end should be 24 hours after start
        start_dt = pd.to_datetime(start)
        cov_end_dt = pd.to_datetime(cov_end)
        assert cov_end_dt == start_dt + pd.Timedelta(hours=24)

    def test_get_start_end_zero_forecast_horizon(self, sample_data):
        """Test get_start_end with zero forecast horizon."""
        _, end, _, cov_end = get_start_end(
            sample_data, forecast_horizon=0, verbose=False
        )
        end_dt = pd.to_datetime(end)
        cov_end_dt = pd.to_datetime(cov_end)
        assert cov_end_dt == end_dt

    def test_get_start_end_large_forecast_horizon(self, sample_data):
        """Test get_start_end with large forecast horizon."""
        _, end, _, cov_end = get_start_end(
            sample_data, forecast_horizon=720, verbose=False
        )
        end_dt = pd.to_datetime(end)
        cov_end_dt = pd.to_datetime(cov_end)
        assert cov_end_dt == end_dt + pd.Timedelta(hours=720)

    def test_get_start_end_long_range_data(self, long_range_data):
        """Test get_start_end with long date range."""
        start, end, cov_start, cov_end = get_start_end(
            long_range_data, forecast_horizon=24, verbose=False
        )
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        cov_start_dt = pd.to_datetime(cov_start)
        cov_end_dt = pd.to_datetime(cov_end)

        assert start_dt == long_range_data.index.min()
        assert end_dt == long_range_data.index.max()
        assert start_dt == cov_start_dt
        assert cov_end_dt == end_dt + pd.Timedelta(hours=24)

    def test_get_start_end_verbose_true(self, sample_data, capsys):
        """Test that verbose=True prints output."""
        get_start_end(sample_data, forecast_horizon=24, verbose=True)
        captured = capsys.readouterr()
        assert "Data range:" in captured.out
        assert "Covariate data range:" in captured.out

    def test_get_start_end_verbose_false(self, sample_data, capsys):
        """Test that verbose=False does not print output."""
        get_start_end(sample_data, forecast_horizon=24, verbose=False)
        captured = capsys.readouterr()
        assert "Data range:" not in captured.out
        assert "Covariate data range:" not in captured.out

    def test_get_start_end_verbose_default(self, sample_data, capsys):
        """Test that verbose defaults to True."""
        get_start_end(sample_data, forecast_horizon=24)
        captured = capsys.readouterr()
        assert "Data range:" in captured.out
        assert "Covariate data range:" in captured.out

    def test_get_start_end_order_of_results(self, sample_data):
        """Test that start and end are in correct order."""
        start, end, cov_start, cov_end = get_start_end(
            sample_data, forecast_horizon=24, verbose=False
        )
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        cov_start_dt = pd.to_datetime(cov_start)
        cov_end_dt = pd.to_datetime(cov_end)

        assert start_dt <= end_dt
        assert cov_start_dt <= cov_end_dt
        assert start_dt <= cov_start_dt

    def test_get_start_end_hourly_frequency(self):
        """Test get_start_end with hourly frequency data."""
        date_rng = pd.date_range(start="2023-01-01 12:30", periods=100, freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, cov_start, cov_end = get_start_end(
            data, forecast_horizon=24, verbose=False
        )

        # Check minute precision is preserved
        assert "12:30" in start
        assert "12:30" in cov_start

    def test_get_start_end_daily_frequency(self):
        """Test get_start_end with daily frequency data."""
        date_rng = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, _, _ = get_start_end(data, forecast_horizon=24, verbose=False)

        # Check that time is 00:00 for daily data
        assert "00:00" in start
        assert "00:00" in end

    def test_get_start_end_example_from_docstring(self):
        """Test example from docstring."""
        date_rng = pd.date_range(start="2023-01-01", end="2023-01-10", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, cov_start, cov_end = get_start_end(
            data, forecast_horizon=24, verbose=False
        )

        assert start == "2023-01-01T00:00"
        assert end == "2023-01-10T00:00"
        assert cov_start == "2023-01-01T00:00"
        assert cov_end == "2023-01-11T00:00"

    def test_get_start_end_preserves_data(self, sample_data):
        """Test that get_start_end doesn't modify the input data."""
        original_data = sample_data.copy()
        get_start_end(sample_data, forecast_horizon=24, verbose=False)
        pd.testing.assert_frame_equal(sample_data, original_data)

    def test_get_start_end_with_timezone_aware_index(self):
        """Test get_start_end with timezone-aware datetime index."""
        date_rng = pd.date_range(
            start="2023-01-01", end="2023-01-10", freq="h", tz="UTC"
        )
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, cov_start, cov_end = get_start_end(
            data, forecast_horizon=24, verbose=False
        )

        # Should still return valid datetime strings
        datetime.strptime(start, "%Y-%m-%dT%H:%M")
        datetime.strptime(end, "%Y-%m-%dT%H:%M")
        datetime.strptime(cov_start, "%Y-%m-%dT%H:%M")
        datetime.strptime(cov_end, "%Y-%m-%dT%H:%M")

    def test_get_start_end_leap_year(self):
        """Test get_start_end with leap year data."""
        date_rng = pd.date_range(start="2020-02-28", end="2020-03-01", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, _, _ = get_start_end(data, forecast_horizon=24, verbose=False)

        assert "2020-02-28" in start
        assert "2020-03-01" in end

    def test_get_start_end_year_boundary(self):
        """Test get_start_end crossing year boundary."""
        date_rng = pd.date_range(start="2022-12-31", end="2023-01-02", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        start, end, _, cov_end = get_start_end(data, forecast_horizon=24, verbose=False)

        assert "2022" in start
        assert "2023" in end
        assert "2023" in cov_end

    def test_get_start_end_month_boundary(self):
        """Test get_start_end crossing month boundary."""
        date_rng = pd.date_range(start="2023-01-31", end="2023-02-02", freq="h")
        data = pd.DataFrame({"value": range(len(date_rng))}, index=date_rng)
        data.index.name = "date"

        _, end, _, cov_end = get_start_end(data, forecast_horizon=24, verbose=False)

        assert "2023-02-02" in end
        assert "2023-02" in cov_end

    def test_get_start_end_consistency_across_calls(self, sample_data):
        """Test that multiple calls return consistent results."""
        result1 = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        result2 = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        assert result1 == result2

    def test_get_start_end_forecast_horizon_parameter_types(self, sample_data):
        """Test get_start_end with different forecast_horizon parameter types."""
        # Integer
        result_int = get_start_end(sample_data, forecast_horizon=24, verbose=False)
        assert len(result_int) == 4

        # Float (should still work)
        result_float = get_start_end(sample_data, forecast_horizon=24.0, verbose=False)
        assert len(result_float) == 4


class TestGetStartEndEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_start_end_very_small_dataframe(self):
        """Test with very small dataframe."""
        date_rng = pd.date_range(start="2023-01-01", periods=2, freq="h")
        data = pd.DataFrame({"value": [1, 2]}, index=date_rng)
        data.index.name = "date"

        start, end, cov_start, cov_end = get_start_end(
            data, forecast_horizon=24, verbose=False
        )

        assert start is not None
        assert end is not None
        assert cov_start is not None
        assert cov_end is not None

    def test_get_start_end_irregular_frequency(self):
        """Test with irregular datetime index."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05", "2023-01-10"])
        data = pd.DataFrame({"value": [1, 2, 3, 4]}, index=dates)
        data.index.name = "date"

        start, end, cov_start, cov_end = get_start_end(
            data, forecast_horizon=24, verbose=False
        )

        assert start == "2023-01-01T00:00"
        assert end == "2023-01-10T00:00"


class TestRemoveDuplicateTimestamps:
    """Test suite for remove_duplicate_timestamps."""

    def test_averages_and_sorts_multi_column(self):
        """Mean-aggregates duplicate timestamps across multiple columns."""
        df = pd.DataFrame(
            {
                "Time (UTC)": pd.to_datetime(
                    [
                        "2026-01-01 01:00:00",
                        "2026-01-01 00:00:00",
                        "2026-01-01 00:00:00",
                    ],
                    utc=True,
                ),
                "Load A": [130.0, 100.0, 120.0],
                "Load B": [50.0, 10.0, 30.0],
            }
        )

        out = remove_duplicate_timestamps(df=df)

        assert len(out) == 2
        assert out["Time (UTC)"].is_monotonic_increasing
        assert out.loc[0, "Load A"] == pytest.approx(110.0)
        assert out.loc[1, "Load A"] == pytest.approx(130.0)
        assert out.loc[0, "Load B"] == pytest.approx(20.0)

    def test_raises_on_missing_time_col(self):
        """Raises KeyError when time_col is absent from the dataframe."""
        df = pd.DataFrame({"ts": ["2026-01-01 00:00:00"], "load": [1.0]})

        with pytest.raises(KeyError):
            remove_duplicate_timestamps(df=df)

    def test_custom_time_col(self):
        """Respects a user-supplied time_col name."""
        df = pd.DataFrame(
            {
                "ts": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-02 00:00:00",
                ],
                "value": [10.0, 30.0, 50.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, time_col="ts")

        assert len(out) == 2
        assert out.loc[0, "value"] == pytest.approx(20.0)

    def test_agg_median(self):
        """Median aggregation takes the midpoint, not the mean."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01", "2026-01-01", "2026-01-01"],
                "value": [10.0, 20.0, 90.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, agg="median")

        assert float(out.loc[0, "value"]) == pytest.approx(20.0)

    def test_agg_min(self):
        """Min aggregation returns the smallest value per group."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01", "2026-01-01"],
                "value": [10.0, 90.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, agg="min")

        assert float(out.loc[0, "value"]) == pytest.approx(10.0)

    def test_agg_max(self):
        """Max aggregation returns the largest value per group."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01", "2026-01-01"],
                "value": [10.0, 90.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, agg="max")

        assert float(out.loc[0, "value"]) == pytest.approx(90.0)

    def test_agg_mode(self):
        """Mode aggregation returns the most frequent value per group."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01", "2026-01-01", "2026-01-01"],
                "value": [10.0, 10.0, 90.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, agg="mode")

        assert float(out.loc[0, "value"]) == pytest.approx(10.0)

    def test_agg_callable(self):
        """Accepts a custom callable aggregation function."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01", "2026-01-01"],
                "value": [10.0, 90.0],
            }
        )

        out = remove_duplicate_timestamps(df=df, agg=lambda x: 42.0)

        assert float(out.loc[0, "value"]) == pytest.approx(42.0)

    def test_no_duplicates_unchanged_length(self):
        """Frame with no duplicates passes through with the same row count."""
        df = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
                "value": [1.0, 2.0],
            }
        )

        out = remove_duplicate_timestamps(df=df)

        assert len(out) == 2
