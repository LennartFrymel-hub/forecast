"""
Comprehensive tests for check_preprocess_series function.

Safety-critical requirements:
- Validates all input types and formats
- Ensures data consistency across multi-series
- Detects missing or malformed data
- Provides clear error messages for debugging
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.exceptions import IgnoredArgumentWarning, InputTypeWarning
from spotforecast2_safe.forecaster.utils import check_preprocess_series


class TestCheckPreprocessSeriesBasicValidation:
    """Test basic input validation."""

    def test_invalid_input_type(self):
        """Reject non-DataFrame and non-dict inputs."""
        with pytest.raises(TypeError, match="`series` must be a pandas DataFrame"):
            check_preprocess_series([1, 2, 3])

        with pytest.raises(TypeError, match="`series` must be a pandas DataFrame"):
            check_preprocess_series("invalid")

        with pytest.raises(TypeError, match="`series` must be a pandas DataFrame"):
            check_preprocess_series(12.34)

    def test_empty_dict_raises_error(self):
        """Empty dictionary raises ValueError due to no valid indexes."""
        with pytest.raises(ValueError, match="Found frequencies:"):
            check_preprocess_series({})


class TestCheckPreprocessSeriesDictInput:
    """Test dictionary input format."""

    def test_dict_with_series_valid(self):
        """Valid dictionary with pandas Series."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
            "series_2": pd.Series(np.arange(10, 20), index=dates, name="series_2"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert isinstance(result_dict, dict)
        assert isinstance(result_indexes, dict)
        assert len(result_dict) == 2
        assert "series_1" in result_dict
        assert "series_2" in result_dict
        assert result_dict["series_1"].name == "series_1"
        assert result_dict["series_2"].name == "series_2"

    def test_dict_with_dataframe_single_column_valid(self):
        """Valid dictionary with single-column DataFrames."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.DataFrame({"value": np.arange(10)}, index=dates),
            "series_2": pd.DataFrame({"value": np.arange(10, 20)}, index=dates),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert len(result_dict) == 2
        assert all(isinstance(s, pd.Series) for s in result_dict.values())

    def test_dict_with_mixed_series_and_dataframe(self):
        """Valid dictionary with mixed Series and single-column DataFrames."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
            "series_2": pd.DataFrame({"value": np.arange(10, 20)}, index=dates),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert len(result_dict) == 2
        assert all(isinstance(s, pd.Series) for s in result_dict.values())

    def test_dict_with_invalid_series_type(self):
        """Reject dictionary with non-Series/DataFrame values."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
            "series_2": [1, 2, 3],  # Invalid: list
        }

        with pytest.raises(TypeError, match="all series must be a named"):
            check_preprocess_series(series_dict)

    def test_dict_with_multi_column_dataframe_raises_error(self):
        """Reject DataFrames with multiple columns in dictionary."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.DataFrame(
                {"value1": np.arange(10), "value2": np.arange(10, 20)}, index=dates
            ),
        }

        with pytest.raises(ValueError, match="must be a named pandas Series"):
            check_preprocess_series(series_dict)


class TestCheckPreprocessSeriesIndexType:
    """Test index type validation."""

    def test_datetimeindex_valid(self):
        """DatetimeIndex with frequency is valid."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)
        assert isinstance(result_indexes["series_1"], pd.DatetimeIndex)

    def test_rangeindex_valid(self):
        """RangeIndex with step is valid."""
        series_dict = {
            "series_1": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 10, 1), name="series_1"
            ),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)
        assert isinstance(result_indexes["series_1"], pd.RangeIndex)

    def test_invalid_index_type(self):
        """Reject invalid index types."""
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=np.arange(10), name="series_1"),
        }

        with pytest.raises(
            TypeError, match="must have a Pandas RangeIndex or DatetimeIndex"
        ):
            check_preprocess_series(series_dict)

    def test_datetimeindex_without_frequency_raises_error(self):
        """DatetimeIndex without frequency raises error."""
        # Create DatetimeIndex without frequency (no freq parameter)
        dates = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-04"]
        )  # Gap breaks frequency
        series_dict = {
            "series_1": pd.Series(np.arange(3), index=dates, name="series_1"),
        }

        with pytest.raises(ValueError, match="Found series with no frequency or step"):
            check_preprocess_series(series_dict)

    def test_rangeindex_valid_has_default_step(self):
        """RangeIndex always has a step (default 1)."""
        # RangeIndex without explicit step still has step=1
        series_dict = {
            "series_1": pd.Series(
                np.arange(10), index=pd.RangeIndex(10), name="series_1"
            ),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)
        assert isinstance(result_indexes["series_1"], pd.RangeIndex)
        assert result_indexes["series_1"].step == 1


class TestCheckPreprocessSeriesFrequencyConsistency:
    """Test frequency/step consistency validation."""

    def test_same_datetimeindex_frequencies_valid(self):
        """Series with same DatetimeIndex frequencies are valid."""
        dates_daily = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates_daily, name="series_1"),
            "series_2": pd.Series(np.arange(10), index=dates_daily, name="series_2"),
        }

        result_dict, _ = check_preprocess_series(series_dict)
        assert len(result_dict) == 2

    def test_different_datetimeindex_frequencies_raises_error(self):
        """Series with different DatetimeIndex frequencies raise error."""
        dates_daily = pd.date_range("2020-01-01", periods=10, freq="D")
        dates_2daily = pd.date_range("2020-01-01", periods=10, freq="2D")

        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates_daily, name="series_1"),
            "series_2": pd.Series(np.arange(10), index=dates_2daily, name="series_2"),
        }

        with pytest.raises(
            (ValueError, TypeError)
        ):  # May raise TypeError if sorting offsets fails
            check_preprocess_series(series_dict)

    def test_same_rangeindex_steps_valid(self):
        """Series with same RangeIndex steps are valid."""
        series_dict = {
            "series_1": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 10, 1), name="series_1"
            ),
            "series_2": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 10, 1), name="series_2"
            ),
        }

        result_dict, _ = check_preprocess_series(series_dict)
        assert len(result_dict) == 2

    def test_different_rangeindex_steps_raises_error(self):
        """Series with different RangeIndex steps raise error."""
        series_dict = {
            "series_1": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 10, 1), name="series_1"
            ),
            "series_2": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 20, 2), name="series_2"
            ),
        }

        with pytest.raises(ValueError, match="Found frequencies:"):
            check_preprocess_series(series_dict)

    def test_mixed_index_types_raises_error(self):
        """Series with mixed DatetimeIndex and RangeIndex raise error."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
            "series_2": pd.Series(
                np.arange(10), index=pd.RangeIndex(0, 10, 1), name="series_2"
            ),
        }

        with pytest.raises(
            (ValueError, TypeError)
        ):  # May raise TypeError if sorting offsets fails
            check_preprocess_series(series_dict)


class TestCheckPreprocessSeriesDataValidation:
    """Test data content validation."""

    def test_all_nan_series_raises_error(self):
        """Series with all NaN values raise error."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series([np.nan] * 10, index=dates, name="series_1"),
        }

        with pytest.raises(ValueError, match="All values of series 'series_1' are NaN"):
            check_preprocess_series(series_dict)

    def test_series_with_some_nan_valid(self):
        """Series with some NaN values is valid."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = np.arange(10, dtype=float)
        data[3] = np.nan  # One NaN value

        series_dict = {
            "series_1": pd.Series(data, index=dates, name="series_1"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)
        assert len(result_dict) == 1

    def test_series_with_zero_values_valid(self):
        """Series with zero values is valid."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series([0] * 10, index=dates, name="series_1"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)
        assert len(result_dict) == 1


class TestCheckPreprocessSeriesDataFrameWideFormat:
    """Test DataFrame wide-format input."""

    def test_wide_format_dataframe_valid(self):
        """Wide-format DataFrame with multiple columns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "series_1": np.arange(10),
                "series_2": np.arange(10, 20),
                "series_3": np.arange(20, 30),
            },
            index=dates,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_dict, result_indexes = check_preprocess_series(df)

            # Should raise InputTypeWarning
            assert len(w) == 1
            assert issubclass(w[0].category, InputTypeWarning)

        assert len(result_dict) == 3
        assert all(isinstance(s, pd.Series) for s in result_dict.values())

    def test_wide_format_single_column_dataframe(self):
        """Wide-format DataFrame with single column."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"series_1": np.arange(10)}, index=dates)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_dict, result_indexes = check_preprocess_series(df)
            assert len(w) == 1
            assert issubclass(w[0].category, InputTypeWarning)

        assert len(result_dict) == 1

    def test_wide_format_without_frequency_raises_error(self):
        """Wide-format DataFrame without frequency raises InputTypeWarning then ValueError."""
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])
        df = pd.DataFrame(
            {
                "series_1": np.arange(3),
                "series_2": np.arange(3, 6),
            },
            index=dates,
        )

        with pytest.warns(InputTypeWarning):
            with pytest.raises(ValueError):
                check_preprocess_series(df)


class TestCheckPreprocessSeriesDataFrameLongFormat:
    """Test DataFrame long-format (MultiIndex) input."""

    def test_long_format_multiindex_valid(self):
        """Long-format DataFrame with MultiIndex."""
        # Create MultiIndex: (series_id, date)
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        index = pd.MultiIndex.from_product(
            [["series_1", "series_2"], dates], names=["series_id", "date"]
        )
        df = pd.DataFrame({"value": np.arange(10)}, index=index)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_dict, result_indexes = check_preprocess_series(df)
            assert len(w) == 1
            assert issubclass(w[0].category, InputTypeWarning)

        assert len(result_dict) == 2
        assert "series_1" in result_dict
        assert "series_2" in result_dict

    def test_long_format_multiindex_multiple_columns_warning(self):
        """Long-format DataFrame with multiple columns raises warning."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        index = pd.MultiIndex.from_product(
            [["series_1", "series_2"], dates], names=["series_id", "date"]
        )
        df = pd.DataFrame(
            {
                "value1": np.arange(10),
                "value2": np.arange(10, 20),
            },
            index=index,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _ = check_preprocess_series(df)

            # Should have two warnings: IgnoredArgumentWarning and InputTypeWarning
            assert len(w) == 2
            warning_types = [warning.category for warning in w]
            assert IgnoredArgumentWarning in warning_types
            assert InputTypeWarning in warning_types

    def test_long_format_invalid_multiindex_structure(self):
        """Long-format DataFrame with non-DatetimeIndex second level raises error."""
        index = pd.MultiIndex.from_product(
            [["series_1", "series_2"], range(5)], names=["series_id", "time_step"]
        )
        df = pd.DataFrame({"value": np.arange(10)}, index=index)

        with pytest.raises(TypeError, match="second level of the MultiIndex"):
            check_preprocess_series(df)


class TestCheckPreprocessSeriesReturnValues:
    """Test return value format and content."""

    def test_return_structure(self):
        """Verify return structure is correct."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict_input = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
            "series_2": pd.Series(np.arange(10, 20), index=dates, name="series_2"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict_input)

        # Verify structure
        assert isinstance(result_dict, dict)
        assert isinstance(result_indexes, dict)
        assert set(result_dict.keys()) == set(result_indexes.keys())

    def test_series_are_copies(self):
        """Verify that returned series are copies, not references."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        original = pd.Series(np.arange(10), index=dates, name="series_1")
        series_dict_input = {"series_1": original}

        result_dict, _ = check_preprocess_series(series_dict_input)

        # Modify the returned series
        result_dict["series_1"].iloc[0] = 999

        # Original should not be modified
        assert original.iloc[0] == 0

    def test_index_preserved_in_return(self):
        """Verify that indexes are preserved in return values."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict_input = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
        }

        _, result_indexes = check_preprocess_series(series_dict_input)

        assert result_indexes["series_1"].equals(dates)

    def test_series_names_set_correctly(self):
        """Verify that series names are set to their keys."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict_input = {
            "my_series": pd.Series(np.arange(10), index=dates),
        }

        result_dict, _ = check_preprocess_series(series_dict_input)

        assert result_dict["my_series"].name == "my_series"


class TestCheckPreprocessSeriesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_series(self):
        """Single series in dictionary."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "only_series": pd.Series(np.arange(10), index=dates, name="only_series"),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert len(result_dict) == 1
        assert "only_series" in result_dict

    def test_many_series(self):
        """Many series in dictionary."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            f"series_{i}": pd.Series(
                np.arange(10) + i * 10, index=dates, name=f"series_{i}"
            )
            for i in range(100)
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert len(result_dict) == 100

    def test_very_short_series(self):
        """Series with minimal length."""
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        series_dict = {
            "series_1": pd.Series([1, 2], index=dates, name="series_1"),
        }

        result_dict, _ = check_preprocess_series(series_dict)

        assert len(result_dict["series_1"]) == 2

    def test_very_long_series(self):
        """Series with many observations."""
        dates = pd.date_range("2020-01-01", periods=5000, freq="h")
        series_dict = {
            "series_1": pd.Series(np.arange(5000), index=dates, name="series_1"),
        }

        result_dict, _ = check_preprocess_series(series_dict)

        assert len(result_dict["series_1"]) == 5000

    def test_negative_values_valid(self):
        """Series with negative values are valid."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(-10, 0), index=dates, name="series_1"),
        }

        result_dict, _ = check_preprocess_series(series_dict)

        assert len(result_dict) == 1

    def test_large_values_valid(self):
        """Series with very large values are valid."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(
                [1e15 + i for i in range(10)], index=dates, name="series_1"
            ),
        }

        result_dict, _ = check_preprocess_series(series_dict)

        assert len(result_dict) == 1


class TestCheckPreprocessSeriesIntegration:
    """Integration tests combining multiple features."""

    def test_real_world_scenario_multiple_daily_series(self):
        """Real-world scenario: multiple daily time series."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        series_dict = {
            "store_1_sales": pd.Series(
                np.random.randn(365).cumsum() + 100, index=dates, name="store_1_sales"
            ),
            "store_2_sales": pd.Series(
                np.random.randn(365).cumsum() + 120, index=dates, name="store_2_sales"
            ),
            "store_3_sales": pd.Series(
                np.random.randn(365).cumsum() + 95, index=dates, name="store_3_sales"
            ),
        }

        result_dict, result_indexes = check_preprocess_series(series_dict)

        assert len(result_dict) == 3
        assert all(len(v) == 365 for v in result_dict.values())

    def test_conversion_from_dataframe_to_dict(self):
        """Verify DataFrame conversion maintains data integrity."""
        dates = pd.date_range("2020-01-01", periods=50, freq="h")
        original_df = pd.DataFrame(
            {
                "device_1": np.arange(50, dtype=float),
                "device_2": np.arange(50, 100, dtype=float),
            },
            index=dates,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result_dict, _ = check_preprocess_series(original_df)

        # Verify data matches
        assert np.array_equal(
            result_dict["device_1"].values, original_df["device_1"].values
        )
        assert np.array_equal(
            result_dict["device_2"].values, original_df["device_2"].values
        )


class TestCheckPreprocessSeriesSafetyFeatures:
    """Test safety-critical features."""

    def test_no_data_corruption(self):
        """Verify function does not corrupt input data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        original_values = np.arange(10, dtype=float)
        original_values[2] = np.nan
        original_series = pd.Series(
            original_values.copy(), index=dates, name="series_1"
        )

        series_dict = {"series_1": original_series.copy()}

        check_preprocess_series(series_dict)

        # Original should not be modified
        assert np.array_equal(original_series.values, original_values, equal_nan=True)

    def test_deterministic_output(self):
        """Verify function behavior is deterministic."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series_dict = {
            "series_1": pd.Series(np.arange(10), index=dates, name="series_1"),
        }

        result1_dict, result1_idx = check_preprocess_series(series_dict)
        result2_dict, result2_idx = check_preprocess_series(series_dict)

        # Results should be identical
        assert result1_dict["series_1"].equals(result2_dict["series_1"])
        assert result1_idx["series_1"].equals(result2_idx["series_1"])

    def test_clear_error_messages(self):
        """Verify error messages are clear and actionable."""
        # Test missing frequency error message
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])  # Gap
        series_dict = {
            "series_1": pd.Series(np.arange(3), index=dates, name="series_1"),
        }

        with pytest.raises(ValueError) as exc_info:
            check_preprocess_series(series_dict)

        error_msg = str(exc_info.value)
        assert "frequency" in error_msg.lower() or "step" in error_msg.lower()
