"""Tests for validation utilities."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.exceptions import DataTypeWarning, MissingValuesWarning
from spotforecast2_safe.utils.validation import (  # MissingValuesWarning, # Imported locally in function if needed, or from exceptions; DataTypeWarning, # Imported locally or from exceptions
    check_exog,
    check_exog_dtypes,
    check_interval,
    check_predict_input,
    check_y,
    get_exog_dtypes,
)


class TestCheckY:
    """Test check_y validation function."""

    def test_check_y_valid_series(self):
        """Test check_y with valid pandas Series."""
        y = pd.Series([1, 2, 3, 4, 5], name="y")
        # Should not raise any exception
        check_y(y)

    def test_check_y_valid_series_with_datetime_index(self):
        """Test check_y with DatetimeIndex."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, name="y")
        # Should not raise any exception
        check_y(y)

    def test_check_y_valid_series_with_range_index(self):
        """Test check_y with RangeIndex."""
        y = pd.Series(range(20), index=pd.RangeIndex(start=0, stop=20), name="y")
        # Should not raise any exception
        check_y(y)

    def test_check_y_invalid_type_list(self):
        """Test check_y raises TypeError for list input."""
        with pytest.raises(TypeError, match="must be a pandas Series"):
            check_y([1, 2, 3, 4, 5])

    def test_check_y_invalid_type_numpy_array(self):
        """Test check_y raises TypeError for numpy array input."""
        with pytest.raises(TypeError, match="must be a pandas Series"):
            check_y(np.array([1, 2, 3, 4, 5]))

    def test_check_y_invalid_type_dataframe(self):
        """Test check_y raises TypeError for DataFrame input."""
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        with pytest.raises(TypeError, match="must be a pandas Series"):
            check_y(df)

    def test_check_y_invalid_type_dict(self):
        """Test check_y raises TypeError for dict input."""
        with pytest.raises(TypeError, match="must be a pandas Series"):
            check_y({"a": 1, "b": 2})

    def test_check_y_invalid_type_none(self):
        """Test check_y raises TypeError for None input."""
        with pytest.raises(TypeError, match="must be a pandas Series"):
            check_y(None)

    def test_check_y_with_missing_values_nan(self):
        """Test check_y raises ValueError when series contains NaN."""
        y = pd.Series([1, 2, np.nan, 4, 5])
        with pytest.raises(ValueError, match="has missing values"):
            check_y(y)

    def test_check_y_with_multiple_missing_values(self):
        """Test check_y raises ValueError with multiple NaN values."""
        y = pd.Series([np.nan, 2, 3, np.nan, 5])
        with pytest.raises(ValueError, match="has missing values"):
            check_y(y)

    def test_check_y_with_all_missing_values(self):
        """Test check_y raises ValueError when all values are NaN."""
        y = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="has missing values"):
            check_y(y)

    def test_check_y_custom_series_id(self):
        """Test check_y uses custom series_id in error messages."""
        y_list = [1, 2, 3]
        with pytest.raises(TypeError, match="`train_series` must be a pandas Series"):
            check_y(y_list, series_id="`train_series`")

    def test_check_y_custom_series_id_with_nan(self):
        """Test check_y uses custom series_id for missing values error."""
        y = pd.Series([1, np.nan, 3])
        with pytest.raises(ValueError, match="`my_series` has missing values"):
            check_y(y, series_id="`my_series`")

    def test_check_y_with_zero_values(self):
        """Test check_y accepts series with zero values (not NaN)."""
        y = pd.Series([0, 0, 0, 0])
        # Should not raise - zeros are valid
        check_y(y)

    def test_check_y_with_negative_values(self):
        """Test check_y accepts series with negative values."""
        y = pd.Series([-5, -2, -1, 0, 1, 2])
        # Should not raise - negative values are valid
        check_y(y)

    def test_check_y_empty_series(self):
        """Test check_y with empty series."""
        y = pd.Series([], dtype=float)
        # Should not raise - empty series is technically valid (no missing values)
        check_y(y)

    def test_check_y_single_element_series(self):
        """Test check_y with single element series."""
        y = pd.Series([42])
        # Should not raise
        check_y(y)

    def test_check_y_with_float_values(self):
        """Test check_y accepts series with float values."""
        y = pd.Series([1.5, 2.3, 3.7, 4.2])
        # Should not raise
        check_y(y)

    def test_check_y_returns_none(self):
        """Test that check_y returns None on success."""
        y = pd.Series([1, 2, 3])
        result = check_y(y)
        assert result is None


class TestCheckExog:
    """Test check_exog validation function."""

    def test_check_exog_valid_dataframe(self):
        """Test check_exog with valid DataFrame."""
        exog = pd.DataFrame({"temp": [20, 21, 22], "humidity": [50, 55, 60]})
        # Should not raise
        check_exog(exog)

    def test_check_exog_valid_series_with_name(self):
        """Test check_exog with valid Series that has a name."""
        exog = pd.Series([1, 2, 3], name="temperature")
        # Should not raise
        check_exog(exog)

    def test_check_exog_invalid_series_without_name(self):
        """Test check_exog raises ValueError for Series without name."""
        exog = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="must have a name"):
            check_exog(exog)

    def test_check_exog_invalid_type_list(self):
        """Test check_exog raises TypeError for list."""
        with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
            check_exog([1, 2, 3])

    def test_check_exog_invalid_type_numpy_array(self):
        """Test check_exog raises TypeError for numpy array."""
        with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
            check_exog(np.array([1, 2, 3]))

    def test_check_exog_invalid_type_dict(self):
        """Test check_exog raises TypeError for dict."""
        with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
            check_exog({"a": 1, "b": 2})

    def test_check_exog_with_nan_allow_true(self):
        """Test check_exog allows NaN when allow_nan=True (default)."""
        exog = pd.DataFrame({"temp": [20, np.nan, 22]})
        # Should not raise or warn when allow_nan=True
        check_exog(exog, allow_nan=True)

    def test_check_exog_with_nan_allow_false(self):
        """Test check_exog warns about NaN when allow_nan=False."""
        exog = pd.DataFrame({"temp": [20, np.nan, 22]})
        with pytest.warns(MissingValuesWarning, match="has missing values"):
            check_exog(exog, allow_nan=False)

    def test_check_exog_custom_series_id(self):
        """Test check_exog uses custom series_id in error messages."""
        with pytest.raises(TypeError, match="`my_exog` must be a pandas Series"):
            check_exog([1, 2, 3], series_id="`my_exog`")

    def test_check_exog_empty_dataframe(self):
        """Test check_exog with empty DataFrame."""
        exog = pd.DataFrame()
        # Should not raise
        check_exog(exog)

    def test_check_exog_returns_none(self):
        """Test that check_exog returns None on success."""
        exog = pd.Series([1, 2, 3], name="test")
        result = check_exog(exog)
        assert result is None


class TestGetExogDtypes:
    """Test get_exog_dtypes function."""

    def test_get_exog_dtypes_dataframe(self):
        """Test get_exog_dtypes with DataFrame."""
        exog = pd.DataFrame(
            {
                "temp": pd.Series([20.5, 21.3], dtype="float64"),
                "day": pd.Series([1, 2], dtype="int64"),
            }
        )
        dtypes = get_exog_dtypes(exog)

        assert isinstance(dtypes, dict)
        assert "temp" in dtypes
        assert "day" in dtypes
        assert dtypes["temp"] == np.dtype("float64")
        assert dtypes["day"] == np.dtype("int64")

    def test_get_exog_dtypes_series(self):
        """Test get_exog_dtypes with Series."""
        exog = pd.Series([1.0, 2.0, 3.0], name="temperature", dtype="float64")
        dtypes = get_exog_dtypes(exog)

        assert isinstance(dtypes, dict)
        assert "temperature" in dtypes
        assert dtypes["temperature"] == np.dtype("float64")

    def test_get_exog_dtypes_mixed_types(self):
        """Test get_exog_dtypes with mixed data types."""
        exog = pd.DataFrame(
            {
                "float_col": pd.Series([1.5, 2.5], dtype="float64"),
                "int_col": pd.Series([1, 2], dtype="int64"),
                "bool_col": pd.Series([True, False], dtype="bool"),
            }
        )
        dtypes = get_exog_dtypes(exog)

        assert len(dtypes) == 3
        assert dtypes["float_col"] == np.dtype("float64")
        assert dtypes["int_col"] == np.dtype("int64")
        assert dtypes["bool_col"] == np.dtype("bool")

    def test_get_exog_dtypes_single_column_dataframe(self):
        """Test get_exog_dtypes with single-column DataFrame."""
        exog = pd.DataFrame({"temp": [20, 21, 22]})
        dtypes = get_exog_dtypes(exog)

        assert len(dtypes) == 1
        assert "temp" in dtypes


class TestCheckInterval:
    """Test check_interval validation function."""

    def test_check_interval_valid_95_percent(self):
        """Test check_interval with valid 95% interval."""
        # Should not raise
        check_interval(interval=[2.5, 97.5])

    def test_check_interval_valid_90_percent(self):
        """Test check_interval with valid 90% interval."""
        # Should not raise
        check_interval(interval=[5, 95])

    def test_check_interval_valid_tuple(self):
        """Test check_interval accepts tuple."""
        # Should not raise
        check_interval(interval=(2.5, 97.5))

    def test_check_interval_none(self):
        """Test check_interval allows None."""
        # Should not raise
        check_interval(interval=None)

    def test_check_interval_invalid_type_string(self):
        """Test check_interval raises TypeError for string."""
        with pytest.raises(TypeError, match="must be a `list` or `tuple`"):
            check_interval(interval="[2.5, 97.5]")

    def test_check_interval_invalid_type_dict(self):
        """Test check_interval raises TypeError for dict."""
        with pytest.raises(TypeError, match="must be a `list` or `tuple`"):
            check_interval(interval={2.5: 97.5})

    def test_check_interval_wrong_length_one_value(self):
        """Test check_interval raises ValueError for single value."""
        with pytest.raises(ValueError, match="must contain exactly 2 values"):
            check_interval(interval=[50])

    def test_check_interval_wrong_length_three_values(self):
        """Test check_interval raises ValueError for three values."""
        with pytest.raises(ValueError, match="must contain exactly 2 values"):
            check_interval(interval=[2.5, 50, 97.5])

    def test_check_interval_lower_bound_negative(self):
        """Test check_interval raises ValueError for negative lower bound."""
        with pytest.raises(ValueError, match="Lower interval bound.*must be >= 0"):
            check_interval(interval=[-5, 95])

    def test_check_interval_lower_bound_too_high(self):
        """Test check_interval raises ValueError for lower bound >= 100."""
        with pytest.raises(ValueError, match="Lower interval bound.*must be.*< 100"):
            check_interval(interval=[100, 105])

    def test_check_interval_upper_bound_zero(self):
        """Test check_interval raises ValueError for upper bound <= 0."""
        with pytest.raises(ValueError, match="Upper interval bound.*must be > 0"):
            check_interval(interval=[0, 0])

    def test_check_interval_upper_bound_too_high(self):
        """Test check_interval raises ValueError for upper bound > 100."""
        with pytest.raises(ValueError, match="Upper interval bound.*must be.*<= 100"):
            check_interval(interval=[5, 105])

    def test_check_interval_lower_greater_than_upper(self):
        """Test check_interval raises ValueError when lower >= upper."""
        with pytest.raises(ValueError, match="Lower interval bound.*must be less than"):
            check_interval(interval=[95, 5])

    def test_check_interval_equal_bounds(self):
        """Test check_interval raises ValueError for equal bounds."""
        with pytest.raises(ValueError, match="must be less than"):
            check_interval(interval=[50, 50])

    def test_check_interval_symmetric_valid(self):
        """Test check_interval with valid symmetric interval."""
        # Should not raise
        check_interval(interval=[2.5, 97.5], ensure_symmetric_intervals=True)

    def test_check_interval_symmetric_invalid(self):
        """Test check_interval raises ValueError for non-symmetric when required."""
        with pytest.raises(ValueError, match="Interval must be symmetric"):
            check_interval(interval=[5, 90], ensure_symmetric_intervals=True)

    def test_check_interval_edge_case_0_100(self):
        """Test check_interval with edge case [0.1, 99.9]."""
        # Very wide interval
        check_interval(interval=[0.1, 99.9])  # Should not raise

    def test_check_interval_small_interval(self):
        """Test check_interval with very small interval."""
        # Should not raise
        check_interval(interval=[49, 51])

    def test_check_interval_returns_none(self):
        """Test that check_interval returns None on success."""
        result = check_interval(interval=[2.5, 97.5])
        assert result is None


class TestCheckExogDtypes:
    """Test check_exog_dtypes function."""

    def test_check_exog_dtypes_valid_dataframe(self):
        """Test with valid DataFrame dtypes (int, float)."""
        exog = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3], "col3": [1, 2, 3]}
        )
        exog["col3"] = exog["col3"].astype("int64")
        check_exog_dtypes(exog)

    def test_check_exog_dtypes_valid_series(self):
        """Test with valid Series dtype."""
        exog = pd.Series([1, 2, 3], name="exog")
        check_exog_dtypes(exog)

    def test_check_exog_dtypes_valid_categorical(self):
        """Test with valid categorical dtype (integer categories)."""
        exog = pd.DataFrame({"col1": [1, 2, 1]})
        exog["col1"] = exog["col1"].astype("category")
        check_exog_dtypes(exog)

    def test_check_exog_dtypes_warning_invalid_dtype_dataframe(self):
        """Test DataTypeWarning for invalid dtypes in DataFrame (object)."""
        exog = pd.DataFrame({"col1": ["a", "b", "c"]})
        with pytest.warns(
            DataTypeWarning, match="may contain only `int`, `float` or `category`"
        ):
            check_exog_dtypes(exog)

    def test_check_exog_dtypes_warning_invalid_dtype_series(self):
        """Test DataTypeWarning for invalid dtype in Series (object)."""
        exog = pd.Series(["a", "b", "c"], name="exog")
        with pytest.warns(
            DataTypeWarning, match="may contain only `int`, `float` or `category`"
        ):
            check_exog_dtypes(exog)

    def test_check_exog_dtypes_error_categorical_non_int(self):
        """Test TypeError for categorical column with non-integer categories."""
        exog = pd.DataFrame({"col1": ["a", "b", "a"]})
        exog["col1"] = exog["col1"].astype("category")
        with pytest.raises(
            TypeError,
            match="Categorical dtypes in exog must contain only integer values",
        ):
            check_exog_dtypes(exog)


class TestCheckPredictInput:
    """Test check_predict_input function."""

    def test_check_predict_input_not_fitted(self):
        """Test check_predict_input raises RuntimeError if not fitted."""
        with pytest.raises(RuntimeError, match="This forecaster is not fitted yet"):
            check_predict_input(
                forecaster_name="forecaster",
                steps=5,
                is_fitted=False,
                exog_in_=False,
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=None,
            )

    def test_check_predict_input_steps_invalid_int(self):
        """Test check_predict_input raises ValueError for steps < 1."""
        with pytest.raises(ValueError, match="steps.*greater than or equal to 1"):
            check_predict_input(
                forecaster_name="forecaster",
                steps=0,
                is_fitted=True,
                exog_in_=False,
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=pd.Series([1, 2, 3]),
            )

    def test_check_predict_input_steps_invalid_list(self):
        """Test check_predict_input raises ValueError for list of steps with value < 1."""
        with pytest.raises(ValueError, match="steps.*greater than or equal to 1"):
            check_predict_input(
                forecaster_name="forecaster",
                steps=[1, 0, 3],
                is_fitted=True,
                exog_in_=False,
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=pd.Series([1, 2, 3]),
            )

    def test_check_predict_input_missing_exog(self):
        """Test check_predict_input raises ValueError if exog missing but required."""
        with pytest.raises(
            ValueError, match="Forecaster trained with exogenous variable/s"
        ):
            check_predict_input(
                forecaster_name="forecaster",
                steps=5,
                is_fitted=True,
                exog_in_=True,  # Requires exog
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=pd.Series([1, 2, 3]),
                exog=None,
            )

    def test_check_predict_input_unexpected_exog(self):
        """Test check_predict_input raises ValueError if exog provided but not required."""
        with pytest.raises(
            ValueError, match="Forecaster trained without exogenous variable/s"
        ):
            check_predict_input(
                forecaster_name="forecaster",
                steps=5,
                is_fitted=True,
                exog_in_=False,  # Does not require exog
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=pd.Series([1, 2, 3]),
                exog=pd.DataFrame({"a": [1, 2, 3]}),
            )

    def test_check_predict_input_valid(self):
        """Test check_predict_input with valid inputs."""
        check_predict_input(
            forecaster_name="forecaster",
            steps=5,
            is_fitted=True,
            exog_in_=False,
            index_type_=pd.DatetimeIndex,
            index_freq_="D",
            window_size=5,
            last_window=pd.Series([1, 2, 3], name="y"),
            exog=None,
        )

    def test_check_predict_input_valid_with_exog(self):
        """Test check_predict_input with valid exog inputs."""
        exog = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        check_predict_input(
            forecaster_name="forecaster",
            steps=5,
            is_fitted=True,
            exog_in_=True,
            index_type_=pd.DatetimeIndex,
            index_freq_="D",
            window_size=5,
            last_window=pd.Series([1, 2, 3], name="y"),
            exog=exog,
            exog_names_in_=["col1"],
        )

    def test_check_predict_input_invalid_exog_columns(self):
        """Test check_predict_input raises ValueError for mismatched exog columns."""
        exog = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})  # Different column name
        with pytest.raises(ValueError, match="Missing columns in `exog`"):
            check_predict_input(
                forecaster_name="forecaster",
                steps=5,
                is_fitted=True,
                exog_in_=True,
                index_type_=pd.DatetimeIndex,
                index_freq_="D",
                window_size=5,
                last_window=pd.Series([1, 2, 3], name="y"),
                exog=exog,
                exog_names_in_=["col1"],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
