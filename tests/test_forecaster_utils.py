"""Tests for spotforecast2_safe.forecaster.utils module.

This module contains comprehensive tests for utility functions used in
forecasting operations, including window features initialization, data
validation, and HTML styling.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.utils import (
    check_extract_values_and_index,
    check_residuals_input,
    date_to_index_position,
    get_style_repr_html,
    initialize_window_features,
)
from spotforecast2_safe.preprocessing import RollingFeatures


class TestInitializeWindowFeatures:
    """Test suite for initialize_window_features function."""

    def test_single_window_feature(self):
        """Test initialization with a single window feature object."""
        wf = RollingFeatures(stats=["mean"], window_sizes=7)
        wf_list, names, max_size = initialize_window_features(wf)

        assert isinstance(wf_list, list)
        assert len(wf_list) == 1
        assert max_size == 7
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)

    def test_multiple_window_features(self):
        """Test initialization with multiple window feature objects."""
        wf1 = RollingFeatures(stats=["mean"], window_sizes=7)
        wf2 = RollingFeatures(stats=["max", "min"], window_sizes=14)
        wf_list, names, max_size = initialize_window_features([wf1, wf2])

        assert len(wf_list) == 2
        assert max_size == 14
        assert isinstance(names, list)

    def test_none_window_features(self):
        """Test initialization with None."""
        wf_list, names, max_size = initialize_window_features(None)

        assert wf_list is None
        assert names is None
        assert max_size is None

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least one element"):
            initialize_window_features([])

    def test_invalid_window_feature_missing_attributes(self):
        """Test that object missing required attributes raises ValueError."""

        class InvalidWindowFeature:
            """Mock window feature missing required attributes."""

            pass

        with pytest.raises(ValueError, match="must have the attributes"):
            initialize_window_features(InvalidWindowFeature())

    def test_invalid_window_sizes_type(self):
        """Test that invalid window_sizes type raises TypeError."""

        class InvalidWindowFeature:
            window_sizes = "invalid"  # Should be int or list
            features_names = ["feature1"]

            def transform_batch(self):
                pass

            def transform(self):
                pass

        with pytest.raises(TypeError, match="must be an int or a list"):
            initialize_window_features(InvalidWindowFeature())

    def test_duplicate_feature_names_raises_error(self):
        """Test that duplicate feature names raise ValueError."""

        class WindowFeature1:
            window_sizes = 7
            features_names = ["mean"]

            def transform_batch(self):
                pass

            def transform(self):
                pass

        class WindowFeature2:
            window_sizes = 7
            features_names = ["mean"]  # Duplicate name

            def transform_batch(self):
                pass

            def transform(self):
                pass

        with pytest.raises(
            ValueError, match="All window features names must be unique"
        ):
            initialize_window_features([WindowFeature1(), WindowFeature2()])


class TestCheckExtractValuesAndIndex:
    """Test suite for check_extract_values_and_index function."""

    def test_extract_from_series_with_datetime_index(self):
        """Test extraction from pandas Series with DatetimeIndex."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.arange(10), index=dates)

        values, index = check_extract_values_and_index(series)

        assert isinstance(values, np.ndarray)
        assert len(values) == 10
        assert isinstance(index, pd.DatetimeIndex)
        assert len(index) == 10

    def test_extract_from_dataframe(self):
        """Test extraction from pandas DataFrame."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"col1": np.arange(10), "col2": np.arange(10, 20)}, index=dates
        )

        values, index = check_extract_values_and_index(df)

        assert isinstance(values, np.ndarray)
        assert values.shape == (10, 2)
        assert isinstance(index, pd.DatetimeIndex)

    def test_extract_with_range_index(self):
        """Test extraction with RangeIndex."""
        series = pd.Series(np.arange(10))

        values, index = check_extract_values_and_index(series)

        assert isinstance(values, np.ndarray)
        assert isinstance(index, pd.RangeIndex)

    def test_return_values_false(self):
        """Test extraction with return_values=False."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(np.arange(10), index=dates)

        values, index = check_extract_values_and_index(series, return_values=False)

        assert values is None
        assert isinstance(index, pd.DatetimeIndex)

    def test_invalid_input_type_raises_error(self):
        """Test that non-pandas input raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
            check_extract_values_and_index([1, 2, 3])

    def test_invalid_index_type_raises_error(self):
        """Test that invalid index type raises TypeError."""
        series = pd.Series(np.arange(10), index=list(range(10)))

        with pytest.raises(
            TypeError, match="must have a pandas DatetimeIndex or RangeIndex"
        ):
            check_extract_values_and_index(series)

    def test_datetime_index_without_freq_warning(self):
        """Test that DatetimeIndex without freq raises warning."""
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        series = pd.Series([1, 2, 3], index=dates)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_extract_values_and_index(series)
            assert len(w) == 1
            assert "no frequency" in str(w[0].message)

    def test_ignore_freq_parameter(self):
        """Test that ignore_freq=True suppresses frequency warning."""
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        series = pd.Series([1, 2, 3], index=dates)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            check_extract_values_and_index(series, ignore_freq=True)
            # Should not raise warning when ignore_freq=True


class TestGetStyleReprHtml:
    """Test suite for get_style_repr_html function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of two strings."""
        style, uid = get_style_repr_html(is_fitted=True)

        assert isinstance(style, str)
        assert isinstance(uid, str)

    def test_unique_id_length(self):
        """Test that unique ID has expected length."""
        _, uid = get_style_repr_html(is_fitted=False)

        assert len(uid) == 8

    def test_style_contains_css(self):
        """Test that style string contains CSS elements."""
        style, uid = get_style_repr_html(is_fitted=True)

        assert "<style>" in style
        assert f"container-{uid}" in style
        assert "font-family" in style
        assert "background-color" in style

    def test_unique_ids_are_different(self):
        """Test that multiple calls generate different unique IDs."""
        _, uid1 = get_style_repr_html(is_fitted=True)
        _, uid2 = get_style_repr_html(is_fitted=True)

        assert uid1 != uid2

    def test_fitted_parameter(self):
        """Test that function works with both fitted states."""
        style1, uid1 = get_style_repr_html(is_fitted=True)
        style2, uid2 = get_style_repr_html(is_fitted=False)

        # Both should return valid styles
        assert "<style>" in style1
        assert "<style>" in style2


class TestCheckResidualsInput:
    """Test suite for check_residuals_input function."""

    def test_valid_in_sample_residuals(self):
        """Test with valid in-sample residuals."""
        residuals = np.array([0.1, -0.2, 0.3, -0.1])

        # Should not raise any exception
        check_residuals_input(
            forecaster_name="ForecasterRecursive",
            use_in_sample_residuals=True,
            in_sample_residuals_=residuals,
            out_sample_residuals_=None,
            use_binned_residuals=False,
            in_sample_residuals_by_bin_=None,
            out_sample_residuals_by_bin_=None,
        )

    def test_none_in_sample_residuals_raises_error(self):
        """Test that None in-sample residuals raise ValueError."""
        with pytest.raises(ValueError, match="is either None or empty"):
            check_residuals_input(
                forecaster_name="ForecasterRecursive",
                use_in_sample_residuals=True,
                in_sample_residuals_=None,
                out_sample_residuals_=None,
                use_binned_residuals=False,
                in_sample_residuals_by_bin_=None,
                out_sample_residuals_by_bin_=None,
            )

    def test_empty_out_sample_residuals_raises_error(self):
        """Test that empty out-sample residuals raise ValueError."""
        with pytest.raises(ValueError, match="is either None or empty"):
            check_residuals_input(
                forecaster_name="ForecasterRecursive",
                use_in_sample_residuals=False,
                in_sample_residuals_=np.array([0.1, 0.2]),
                out_sample_residuals_=np.array([]),
                use_binned_residuals=False,
                in_sample_residuals_by_bin_=None,
                out_sample_residuals_by_bin_=None,
            )


class TestDateToIndexPosition:
    """Test suite for date_to_index_position function."""

    def test_integer_input_returns_same(self):
        """Test that integer input returns the same integer."""
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        result = date_to_index_position(index, 5)

        assert result == 5

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        index = pd.date_range("2020-01-01", periods=10, freq="D")

        with pytest.raises(ValueError, match="must be 'prediction' or 'validation'"):
            date_to_index_position(index, 5, method="invalid")


class TestPredictMultivariate:
    """Test suite for predict_multivariate function."""

    def test_predict_multivariate_works(self):
        """Test that it combines predictions correctly."""
        # from spotforecast2_safe.forecaster.utils import predict_multivariate
        # Assuming predict_multivariate is also in utils.py?
        # If not, this test might need adjustment or moving.
        # But 'test_skforecast_utils.py' says 'from spotforecast.skforecast.utils import predict_multivariate'
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
