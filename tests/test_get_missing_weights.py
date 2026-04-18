"""
Comprehensive pytest tests for get_missing_weights and custom_weights functions.

Tests the missing weights calculation and weight function creation used in
recursive forecasting with sample weighting to penalize observations near gaps.
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.imputation import (
    custom_weights,
    get_missing_weights,
)


def create_test_data_with_gaps(dates, gap_indices):
    """Helper to create test data with gaps at specified indices."""
    data = pd.DataFrame(
        {
            "value": np.random.randn(len(dates)),
            "other": np.random.randn(len(dates)),
        },
        index=dates,
    )
    data.iloc[gap_indices] = np.nan
    return data


class TestGetMissingWeights:
    """Test suite for get_missing_weights function from spotforecast2_safe."""

    def test_basic_functionality_no_missing(self):
        """Test basic functionality with no missing data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {"value": np.arange(100), "other": np.arange(100) * 2},
            index=dates,
        )

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=72, verbose=False
        )

        # Data should be unchanged (no NaNs to fill)
        pd.testing.assert_frame_equal(imputed_data, data)

        # missing_mask (now weights) should be all 1.0 (no gaps)
        assert isinstance(missing_mask, pd.Series)
        assert (missing_mask == 1.0).all()

    def test_single_gap_creates_missing_mask(self):
        """Test that imputation is applied (ffill/bfill) and no NaN remain."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = create_test_data_with_gaps(dates, [50])

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=10, verbose=False
        )

        # After ffill/bfill, there should be no NaN values
        # weights series should be float and have some 0.0 values for the gap
        assert not imputed_data.isnull().any().any()
        assert np.issubdtype(missing_mask.dtype, np.floating)
        assert (missing_mask == 0.0).any()

    def test_data_imputation_ffill_bfill(self):
        """Test that data is properly imputed with ffill and bfill."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "value": [1, 2, np.nan, 4, 5, np.nan, np.nan, 8, 9, 10] + [np.nan] * 10,
                "other": list(range(10)) + [np.nan] * 10,
            },
            index=dates,
        )

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=5, verbose=False
        )

        # After ffill and bfill, there should be no NaN values in early/middle rows
        # (bfill fills from the end backward)
        # Check that imputation was applied
        assert not imputed_data.isnull().any().any()

    def test_window_size_validation(self):
        """Test that window_size validation works."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"value": np.arange(10)}, index=dates)

        # Window size >= number of rows should raise error
        with pytest.raises(ValueError, match="window_size must be smaller"):
            get_missing_weights(data, window_size=10, verbose=False)

    def test_empty_data_validation(self):
        """Test that empty data raises error."""
        data = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            get_missing_weights(data, window_size=5, verbose=False)

    def test_window_size_zero_validation(self):
        """Test that zero window_size raises error."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"value": np.arange(50)}, index=dates)

        with pytest.raises(ValueError, match="window_size must be a positive integer"):
            get_missing_weights(data, window_size=0, verbose=False)

    def test_multiple_gaps(self):
        """Test handling of multiple gaps in data."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = create_test_data_with_gaps(dates, [50, 100, 150])

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=24, verbose=False
        )

        # After imputation, all data should be filled (no NaNs)
        assert not imputed_data.isnull().any().any()
        # missing_mask (weights) should have 0.0 values at gap locations
        assert (missing_mask == 0.0).any()
        # Verify specific gap locations (e.g., index 50)
        assert missing_mask.iloc[50] == 0.0

    def test_verbose_output(self, capsys):
        """Test verbose output messages."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = create_test_data_with_gaps(dates, [10, 20])

        _, _ = get_missing_weights(data, window_size=72, verbose=True)

        captured = capsys.readouterr()
        assert "Number of rows with missing values" in captured.out
        assert "Percentage of rows with missing values" in captured.out

    def test_return_types(self):
        """Test that return types are correct."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"value": np.arange(50)}, index=dates)

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=24, verbose=False
        )

        # Check types
        assert isinstance(imputed_data, pd.DataFrame)
        assert isinstance(missing_mask, pd.Series)

    def test_missing_mask_numeric(self):
        """Test that missing_mask (weights) is numeric."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = create_test_data_with_gaps(dates, [50])

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=24, verbose=False
        )

        assert np.issubdtype(missing_mask.dtype, np.floating)

    def test_edge_case_single_row_fails(self):
        """Test that single row with window_size validation fails appropriately."""
        dates = pd.date_range("2020-01-01", periods=1, freq="D")
        data = pd.DataFrame({"value": [1.0]}, index=dates)

        # window_size must be smaller than number of rows
        with pytest.raises(ValueError):
            get_missing_weights(data, window_size=1, verbose=False)

    def test_edge_case_minimal_valid(self):
        """Test minimal valid case."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"value": np.arange(10)}, index=dates)

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=5, verbose=False
        )

        assert len(imputed_data) == len(data)
        assert len(missing_mask) == len(data)

    def test_data_shape_preservation(self):
        """Test that imputed data preserves shape."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = create_test_data_with_gaps(dates, [25, 50, 75])

        original_shape = data.shape
        imputed_data, missing_mask = get_missing_weights(
            data, window_size=20, verbose=False
        )

        assert imputed_data.shape == original_shape

    def test_index_preservation(self):
        """Test that index is preserved after imputation."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = create_test_data_with_gaps(dates, [50])

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=24, verbose=False
        )

        pd.testing.assert_index_equal(imputed_data.index, data.index)


class TestCustomWeights:
    """Test suite for custom_weights function."""

    def test_custom_weights_with_valid_index(self):
        """Test custom_weights with valid scalar index."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        weights_series = pd.Series(np.ones(50) * 0.5, index=dates)

        weight = custom_weights(dates[10], weights_series)

        assert isinstance(weight, (float, np.floating))
        assert weight == 0.5

    def test_custom_weights_with_pd_index(self):
        """Test custom_weights with pd.Index."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        weights_series = pd.Series(np.ones(50) * 0.5, index=dates)

        weights = custom_weights(dates[10:20], weights_series)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == 10
        assert np.allclose(weights, 0.5)

    def test_custom_weights_invalid_scalar_index(self):
        """Test custom_weights raises error for invalid scalar index."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        weights_series = pd.Series(np.ones(50) * 0.5, index=dates)

        invalid_date = pd.Timestamp("2030-01-01")

        with pytest.raises(ValueError, match="Index not found in weights_series"):
            custom_weights(invalid_date, weights_series)

    def test_custom_weights_invalid_pd_index(self):
        """Test custom_weights raises error for invalid pd.Index."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        weights_series = pd.Series(np.ones(50) * 0.5, index=dates)

        invalid_dates = pd.date_range("2030-01-01", periods=5, freq="D")

        with pytest.raises(ValueError, match="Index not found in weights_series"):
            custom_weights(invalid_dates, weights_series)

    def test_custom_weights_mixed_index(self):
        """Test custom_weights with partially valid pd.Index raises error."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        weights_series = pd.Series(np.ones(50) * 0.5, index=dates)

        # Mix valid and invalid indices
        mixed_index = pd.DatetimeIndex(
            [dates[5], dates[10], pd.Timestamp("2030-01-01")]
        )

        with pytest.raises(ValueError, match="Index not found in weights_series"):
            custom_weights(mixed_index, weights_series)


class TestIntegrationWithForecaster:
    """Integration tests with ForecasterRecursive pattern."""

    def test_weight_func_with_missing_weights_typical_use(self):
        """Test typical usage pattern with get_missing_weights and custom_weights."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = create_test_data_with_gaps(dates, [50, 100, 150])

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=24, verbose=False
        )

        # get_missing_weights now returns weights directly
        weights_series = missing_mask

        # Simulate what ForecasterRecursive does: call with window_size:] index
        window_size = 24
        train_index = imputed_data.index[window_size:]

        # Should be able to call custom_weights with train_index
        sample_weights = custom_weights(train_index, weights_series)

        assert len(sample_weights) == len(train_index)
        assert isinstance(sample_weights, np.ndarray)
        # Weights should be 0 or 1
        assert np.all((sample_weights == 0) | (sample_weights == 1))

    def test_weight_func_preserves_value_distribution(self):
        """Test that weight function preserves original weight values."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"value": np.arange(100)}, index=dates)

        imputed_data, missing_mask = get_missing_weights(
            data, window_size=20, verbose=False
        )

        weights_series = missing_mask

        # Test full index
        all_weights = custom_weights(dates, weights_series)
        assert len(all_weights) == len(dates)
        assert np.all((all_weights == 0) | (all_weights == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
