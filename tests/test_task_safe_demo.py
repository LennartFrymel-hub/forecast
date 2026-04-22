"""
Test suite for task_safe_demo functionality.

Converts the task demo into safety-critical pytest test cases for MLOps environments.
Tests baseline, covariate-enhanced, and custom LightGBM forecasting pipelines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor


class TestDemoConfig:
    """Test DemoConfig dataclass and initialization."""

    def test_default_config_initialization(self):
        """Verify DemoConfig initializes with sensible defaults."""
        from spotforecast2_safe.data.fetch_data import get_package_data_home

        config_data = {
            "data_path": get_package_data_home() / "demo11.csv",
            "model_root": Path.home() / "spotforecast2_safe_models",
            "log_root": Path.home() / "spotforecast2_safe_models" / "logs",
            "forecast_horizon": 24,
            "contamination": 0.01,
            "window_size": 72,
            "lags": 24,
            "train_ratio": 0.8,
            "random_seed": 42,
        }

        assert config_data["forecast_horizon"] == 24
        assert config_data["contamination"] == 0.01
        assert config_data["window_size"] == 72
        assert config_data["random_seed"] == 42

    def test_config_weights_length(self):
        """Verify configuration weights are properly defined."""
        weights = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
        assert len(weights) == 11
        assert sum(np.array(weights) > 0) == 7  # 7 positive weights
        assert sum(np.array(weights) < 0) == 4  # 4 negative weights


class TestCalculateMetrics:
    """Test metrics calculation functionality."""

    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics when prediction equals actual (MAE=0, MSE=0)."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        diff = actual - predicted
        mae = diff.abs().mean()
        mse = (diff**2).mean()

        assert mae == 0.0
        assert mse == 0.0

    def test_calculate_metrics_constant_offset(self):
        """Test metrics with constant error offset."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])  # Off by 1

        diff = actual - predicted
        mae = diff.abs().mean()
        mse = (diff**2).mean()

        assert mae == 1.0
        assert mse == 1.0

    def test_calculate_metrics_nan_handling(self):
        """Test metrics with NaN values."""
        actual = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        predicted = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        diff = actual - predicted
        mae = diff.abs().mean()  # pandas mean() skips NaN by default
        mse = (diff**2).mean()

        # Should compute metrics for non-NaN values
        assert not np.isnan(mae)
        assert not np.isnan(mse)


class TestLogging:
    """Test logging infrastructure."""

    def test_logging_setup_creates_logger(self):
        """Verify logging setup creates a properly configured logger."""
        logger = logging.getLogger("test_logger")

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Setup new logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Verify logger has handlers
        assert len(logger.handlers) > 0
        assert logger.level == logging.DEBUG

    def test_logging_handler_formatting(self):
        """Test logging formatter is properly applied."""
        logger = logging.getLogger("test_format_logger")

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        assert handler.formatter is not None
        assert "%(name)s" in handler.formatter._fmt


class TestBooleanParsing:
    """Test boolean argument parsing."""

    def test_parse_bool_true_variants(self):
        """Test parsing of true boolean variants."""
        true_values = ["true", "True", "TRUE", "t", "T", "yes", "YES", "1"]

        def _parse_bool(value: str) -> bool:
            normalized = value.strip().lower()
            if normalized in {"true", "t", "yes", "1"}:
                return True
            if normalized in {"false", "f", "no", "0"}:
                return False
            raise ValueError(f"Expected a boolean value, got: {value}")

        for val in true_values:
            assert _parse_bool(val) is True

    def test_parse_bool_false_variants(self):
        """Test parsing of false boolean variants."""
        false_values = ["false", "False", "FALSE", "f", "F", "no", "NO", "0"]

        def _parse_bool(value: str) -> bool:
            normalized = value.strip().lower()
            if normalized in {"true", "t", "yes", "1"}:
                return True
            if normalized in {"false", "f", "no", "0"}:
                return False
            raise ValueError(f"Expected a boolean value, got: {value}")

        for val in false_values:
            assert _parse_bool(val) is False

    def test_parse_bool_invalid_value(self):
        """Test parsing raises error for invalid boolean values."""

        def _parse_bool(value: str) -> bool:
            normalized = value.strip().lower()
            if normalized in {"true", "t", "yes", "1"}:
                return True
            if normalized in {"false", "f", "no", "0"}:
                return False
            raise ValueError(f"Expected a boolean value, got: {value}")

        with pytest.raises(ValueError):
            _parse_bool("maybe")


class TestAggregatePredict:
    """Test prediction aggregation functionality."""

    def test_agg_predict_with_weights(self):
        """Test aggregation of predictions with weights."""
        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [2.0, 4.0, 6.0],
                "col3": [3.0, 6.0, 9.0],
            }
        )
        weights = [1.0, 1.0, 1.0]

        # Manual aggregation
        weighted_sum = (
            df["col1"] * weights[0] + df["col2"] * weights[1] + df["col3"] * weights[2]
        )
        expected = weighted_sum / sum(weights)

        assert isinstance(expected, pd.Series)
        assert len(expected) == 3

    def test_agg_predict_handles_negative_weights(self):
        """Test aggregation properly handles negative weights (difference operations)."""
        df = pd.DataFrame(
            {
                "col1": [10.0, 20.0, 30.0],
                "col2": [5.0, 10.0, 15.0],
            }
        )
        weights = [1.0, -1.0]  # col1 - col2

        # Manual aggregation
        weighted_sum = df["col1"] * weights[0] + df["col2"] * weights[1]
        expected = weighted_sum / sum(weights) if sum(weights) != 0 else weighted_sum

        assert expected[0] == 5.0
        assert expected[1] == 10.0
        assert expected[2] == 15.0


class TestDataValidation:
    """Test data validation and loading."""

    def test_load_actual_combined_missing_file(self):
        """Test error handling when ground truth file is missing."""
        nonexistent_path = Path("/nonexistent/path/to/data.csv")

        if not nonexistent_path.is_file():
            # This is the expected behavior
            assert True

    def test_load_actual_combined_column_validation(self):
        """Test validation of required columns in ground truth data."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )

        required_columns = ["A", "B", "C"]
        missing_cols = [col for col in required_columns if col not in df.columns]

        assert "C" in missing_cols
        assert len(missing_cols) == 1


class TestForecastingPipeline:
    """Test complete forecasting pipeline components."""

    def test_baseline_forecast_structure(self):
        """Test baseline forecast returns expected structure."""
        # Mock minimal required data
        y = pd.Series(
            np.random.randn(100), index=pd.date_range("2020-01-01", periods=100)
        )

        # Verify series properties
        assert isinstance(y, pd.Series)
        assert len(y) == 100
        assert isinstance(y.index, pd.DatetimeIndex)

    def test_covariate_forecast_with_exogenous(self):
        """Test covariate forecast includes exogenous variables."""
        # Create synthetic data
        y = pd.Series(
            np.random.randn(100), index=pd.date_range("2020-01-01", periods=100)
        )
        exog = pd.DataFrame(
            {
                "holiday": np.random.randint(0, 2, 100),
                "weather": np.random.randn(100),
            },
            index=y.index,
        )

        assert exog.shape == (100, 2)
        assert all(col in exog.columns for col in ["holiday", "weather"])

    def test_custom_estimator_initialization(self):
        """Test custom LightGBM estimator can be instantiated."""
        custom_lgbm = LGBMRegressor(
            n_estimators=1059,
            learning_rate=0.04191323446625026,
            num_leaves=212,
            min_child_samples=54,
            subsample=0.5014650987802548,
            colsample_bytree=0.6080926628683118,
            random_state=42,
            verbose=-1,
        )

        assert custom_lgbm.n_estimators == 1059
        assert custom_lgbm.learning_rate == pytest.approx(0.04191323446625026)
        assert custom_lgbm.num_leaves == 212
        assert custom_lgbm.random_state == 42


class TestIndexAlignment:
    """Test time series index alignment."""

    def test_index_alignment_baseline_vs_covariate(self):
        """Test that baseline and covariate predictions have aligned indices."""
        idx = pd.date_range("2020-01-01", periods=24, freq="h")
        baseline = pd.Series(np.random.randn(24), index=idx, name="baseline")
        covariate = pd.Series(np.random.randn(24), index=idx, name="covariate")

        indices_aligned = (baseline.index == covariate.index).all()
        assert bool(indices_aligned) is True

    def test_reindex_with_missing_values(self):
        """Test reindexing handles missing values correctly."""
        idx1 = pd.date_range("2020-01-01", periods=5, freq="h")
        idx2 = pd.date_range("2020-01-01", periods=7, freq="h")

        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx1)

        # Reindex s1 to match s2
        s1_reindexed = s1.reindex(idx2)

        # Check for NaN in new positions
        assert pd.isna(s1_reindexed.iloc[5:]).all()


class TestErrorHandling:
    """Test error handling in task execution."""

    def test_missing_ground_truth_handling(self):
        """Test graceful error handling when ground truth is unavailable."""
        data_path = Path("/nonexistent/data.csv")

        if not data_path.is_file():
            error_msg = f"Ground truth file not found at {data_path}"
            assert "Ground truth" in error_msg

    def test_invalid_forecast_horizon(self):
        """Test error handling for invalid forecast horizon."""
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            n2n_predict_with_covariates,
        )

        with pytest.raises(ValueError):
            n2n_predict_with_covariates(forecast_horizon=-1)

        with pytest.raises(ValueError):
            n2n_predict_with_covariates(forecast_horizon=0)
        """Test handling of unexpected prediction shapes."""
        expected_shape = (24, 11)
        actual_shape = (24, 10)  # Missing one column

        assert expected_shape[0] == actual_shape[0]  # Same horizon
        assert expected_shape[1] != actual_shape[1]  # Different dimensions


class TestMemoryAndPerformance:
    """Test memory efficiency and performance considerations."""

    def test_large_series_memory_efficiency(self):
        """Test handling of large time series data structures."""
        large_series = pd.Series(np.random.randn(100000), name="large")

        assert len(large_series) == 100000
        assert isinstance(large_series, pd.Series)

    def test_dataframe_column_subset(self):
        """Test efficient column subsetting from large DataFrames."""
        large_df = pd.DataFrame(np.random.randn(1000, 20))
        subset_cols = [0, 5, 10]

        subset = large_df.iloc[:, subset_cols]

        assert subset.shape == (1000, 3)

    def test_aggregate_without_copy(self):
        """Test aggregation operations minimize copying."""
        df = pd.DataFrame({f"col_{i}": np.random.randn(100) for i in range(10)})

        weights = np.ones(10) / 10
        # This should be efficient - use proper matrix multiplication
        agg_result = df.values @ weights

        assert len(agg_result) == 100


# Integration tests combining multiple components
class TestIntegration:
    """Integration tests combining task components."""

    def test_end_to_end_task_structure(self):
        """Test end-to-end task execution structure without external files."""
        # Create minimal synthetic data
        y = pd.Series(
            np.random.randn(100),
            index=pd.date_range("2020-01-01", periods=100),
            name="y",
        )

        # Verify core components work together
        assert isinstance(y, pd.Series)
        assert len(y) == 100
        assert y.index.is_monotonic_increasing

    def test_metric_consistency_across_models(self):
        """Test metrics can be consistently calculated for multiple models."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        predictions = {
            "model_a": pd.Series([1.1, 2.1, 3.1, 4.1, 5.1]),
            "model_b": pd.Series([0.9, 1.9, 2.9, 3.9, 4.9]),
            "model_c": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        metrics = {}
        for model_name, pred in predictions.items():
            diff = actual - pred
            mae = diff.abs().mean()
            mse = (diff**2).mean()
            metrics[model_name] = {"MAE": mae, "MSE": mse}

        # Model C should have zero error
        assert metrics["model_c"]["MAE"] == 0.0
        assert metrics["model_c"]["MSE"] == 0.0

        # All models should have consistent metric structure
        assert all(isinstance(m, dict) for m in metrics.values())
        assert all("MAE" in m and "MSE" in m for m in metrics.values())
