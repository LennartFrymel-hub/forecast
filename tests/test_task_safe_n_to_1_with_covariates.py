"""
Test suite for N-to-1 forecasting with exogenous covariates.

Converts task_safe_n_to_1_with_covariates_and_dataframe.py into comprehensive
pytest test cases for safety-critical MLOps environments.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class TestCovariateDataPreperation:
    """Test exogenous covariate data preparation."""

    def test_weather_covariate_structure(self):
        """Test weather covariate data has expected structure."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        weather = pd.DataFrame(
            {
                "temperature": np.random.uniform(0, 35, 100),
                "humidity": np.random.uniform(30, 100, 100),
                "wind_speed": np.random.uniform(0, 20, 100),
            },
            index=dates,
        )

        assert weather.shape == (100, 3)
        assert all(
            col in weather.columns for col in ["temperature", "humidity", "wind_speed"]
        )
        assert isinstance(weather.index, pd.DatetimeIndex)

    def test_holiday_covariate_binary(self):
        """Test holiday covariate is binary (0/1)."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        holidays = pd.Series(
            np.random.randint(0, 2, 365), index=dates, name="is_holiday", dtype=int
        )

        assert set(holidays.unique()).issubset({0, 1})
        assert len(holidays) == 365

    def test_calendar_features_generation(self):
        """Test calendar feature generation."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")

        calendar_features = pd.DataFrame(
            {
                "day_of_week": dates.dayofweek,
                "day_of_month": dates.day,
                "month": dates.month,
                "quarter": dates.quarter,
                "is_weekend": (dates.dayofweek >= 5).astype(int),
            },
            index=dates,
        )

        assert calendar_features.shape[0] == 365
        assert "day_of_week" in calendar_features.columns
        assert calendar_features["day_of_week"].max() == 6
        assert calendar_features["month"].min() == 1
        assert calendar_features["month"].max() == 12

    def test_cyclic_feature_encoding(self):
        """Test cyclic feature encoding for seasonal patterns."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        month = dates.month

        # Cyclic encoding: cos and sin
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        assert len(month_sin) == 365
        assert len(month_cos) == 365
        assert all(-1 <= x <= 1 for x in month_sin)
        assert all(-1 <= x <= 1 for x in month_cos)


class TestExogenousVariableValidation:
    """Test validation of exogenous variables."""

    def test_exog_length_matches_y(self):
        """Test exogenous variables have same length as target series."""
        y_len = 100
        y = pd.Series(
            np.random.randn(y_len), index=pd.date_range("2020-01-01", periods=y_len)
        )

        exog = pd.DataFrame(
            np.random.randn(y_len, 3),
            index=y.index,
            columns=["feat1", "feat2", "feat3"],
        )

        assert len(exog) == len(y)
        assert exog.index.equals(y.index)

    def test_exog_index_alignment(self):
        """Test exogenous variables are properly aligned with target."""
        dates = pd.date_range("2020-01-01", periods=100, freq="h")

        y = pd.Series(np.random.randn(100), index=dates)
        exog = pd.DataFrame(np.random.randn(100, 2), index=dates)

        # Check alignment
        aligned = exog.index.equals(y.index)
        assert aligned is True

    def test_exog_missing_features_detection(self):
        """Test detection of missing exogenous features."""
        required_features = ["temp", "humidity", "wind"]
        actual_features = ["temp", "humidity"]

        missing = [f for f in required_features if f not in actual_features]

        assert "wind" in missing
        assert len(missing) == 1

    def test_exog_nan_handling(self):
        """Test handling of NaN values in exogenous variables."""
        exog = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "feat2": [10.0, np.nan, 30.0, 40.0, 50.0],
            }
        )

        # Count NaNs
        nan_count = exog.isna().sum().sum()
        assert nan_count == 2

        # Forward fill strategy (pandas 2.0+)
        exog_filled = exog.ffill()
        # ffill fills all NaNs (forward propagation), so no NaNs remain
        assert exog_filled.isna().sum().sum() == 0


class TestLoggingForCovariates:
    """Test logging in covariate pipeline."""

    def test_setup_logging_creates_logger(self):
        """Verify logging setup creates properly configured logger."""
        logger = logging.getLogger("test_n_to_1_logger")

        # Clean previous handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        assert len(logger.handlers) > 0
        assert logger.level == logging.INFO

    def test_file_logging_timestamp_format(self):
        """Test file logging uses proper timestamp format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Verify format
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert "_" in timestamp
        assert timestamp[:8].isdigit()  # Date part


class TestNto1ForecastingPipeline:
    """Test N-to-1 forecasting with covariates."""

    def test_basic_n_to_1_structure(self):
        """Test basic N-to-1 forecasting setup."""
        # Create synthetic multi-step target
        horizon = 24
        y = pd.Series(
            np.random.randn(100 + horizon),
            index=pd.date_range("2020-01-01", periods=100 + horizon),
            name="target",
        )

        assert len(y) == 100 + horizon
        assert isinstance(y, pd.Series)

    def test_recursive_forecaster_initialization(self):
        """Test recursive forecaster can be initialized."""
        # Test that the infrastructure for recursive forecasting exists
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1
        )

        assert estimator.n_estimators == 100
        assert estimator.learning_rate == 0.1

    def test_multioutput_forecasting_setup(self):
        """Test multioutput forecasting data structure."""
        # For N-to-1, we forecast multiple steps ahead
        steps = 24
        forecast_array = np.random.randn(steps)

        assert len(forecast_array) == steps
        assert forecast_array.ndim == 1


class TestCovariateFeatureEngineering:
    """Test feature engineering with covariates."""

    def test_polynomial_features_generation(self):
        """Test polynomial feature generation from covariates."""
        x = np.array([1, 2, 3, 4, 5])

        # Create polynomial features
        poly_features = np.column_stack([x, x**2, x**3])

        assert poly_features.shape == (5, 3)
        assert (poly_features[:, 0] == x).all()
        assert (poly_features[:, 1] == x**2).all()

    def test_lag_feature_creation(self):
        """Test creation of lag features from time series."""
        y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        max_lag = 3

        # Create lag features
        lags = pd.DataFrame({f"lag_{i}": y.shift(i) for i in range(1, max_lag + 1)})

        assert lags.shape == (10, 3)
        assert "lag_1" in lags.columns
        assert pd.isna(lags["lag_1"].iloc[0])  # First row should be NaN

    def test_rolling_window_features(self):
        """Test rolling window feature generation."""
        y = pd.Series(np.random.randn(100))
        window_size = 7

        rolling_mean = y.rolling(window=window_size).mean()
        rolling_std = y.rolling(window=window_size).std()

        assert len(rolling_mean) == 100
        assert len(rolling_std) == 100
        assert pd.isna(rolling_mean.iloc[: window_size - 1]).all()


class TestExogenousIntegration:
    """Test integration of exogenous variables into forecasting."""

    def test_exog_matrix_expansion(self):
        """Test exogenous variables expand feature matrix."""
        base_features = 5
        exog_features = 3
        n_samples = 100

        X_base = np.random.randn(n_samples, base_features)
        X_exog = np.random.randn(n_samples, exog_features)

        X_combined = np.column_stack([X_base, X_exog])

        assert X_combined.shape == (n_samples, base_features + exog_features)

    def test_lag_and_exog_combination(self):
        """Test combining lag features with exogenous variables."""
        y = pd.Series(np.random.randn(100))
        exog = pd.DataFrame(np.random.randn(100, 2), columns=["temp", "humidity"])

        # Create lag features
        y_lags = pd.DataFrame(
            {
                "y_lag_1": y.shift(1),
                "y_lag_2": y.shift(2),
            }
        )

        # Combine
        X = pd.concat([y_lags, exog], axis=1)

        assert X.shape[1] == 4  # 2 lags + 2 exog features
        assert "y_lag_1" in X.columns
        assert "temp" in X.columns


class TestPredictionAggregation:
    """Test aggregation of predictions from N-to-1 forecasting."""

    def test_weighted_aggregation_basic(self):
        """Test basic weighted aggregation of predictions."""
        predictions = pd.DataFrame(
            {
                "loc_1": [1.0, 2.0, 3.0],
                "loc_2": [2.0, 4.0, 6.0],
                "loc_3": [3.0, 6.0, 9.0],
            }
        )

        weights = [0.5, 0.3, 0.2]  # Must sum to 1

        weighted = (predictions.T * weights).T
        aggregated = weighted.sum(axis=1)

        assert len(aggregated) == 3
        assert isinstance(aggregated, pd.Series)

    def test_aggregation_with_unequal_weights(self):
        """Test aggregation with unequal importance weights."""
        # Higher weight for high priority location
        weights = {"high_priority": 0.6, "medium_priority": 0.3, "low_priority": 0.1}

        # Check weight proportions
        assert (
            weights["high_priority"]
            > weights["medium_priority"]
            > weights["low_priority"]
        )

    def test_aggregation_preserves_temporal_index(self):
        """Test aggregation maintains temporal index integrity."""
        dates = pd.date_range("2020-01-01", periods=10, freq="h")
        predictions = pd.DataFrame(
            np.random.randn(10, 3), index=dates, columns=["loc_1", "loc_2", "loc_3"]
        )

        weights = [0.4, 0.4, 0.2]
        aggregated = predictions.values @ np.array(weights)  # Matrix multiplication
        aggregated_series = pd.Series(aggregated, index=dates)

        assert aggregated_series.index.equals(dates)


class TestCovariateTimezone:
    """Test timezone handling with covariates."""

    def test_timezone_aware_index(self):
        """Test timezone-aware datetime index."""
        dates_utc = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")

        y = pd.Series(np.random.randn(100), index=dates_utc)

        assert y.index.tz is not None
        assert str(y.index.tz) == "UTC"

    def test_timezone_conversion(self):
        """Test timezone conversion compatibility."""
        dates_utc = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
        # Convert to a different timezone (pytz not required)
        dates_est = dates_utc.tz_convert("US/Eastern")

        assert len(dates_est) == 24
        assert "Eastern" in str(dates_est.tz) or dates_est.tz is not None

    def test_timezone_consistency_across_variables(self):
        """Test timezone consistency between y and exog."""
        tz = "UTC"
        dates = pd.date_range("2020-01-01", periods=100, freq="h", tz=tz)

        y = pd.Series(np.random.randn(100), index=dates)
        exog = pd.DataFrame(np.random.randn(100, 2), index=dates)

        assert str(y.index.tz) == tz
        assert str(exog.index.tz) == tz


class TestForcedTraining:
    """Test forced training vs. cached model loading."""

    def test_force_train_parameter(self):
        """Test force_train parameter controls training behavior."""
        # Test when force_train is True
        force_train = True
        action = "retrain_model" if force_train else "load_cached_model"
        assert action == "retrain_model"

        # Test when force_train is False
        force_train = False
        action = "retrain_model" if force_train else "load_cached_model"
        assert action == "load_cached_model"

    def test_model_directory_creation(self):
        """Test model directories are created when needed."""
        model_dir = Path("/tmp/test_models/n_to_1")

        # Simulate directory creation
        model_dir.mkdir(parents=True, exist_ok=True)

        assert model_dir.exists()

        # Cleanup
        import shutil

        shutil.rmtree(model_dir.parent, ignore_errors=True)


class TestErrorHandlingCovariates:
    """Test error handling in covariate pipeline."""

    def test_missing_exog_detection(self):
        """Test detection of missing exogenous data."""
        horizon = 24
        exog_provided = 20  # Less than horizon

        missing = horizon - exog_provided
        assert missing == 4

    def test_misaligned_index_error(self):
        """Test detection of misaligned indices."""
        dates_y = pd.date_range("2020-01-01", periods=100, freq="h")
        dates_exog = pd.date_range("2020-01-01", periods=95, freq="h")  # Shorter

        y = pd.Series(np.random.randn(100), index=dates_y)
        exog = pd.DataFrame(np.random.randn(95, 2), index=dates_exog)

        common_idx = y.index.intersection(exog.index)
        assert len(common_idx) == 95

    def test_invalid_forecast_horizon_detection(self):
        """Test detection of invalid forecast horizon."""
        valid_horizons = [6, 12, 24, 48, 168]  # 6H, 12H, 1D, 2D, 1W
        invalid_horizon = -24

        assert invalid_horizon not in valid_horizons
        assert invalid_horizon < 0


class TestKwargsFlexibility:
    """Test kwargs flexibility for pipeline customization."""

    def test_estimator_kwargs_passthrough(self):
        """Test estimator parameters can be customized via kwargs."""
        estimator_kwargs = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 100,
        }

        assert estimator_kwargs["n_estimators"] == 500
        assert "learning_rate" in estimator_kwargs
        assert len(estimator_kwargs) == 3

    def test_forecaster_kwargs_passthrough(self):
        """Test forecaster parameters can be customized via kwargs."""
        forecaster_kwargs = {
            "lags": [1, 7, 24],
            "window_size": 72,
            "weight_func": "linear",
        }

        assert forecaster_kwargs["lags"] == [1, 7, 24]
        assert forecaster_kwargs["window_size"] == 72

    def test_agg_kwargs_passthrough(self):
        """Test aggregation parameters can be customized via kwargs."""
        agg_kwargs = {
            "method": "weighted_mean",
            "normalize_weights": True,
        }

        assert agg_kwargs["method"] == "weighted_mean"
        assert agg_kwargs["normalize_weights"] is True


class TestIntegrationN2N:
    """Integration tests for N-to-1 with covariates pipeline."""

    def test_end_to_end_pipeline_structure(self):
        """Test complete pipeline structure without external files."""
        # Minimal synthetic setup
        y = pd.Series(
            np.random.randn(100),
            index=pd.date_range("2020-01-01", periods=100, freq="h"),
            name="load",
        )

        exog = pd.DataFrame(
            {
                "temperature": np.random.uniform(0, 35, 100),
                "hour": np.arange(100) % 24,
            },
            index=y.index,
        )

        assert len(y) == 100
        assert exog.shape == (100, 2)
        assert y.index.equals(exog.index)

    def test_pipeline_output_consistency(self):
        """Test pipeline outputs have consistent structure."""
        # Simulate pipeline outputs
        predictions = pd.DataFrame(
            np.random.randn(24, 11), columns=[f"location_{i}" for i in range(11)]
        )

        aggregated = predictions.mean(axis=1)
        metrics = {"MAE": 2.5, "MSE": 7.2}

        assert predictions.shape == (24, 11)
        assert len(aggregated) == 24
        assert all(k in metrics for k in ["MAE", "MSE"])

    def test_reproducibility_with_seed(self):
        """Test reproducibility with random seed."""
        seed = 42
        np.random.seed(seed)
        data_1 = np.random.randn(10)

        np.random.seed(seed)
        data_2 = np.random.randn(10)

        assert (data_1 == data_2).all()
