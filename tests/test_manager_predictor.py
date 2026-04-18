# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe.manager.predictor.

Covers:
- get_model_prediction: model lookup, missing model, missing method, exception
- build_prediction_package: return structure, no-exog, with-exog, with df_test,
  missing target in df_test, empty test overlap, package import, docstring examples
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from spotforecast2_safe.manager.predictor import (
    build_prediction_package,
    get_model_prediction,
)

# =============================================================================
# Existing tests for get_model_prediction
# =============================================================================


class TestPredictor(unittest.TestCase):
    """Tests for the predictor manager."""

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_success(self, mock_get_last):
        """Test successful prediction retrieval."""
        mock_model = MagicMock()
        mock_model.package_prediction.return_value = {"key": "value"}
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertEqual(result, {"key": "value"})
        mock_get_last.assert_called_once_with("lgbm", None)
        mock_model.package_prediction.assert_called_once()

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_no_model(self, mock_get_last):
        """Test handling when no model is found."""
        mock_get_last.return_value = (-1, None)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_missing_method(self, mock_get_last):
        """Test handling when model lacks package_prediction method."""
        mock_model = MagicMock(spec=[])  # No methods
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_exception(self, mock_get_last):
        """Test handling when package_prediction raises an exception."""
        mock_model = MagicMock()
        mock_model.package_prediction.side_effect = Exception("Prediction failed")
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)


# =============================================================================
# Shared fixtures for build_prediction_package tests
# =============================================================================

N_TRAIN = 100
N_LAGS = 24
N_FUTURE = 24


@pytest.fixture
def train_idx():
    return pd.date_range("2024-01-01", periods=N_TRAIN, freq="h", tz="UTC")


@pytest.fixture
def y_train(train_idx):
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(50.0, 5.0, N_TRAIN), index=train_idx, name="load")


@pytest.fixture
def future_idx(train_idx):
    return pd.date_range(
        train_idx[-1] + pd.Timedelta(hours=1), periods=N_FUTURE, freq="h", tz="UTC"
    )


@pytest.fixture
def mock_forecaster(train_idx, future_idx):
    """Mock ForecasterRecursive returning deterministic synthetic data."""
    internal_idx = train_idx[N_LAGS:]  # first N_LAGS points consumed by lags
    n_internal = len(internal_idx)

    forecaster = MagicMock()
    X_matrix = np.ones((n_internal, N_LAGS))
    y_internal = pd.Series(np.full(n_internal, 50.0), index=internal_idx)

    forecaster.create_train_X_y.return_value = (X_matrix, y_internal)
    forecaster.estimator.predict.return_value = np.full(n_internal, 49.0)
    forecaster.predict.return_value = pd.Series(
        np.full(N_FUTURE, 48.0), index=future_idx, name="pred"
    )
    return forecaster


@pytest.fixture
def df_test_matching(future_idx):
    """Test DataFrame whose DateTime column aligns exactly with future_idx."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "DateTime": future_idx.tz_convert(None),  # strip tz → gets re-localized
            "load": rng.normal(50.0, 5.0, N_FUTURE),
        }
    )


@pytest.fixture
def df_test_no_match(future_idx):
    """Test DataFrame that does NOT contain the 'load' column."""
    return pd.DataFrame(
        {
            "DateTime": future_idx.tz_convert(None),
            "other": np.ones(N_FUTURE),
        }
    )


@pytest.fixture
def df_test_non_overlapping():
    """Test DataFrame whose DateTime values are far in the past (no index overlap)."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=N_FUTURE, freq="h")
    return pd.DataFrame({"DateTime": idx, "load": rng.normal(50.0, 5.0, N_FUTURE)})


# =============================================================================
# Tests: build_prediction_package — no exogenous features
# =============================================================================


class TestBuildPredictionPackageNoExog:
    """Return-structure tests when no exogenous features are passed."""

    def test_returns_dict(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg, dict)

    def test_all_required_keys_present(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        required = {
            "train_actual",
            "train_pred",
            "future_actual",
            "future_pred",
            "metrics_train",
            "metrics_future",
            "metrics_future_one_day",
            "validation_passed",
        }
        assert required.issubset(pkg.keys())

    def test_train_actual_is_series(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg["train_actual"], pd.Series)

    def test_train_pred_is_series(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg["train_pred"], pd.Series)

    def test_future_pred_is_series(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg["future_pred"], pd.Series)

    def test_future_pred_length(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert len(pkg["future_pred"]) == N_FUTURE

    def test_future_actual_is_empty_float64_series(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        fa = pkg["future_actual"]
        assert isinstance(fa, pd.Series)
        assert len(fa) == 0
        assert fa.dtype == "float64"

    def test_validation_passed_is_true(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert pkg["validation_passed"] is True

    def test_metrics_train_has_mae_and_mape(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert "mae" in pkg["metrics_train"]
        assert "mape" in pkg["metrics_train"]

    def test_metrics_train_mae_is_float(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg["metrics_train"]["mae"], float)

    def test_metrics_train_mape_is_float(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert isinstance(pkg["metrics_train"]["mape"], float)

    def test_metrics_future_is_empty_dict(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert pkg["metrics_future"] == {}

    def test_metrics_future_one_day_is_empty_dict(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert pkg["metrics_future_one_day"] == {}

    def test_test_actual_key_absent(self, mock_forecaster, y_train):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert "test_actual" not in pkg

    def test_create_train_X_y_called_without_exog(self, mock_forecaster, y_train):
        build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        mock_forecaster.create_train_X_y.assert_called_once_with(y=y_train, exog=None)

    def test_predict_called_with_correct_steps(self, mock_forecaster, y_train):
        build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        mock_forecaster.predict.assert_called_once_with(steps=N_FUTURE, exog=None)

    def test_train_actual_index_matches_train_pred_index(
        self, mock_forecaster, y_train
    ):
        pkg = build_prediction_package(mock_forecaster, "load", y_train, N_FUTURE)
        assert pkg["train_actual"].index.equals(pkg["train_pred"].index)


# =============================================================================
# Tests: build_prediction_package — with exogenous features
# =============================================================================


class TestBuildPredictionPackageWithExog:
    """Tests when exog_train and exog_future are supplied."""

    @pytest.fixture
    def exog_train(self, train_idx):
        return pd.DataFrame(
            {"feat1": np.ones(N_TRAIN), "feat2": np.zeros(N_TRAIN)},
            index=train_idx,
        )

    @pytest.fixture
    def exog_future(self, future_idx):
        return pd.DataFrame(
            {"feat1": np.ones(N_FUTURE), "feat2": np.zeros(N_FUTURE)},
            index=future_idx,
        )

    def test_create_train_X_y_called_with_exog(
        self, mock_forecaster, y_train, exog_train, exog_future
    ):
        build_prediction_package(
            mock_forecaster,
            "load",
            y_train,
            N_FUTURE,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        mock_forecaster.create_train_X_y.assert_called_once_with(
            y=y_train, exog=exog_train
        )

    def test_predict_called_with_exog_future(
        self, mock_forecaster, y_train, exog_train, exog_future
    ):
        build_prediction_package(
            mock_forecaster,
            "load",
            y_train,
            N_FUTURE,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        mock_forecaster.predict.assert_called_once_with(
            steps=N_FUTURE, exog=exog_future
        )

    def test_result_still_has_required_keys(
        self, mock_forecaster, y_train, exog_train, exog_future
    ):
        pkg = build_prediction_package(
            mock_forecaster,
            "load",
            y_train,
            N_FUTURE,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        assert "train_actual" in pkg
        assert "future_pred" in pkg
        assert "validation_passed" in pkg


# =============================================================================
# Tests: build_prediction_package — df_test with matching data
# =============================================================================


class TestBuildPredictionPackageWithTestData:
    """Tests when df_test provides ground-truth for the forecast horizon."""

    def test_test_actual_present(self, mock_forecaster, y_train, df_test_matching):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert "test_actual" in pkg

    def test_test_actual_is_series(self, mock_forecaster, y_train, df_test_matching):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert isinstance(pkg["test_actual"], pd.Series)

    def test_test_actual_non_empty(self, mock_forecaster, y_train, df_test_matching):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert len(pkg["test_actual"]) > 0

    def test_metrics_future_has_mae(self, mock_forecaster, y_train, df_test_matching):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert "mae" in pkg["metrics_future"]

    def test_metrics_future_has_mape(self, mock_forecaster, y_train, df_test_matching):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert "mape" in pkg["metrics_future"]

    def test_metrics_future_mae_is_float(
        self, mock_forecaster, y_train, df_test_matching
    ):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert isinstance(pkg["metrics_future"]["mae"], float)

    def test_metrics_future_mape_is_float(
        self, mock_forecaster, y_train, df_test_matching
    ):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert isinstance(pkg["metrics_future"]["mape"], float)

    def test_metrics_future_mae_non_negative(
        self, mock_forecaster, y_train, df_test_matching
    ):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_matching
        )
        assert pkg["metrics_future"]["mae"] >= 0.0


# =============================================================================
# Tests: build_prediction_package — df_test target column absent
# =============================================================================


class TestBuildPredictionPackageTestTargetMissing:
    """When df_test does not contain the target column, test_actual is absent."""

    def test_test_actual_key_absent(self, mock_forecaster, y_train, df_test_no_match):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_no_match
        )
        assert "test_actual" not in pkg

    def test_metrics_future_empty_dict(
        self, mock_forecaster, y_train, df_test_no_match
    ):
        pkg = build_prediction_package(
            mock_forecaster, "load", y_train, N_FUTURE, df_test=df_test_no_match
        )
        assert pkg["metrics_future"] == {}


# =============================================================================
# Tests: build_prediction_package — df_test with no index overlap
# =============================================================================


class TestBuildPredictionPackageTestNoOverlap:
    """When df_test timestamps do not overlap future_pred, metrics stay empty."""

    def test_test_actual_present_but_empty(
        self, mock_forecaster, y_train, df_test_non_overlapping
    ):
        pkg = build_prediction_package(
            mock_forecaster,
            "load",
            y_train,
            N_FUTURE,
            df_test=df_test_non_overlapping,
        )
        # test_actual key exists but is empty after dropna / reindex
        assert "test_actual" in pkg
        assert len(pkg["test_actual"]) == 0

    def test_metrics_future_stays_empty(
        self, mock_forecaster, y_train, df_test_non_overlapping
    ):
        pkg = build_prediction_package(
            mock_forecaster,
            "load",
            y_train,
            N_FUTURE,
            df_test=df_test_non_overlapping,
        )
        assert pkg["metrics_future"] == {}


# =============================================================================
# Tests: build_prediction_package — package-level import
# =============================================================================


class TestBuildPredictionPackageImport:
    """Verify the function is accessible from the package root."""

    def test_importable_from_manager(self):
        from spotforecast2_safe.manager import build_prediction_package as bpp

        assert callable(bpp)

    def test_same_object_as_predictor_module(self):
        from spotforecast2_safe.manager import build_prediction_package as bpp1
        from spotforecast2_safe.manager.predictor import (
            build_prediction_package as bpp2,
        )

        assert bpp1 is bpp2


# =============================================================================
# Tests: build_prediction_package — docstring living examples
# =============================================================================


class TestBuildPredictionPackageDocstringExamples:
    """Reproduce both {python} examples from the docstring."""

    def test_example_no_exog(self):
        """Example 1: no exogenous features, no test data."""
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        y_train = pd.Series(rng.normal(100, 10, 200), index=idx, name="load")

        forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)
        forecaster.fit(y=y_train)

        pkg = build_prediction_package(
            forecaster=forecaster,
            target="load",
            y_train=y_train,
            predict_size=24,
        )

        assert isinstance(pkg, dict)
        assert len(pkg["future_pred"]) == 24
        assert pkg["validation_passed"] is True
        assert "mae" in pkg["metrics_train"]
        assert pkg["metrics_future"] == {}
        assert "test_actual" not in pkg

    def test_example_with_test_data(self):
        """Example 2: with test DataFrame covering the forecast horizon."""
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        y_train = pd.Series(rng.normal(50, 5, 200), index=idx, name="power")

        forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)
        forecaster.fit(y=y_train)

        df_test = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-01-09 08:00", periods=24, freq="h"),
                "power": rng.normal(50, 5, 24),
            }
        )

        pkg = build_prediction_package(
            forecaster=forecaster,
            target="power",
            y_train=y_train,
            predict_size=24,
            df_test=df_test,
        )

        assert "test_actual" in pkg
        assert len(pkg["test_actual"]) == 24
        assert "mae" in pkg["metrics_future"]
        assert "mape" in pkg["metrics_future"]


if __name__ == "__main__":
    unittest.main()
