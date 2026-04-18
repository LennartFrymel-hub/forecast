# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for class methods ported from chag25a ForecasterRecursiveModel."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)
from spotforecast2_safe.model_selection import TimeSeriesFold

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def base_model():
    """A base model with a simple LinearRegression forecaster."""
    model = ForecasterRecursiveModel(iteration=0, name="test")
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    return model


# ------------------------------------------------------------------ #
# __init__ new attributes
# ------------------------------------------------------------------ #


class TestInitAttributes:
    def test_best_params_default_none(self, base_model):
        assert base_model.best_params is None

    def test_best_lags_default_none(self, base_model):
        assert base_model.best_lags is None

    def test_results_tuning_default_none(self, base_model):
        assert base_model.results_tuning is None

    def test_save_model_to_file_default_true(self, base_model):
        assert base_model.save_model_to_file is True

    def test_save_model_to_file_kwarg(self):
        model = ForecasterRecursiveModel(iteration=0, save_model_to_file=False)
        assert model.save_model_to_file is False

    def test_metrics_default(self, base_model):
        assert base_model.metrics == [
            "mean_absolute_error",
            "mean_absolute_percentage_error",
        ]


# ------------------------------------------------------------------ #
# save_to_file
# ------------------------------------------------------------------ #


class TestSaveToFile:
    def test_creates_joblib_file(self, base_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_model.save_to_file(model_dir=tmpdir)
            files = os.listdir(tmpdir)
            assert any("test_forecaster_0" in f for f in files)

    def test_file_is_loadable(self, base_model):
        from joblib import load

        with tempfile.TemporaryDirectory() as tmpdir:
            base_model.save_to_file(model_dir=tmpdir)
            filepath = Path(tmpdir) / "test_forecaster_0.joblib"
            loaded = load(filepath)
            assert isinstance(loaded, ForecasterRecursiveModel)
            assert loaded.name == "test"


# ------------------------------------------------------------------ #
# _build_cv
# ------------------------------------------------------------------ #


class TestBuildCV:
    def test_returns_time_series_fold(self, base_model):
        cv = base_model._build_cv(train_size=100)
        assert isinstance(cv, TimeSeriesFold)

    def test_steps_matches_predict_size(self, base_model):
        cv = base_model._build_cv(train_size=100)
        assert cv.steps == base_model.predict_size

    def test_initial_train_size(self, base_model):
        cv = base_model._build_cv(train_size=200)
        assert cv.initial_train_size == 200


# ------------------------------------------------------------------ #
# _get_init_train
# ------------------------------------------------------------------ #


class TestGetInitTrain:
    def test_no_train_size_returns_min(self, base_model):
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2025-12-31", tz="UTC")
        result = base_model._get_init_train(start, end)
        assert result == start

    def test_with_train_size(self, base_model):
        base_model.train_size = pd.Timedelta(days=365)
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2025-12-31", tz="UTC")
        result = base_model._get_init_train(start, end)
        expected = end - pd.Timedelta(days=365)
        assert result == expected

    def test_caps_at_min_val(self, base_model):
        base_model.train_size = pd.Timedelta(days=3650)  # 10 years
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2025-12-31", tz="UTC")
        result = base_model._get_init_train(start, end)
        # 10-year lookback would go before start, so cap at start
        assert result == start


# ------------------------------------------------------------------ #
# tune (stub)
# ------------------------------------------------------------------ #


class TestTune:
    def test_marks_as_tuned(self, base_model):
        assert not base_model.is_tuned
        base_model.tune()
        assert base_model.is_tuned

    def test_idempotent(self, base_model):
        base_model.tune()
        base_model.tune()
        assert base_model.is_tuned


# ------------------------------------------------------------------ #
# get_feature_importance
# ------------------------------------------------------------------ #


class TestGetFeatureImportance:
    def test_returns_none_for_unsupported_model(self, base_model):
        base_model.name = "linear"
        result = base_model.get_feature_importance()
        assert result is None

    def test_raises_if_no_forecaster(self):
        model = ForecasterRecursiveModel(iteration=0, name="lgbm")
        model.forecaster = None
        with pytest.raises(ValueError, match="Forecaster not initialized"):
            model.get_feature_importance()


# ------------------------------------------------------------------ #
# get_global_shap_feature_importance (stub)
# ------------------------------------------------------------------ #


class TestShapStub:
    def test_returns_empty_series(self, base_model):
        result = base_model.get_global_shap_feature_importance()
        assert isinstance(result, pd.Series)
        assert len(result) == 0
