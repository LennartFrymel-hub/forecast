"""Pytest tests validating docstring examples for get_params/set_params.

These tests mirror the Examples in
:class:`ForecasterRecursiveModel.get_params` and
:class:`ForecasterRecursiveModel.set_params`.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)

# ---------------------------------------------------------------------------
# get_params examples
# ---------------------------------------------------------------------------


class TestGetParams:
    """Validate get_params docstring examples."""

    def test_get_params_no_forecaster_shallow(self):
        """Example 1: wrapper-level params when no forecaster is attached."""
        model = ForecasterRecursiveModel(iteration=0)
        p = model.get_params(deep=False)

        assert p["iteration"] == 0
        assert p["name"] == "base"
        assert p["predict_size"] == 24
        # No forecaster keys when forecaster is None
        assert not any(k.startswith("forecaster__") for k in p)

    def test_get_params_with_forecaster_shallow(self):
        """Example 2: shallow params include forecaster-level keys."""
        model = ForecasterRecursiveModel(iteration=1)
        model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
        p = model.get_params(deep=False)

        assert len(p["forecaster__lags"]) == 3
        assert isinstance(p["forecaster__estimator"], LinearRegression)

    def test_get_params_with_forecaster_deep(self):
        """Example 3: deep=True adds estimator sub-params."""
        model = ForecasterRecursiveModel(iteration=1)
        model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
        p = model.get_params(deep=True)

        assert "forecaster__estimator__fit_intercept" in p

    def test_get_params_wrapper_defaults(self):
        """Verify all expected wrapper keys are present with defaults."""
        model = ForecasterRecursiveModel(iteration=5, predict_size=48, refit_size=14)
        p = model.get_params(deep=False)

        assert p["iteration"] == 5
        assert p["predict_size"] == 48
        assert p["refit_size"] == 14
        assert p["random_state"] == 123456789
        assert p["train_size"] is None
        assert pd.isna(p["end_dev"])


# ---------------------------------------------------------------------------
# set_params examples
# ---------------------------------------------------------------------------


class TestSetParams:
    """Validate set_params docstring examples."""

    def test_set_wrapper_params_via_kwargs(self):
        """Example 1: setting wrapper-level params via kwargs."""
        model = ForecasterRecursiveModel(iteration=0)
        result = model.set_params(name="updated", predict_size=48)

        assert model.name == "updated"
        assert model.predict_size == 48
        # set_params returns self for chaining
        assert result is model

    def test_set_forecaster_estimator_param(self):
        """Example 2: setting estimator param through forecaster__ prefix."""
        model = ForecasterRecursiveModel(iteration=1)
        model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
        model.set_params(params={"forecaster__estimator__fit_intercept": False})

        assert model.forecaster.estimator.fit_intercept is False

    def test_set_params_returns_self(self):
        """set_params supports method chaining."""
        model = ForecasterRecursiveModel(iteration=0)
        chained = model.set_params(name="a").set_params(name="b")
        assert chained.name == "b"
        assert chained is model

    def test_set_params_empty_is_noop(self):
        """Calling set_params with no arguments returns self unchanged."""
        model = ForecasterRecursiveModel(iteration=0)
        assert model.set_params() is model
        assert model.name == "base"  # unchanged
