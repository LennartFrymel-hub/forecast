# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import (
    ForecasterRecursiveLGBM,
)
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)


def test_forecaster_recursive_lgbm_initialization():
    """Test that ForecasterRecursiveLGBM initializes correctly."""
    model = ForecasterRecursiveLGBM(iteration=0)
    assert model.iteration == 0
    assert model.name == "lgbm"
    assert isinstance(model, ForecasterRecursiveModel)
    assert model.forecaster is not None
    assert isinstance(model.forecaster.estimator, LGBMRegressor)


def test_forecaster_recursive_lgbm_inheritance():
    """Test inheritance from ForecasterRecursiveModel."""
    assert issubclass(ForecasterRecursiveLGBM, ForecasterRecursiveModel)


def test_forecaster_recursive_lgbm_import_location():
    """Verify it can be imported from the new location."""
    from spotforecast2_safe.manager.models import ForecasterRecursiveLGBM as LGBM

    assert LGBM is ForecasterRecursiveLGBM


def test_forecaster_recursive_lgbm_no_longer_in_preprocessing():
    """Verify it's removed from preprocessing __init__."""
    import spotforecast2_safe.preprocessing as preprocessing

    assert not hasattr(preprocessing, "ForecasterRecursiveLGBM")


@pytest.fixture
def sample_ts_data():
    """Simple time series for testing."""
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    y = pd.Series(np.random.randn(100), index=idx, name="load")
    return y


def test_forecaster_recursive_lgbm_fit_predict_interface(sample_ts_data):
    """Test the fit/predict interface."""
    model = ForecasterRecursiveLGBM(iteration=0, lags=3)
    # Override forecaster without window_features for small test data
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    model.forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=123456789),
        lags=3,
    )
    model.fit(sample_ts_data)
    predictions = model.forecaster.predict(steps=5)
    assert len(predictions) == 5


def test_package_prediction_interface():
    """Test package_prediction method exists."""
    model = ForecasterRecursiveLGBM(iteration=0)
    assert hasattr(model, "package_prediction")


def test_lgbm_determinism_flags_are_pinned():
    """The LightGBM backend must be instantiated with the flags that
    eliminate the row-wise vs. column-wise histogram divergence on
    multi-core systems. Changing any of these flags silently would break
    the bit-level determinism claim in the compliance narrative; this
    test fails loudly when that happens."""
    model = ForecasterRecursiveLGBM(iteration=0)
    params = model.forecaster.estimator.get_params()
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["random_state"] == model.random_state
