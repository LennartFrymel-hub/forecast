# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)
from spotforecast2_safe.manager.models.forecaster_recursive_xgb import (
    ForecasterRecursiveXGB,
)


def test_forecaster_recursive_xgb_initialization():
    """Test that ForecasterRecursiveXGB initializes correctly."""
    model = ForecasterRecursiveXGB(iteration=0)
    assert model.iteration == 0
    assert model.name == "xgb"
    assert isinstance(model, ForecasterRecursiveModel)


def test_forecaster_recursive_xgb_inheritance():
    """Test inheritance from ForecasterRecursiveModel."""
    assert issubclass(ForecasterRecursiveXGB, ForecasterRecursiveModel)


def test_forecaster_recursive_xgb_import_location():
    """Verify it can be imported from the new location."""
    from spotforecast2_safe.manager.models import ForecasterRecursiveXGB as XGB

    assert XGB is ForecasterRecursiveXGB


def test_forecaster_recursive_xgb_no_longer_in_preprocessing():
    """Verify it's removed from preprocessing __init__."""
    import spotforecast2_safe.preprocessing as preprocessing

    assert not hasattr(preprocessing, "ForecasterRecursiveXGB")


@pytest.fixture
def sample_ts_data():
    """Simple time series for testing."""
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    y = pd.Series(np.random.randn(100), index=idx, name="load")
    return y


def test_forecaster_recursive_xgb_fit_predict_interface(sample_ts_data):
    """Test the fit/predict interface (smoke test)."""
    model = ForecasterRecursiveXGB(iteration=0, lags=3)

    # If XGBoost is available, we can test fit/predict
    # If not, it should log a warning but the interface exists
    from spotforecast2_safe.manager.models.forecaster_recursive_xgb import XGBRegressor

    if XGBRegressor is not None:
        model.fit(sample_ts_data)
        predictions = model.forecaster.predict(steps=5)
        assert len(predictions) == 5
    else:
        pytest.skip("XGBoost not installed, skipping fit/predict test")


def test_package_prediction_interface():
    """Test package_prediction method exists."""
    model = ForecasterRecursiveXGB(iteration=0)
    assert hasattr(model, "package_prediction")
