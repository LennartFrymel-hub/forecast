import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models import (
    ForecasterRecursiveLGBM,
    ForecasterRecursiveModel,
    ForecasterRecursiveXGB,
)


def test_forecaster_recursive_model_init():
    """Test basic initialization of ForecasterRecursiveModel."""
    model = ForecasterRecursiveModel(iteration=1, end_dev="2024-01-01")
    assert model.iteration == 1
    assert isinstance(model.end_dev, pd.Timestamp)
    assert model.name == "base"
    assert model.forecaster is None
    assert model.is_tuned is False


def test_forecaster_recursive_model_tune():
    """Test the (simulated) tuning method."""
    model = ForecasterRecursiveModel(iteration=0)
    model.tune()
    assert model.is_tuned is True


def test_forecaster_recursive_model_fit(tmp_path):
    """Test fitting with a simple estimator."""
    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)

    y = pd.Series(
        [1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5, freq="h")
    )
    model.fit(y=y)
    assert model.forecaster.is_fitted is True


def test_forecaster_recursive_lgbm_init():
    """Test LGBM specialization."""
    model = ForecasterRecursiveLGBM(iteration=5, lags=24)
    assert model.name == "lgbm"
    assert isinstance(model.forecaster, ForecasterRecursive)
    assert isinstance(model.forecaster.estimator, LGBMRegressor)
    assert model.forecaster.max_lag == 24


def test_forecaster_recursive_xgb_init():
    """Test XGBoost specialization (handles missing installation)."""
    model = ForecasterRecursiveXGB(iteration=2)
    assert model.name == "xgb"
    # If XGBoost is installed, forecaster should be set.
    # Otherwise, it might be None or warn.
    # In our implementation, we set it if XGBRegressor is not None.
    try:
        from xgboost import XGBRegressor

        if XGBRegressor is not None:
            assert model.forecaster is not None
    except ImportError:
        assert model.forecaster is None


def test_forecaster_recursive_model_fit_uninitialized():
    """Test fitting without a forecaster raises ValueError."""
    model = ForecasterRecursiveModel(iteration=0)
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Forecaster not initialized"):
        model.fit(y=y)


def test_package_prediction_no_forecaster():
    """Test package_prediction returns empty dict if forecaster is None."""
    model = ForecasterRecursiveModel(iteration=0)
    assert model.package_prediction() == {}


def test_package_prediction_missing_file(tmp_path, monkeypatch):
    """Test package_prediction handling of missing data file."""
    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)

    # Mock get_data_home to point to an empty temp dir
    import sys

    fd_mod = sys.modules["spotforecast2_safe.data.fetch_data"]
    monkeypatch.setattr(fd_mod, "get_data_home", lambda: tmp_path)

    result = model.package_prediction()
    assert result == {}


def test_forecaster_recursive_model_repr():
    """Verify that the representation contains the name."""
    model = ForecasterRecursiveModel(iteration=0, name="custom")
    # We didn't explicitly define __repr__, but check basics
    assert "ForecasterRecursiveModel" in str(model)
