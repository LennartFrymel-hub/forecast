import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

# If RollingFeatures is not available, skip that test
try:
    from spotforecast2_safe.preprocessing import RollingFeatures

    HAS_ROLLING = True
except ImportError:
    HAS_ROLLING = False


def test_basic_forecaster_with_lags():
    y = pd.Series(np.random.randn(100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=10)
    forecaster.fit(y)
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert isinstance(predictions, pd.Series)


def test_forecaster_with_window_features_and_transformations():
    if not HAS_ROLLING:
        pytest.skip("RollingFeatures not available")
    y = pd.Series(np.random.randn(100), name="y")
    forecaster = ForecasterRecursive(
        estimator=RandomForestRegressor(n_estimators=100),
        lags=[1, 7, 30],
        window_features=[RollingFeatures(stats="mean", window_sizes=7)],
        transformer_y=StandardScaler(),
        differentiation=1,
    )
    forecaster.fit(y)
    predictions = forecaster.predict(steps=10)
    assert len(predictions) == 10
    assert isinstance(predictions, pd.Series)


def test_forecaster_with_exogenous_variables():
    y = pd.Series(np.random.randn(100), name="target")
    exog = pd.DataFrame({"temp": np.random.randn(100)}, index=y.index)
    forecaster = ForecasterRecursive(
        estimator=Ridge(), lags=7, forecaster_id="my_forecaster"
    )
    forecaster.fit(y, exog)
    exog_future = pd.DataFrame(
        {"temp": np.random.randn(5)}, index=pd.RangeIndex(start=100, stop=105)
    )
    predictions = forecaster.predict(steps=5, exog=exog_future)
    assert len(predictions) == 5
    assert isinstance(predictions, pd.Series)


def test_forecaster_with_probabilistic_prediction():
    y = pd.Series(np.random.randn(100), name="y")
    forecaster = ForecasterRecursive(
        estimator=GradientBoostingRegressor(),
        lags=14,
        binner_kwargs={"n_bins": 15, "method": "linear"},
    )
    forecaster.fit(y, store_in_sample_residuals=True)
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5
    assert isinstance(predictions, pd.Series)
