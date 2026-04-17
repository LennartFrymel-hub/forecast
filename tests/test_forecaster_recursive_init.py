import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive._forecaster_recursive import (
    ForecasterRecursive,
)


@pytest.fixture
def sample_series():
    return pd.Series(np.random.randn(100), name="y")


@pytest.fixture
def sample_exog():
    return pd.DataFrame({"temp": np.random.randn(100)}, index=pd.RangeIndex(100))


@pytest.mark.parametrize("lags", [3, [1, 2, 3], np.array([1, 2, 3]), range(1, 4)])
def test_init_lags(sample_series, lags):
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=lags)
    assert forecaster.lags is not None
    assert forecaster.lags_names is not None
    assert forecaster.max_lag is not None
    assert forecaster.window_size >= forecaster.max_lag


def test_init_window_features(sample_series):
    from spotforecast2_safe.preprocessing import RollingFeatures

    wf = RollingFeatures(stats="mean", window_sizes=7)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, window_features=[wf]
    )
    assert forecaster.window_features is not None
    assert forecaster.window_features_names is not None
    assert forecaster.window_features_class_names is not None
    assert forecaster.window_size >= forecaster.max_lag


def test_init_differentiation(sample_series):
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    assert forecaster.differentiation == 1
    assert forecaster.differentiator is not None
    assert forecaster.window_size >= forecaster.max_lag + 1


def test_init_binner_kwargs(sample_series):
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        binner_kwargs={"n_bins": 5, "method": "linear"},
    )
    assert forecaster.binner_kwargs["n_bins"] == 5
    assert forecaster.binner_kwargs["method"] == "linear"
    assert forecaster.binner is not None


def test_init_version(sample_series):
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    assert hasattr(forecaster, "spotforecast_version")
    assert hasattr(forecaster, "python_version")


def test_fit_and_predict(sample_series):
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(sample_series)
    preds = forecaster.predict(steps=5)
    assert len(preds) == 5


def test_fit_with_exog(sample_series, sample_exog):
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(sample_series, exog=sample_exog)
    exog_future = pd.DataFrame(
        {"temp": np.random.randn(5)}, index=pd.RangeIndex(100, 105)
    )
    preds = forecaster.predict(steps=5, exog=exog_future)
    assert len(preds) == 5


def test_invalid_differentiation(sample_series):
    with pytest.raises(ValueError):
        ForecasterRecursive(estimator=LinearRegression(), lags=3, differentiation=0)
    with pytest.raises(ValueError):
        ForecasterRecursive(estimator=LinearRegression(), lags=3, differentiation=-1)


def test_invalid_lags_and_window_features():
    with pytest.raises(ValueError):
        ForecasterRecursive(
            estimator=LinearRegression(), lags=None, window_features=None
        )
