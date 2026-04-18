# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


class MockEstimator(LinearRegression):
    def fit(self, X, y, sample_weight=None, custom_arg=None):
        return super().fit(X, y, sample_weight=sample_weight)


def test_set_fit_kwargs():
    """
    Test that set_fit_kwargs updates the fit_kwargs attribute.
    """
    forecaster = ForecasterRecursive(estimator=MockEstimator(), lags=3)

    initial_kwargs = {"custom_arg": 10}
    forecaster.fit_kwargs = initial_kwargs

    new_kwargs = {"custom_arg": 20}
    forecaster.set_fit_kwargs(fit_kwargs=new_kwargs)

    assert forecaster.fit_kwargs["custom_arg"] == 20


def test_set_lags():
    """
    Test that set_lags updates lags, lags_names, max_lag, and window_size.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Initial state
    assert np.array_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3

    # Set new lags
    forecaster.set_lags(lags=5)

    assert np.array_equal(forecaster.lags, np.arange(1, 6))
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5
    assert forecaster.lags_names == [f"lag_{i}" for i in range(1, 6)]


def test_set_window_features():
    """
    Test that set_window_features updates attributes correctly.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Initial state
    assert forecaster.window_features is None

    # Create a RollingFeatures instance
    rolling = RollingFeatures(stats=["mean"], window_sizes=3)

    forecaster.set_window_features(window_features=rolling)

    assert forecaster.window_features is not None
    assert len(forecaster.window_features) == 1
    assert isinstance(forecaster.window_features[0], RollingFeatures)
    assert forecaster.window_features_names == ["roll_mean_3"]
    assert forecaster.max_size_window_features == 3
    # window_size should be max of max_lag (3) and max_size_window_features (3) -> 3
    assert forecaster.window_size == 3

    # Update with larger window feature
    rolling_large = RollingFeatures(stats=["mean"], window_sizes=10)
    forecaster.set_window_features(window_features=rolling_large)

    assert forecaster.max_size_window_features == 10
    assert forecaster.window_size == 10


def test_set_lags_and_window_features_error():
    """
    Test that ValueError is raised if both lags and window_features become None.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Trying to set lags to None when window_features is already None
    msg = (
        "At least one of the arguments `lags` or `window_features` "
        "must be different from None."
    )
    with pytest.raises(ValueError, match=msg):
        forecaster.set_lags(lags=None)

    # Setup with window features so we can set lags to None
    rolling = RollingFeatures(stats=["mean"], window_sizes=3)
    forecaster.set_window_features(window_features=rolling)
    forecaster.set_lags(lags=None)
    assert forecaster.lags is None

    # Now try to set window_features to None (lags is already None)
    with pytest.raises(ValueError, match=msg):
        forecaster.set_window_features(window_features=None)


def test_set_lags_with_differentiation():
    """
    Test that window_size is correctly updated when differentiation is active.
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )

    # Initial window_size = max_lag (3) + differentiation (1) = 4
    assert forecaster.window_size == 4

    # Update lags to 5
    forecaster.set_lags(lags=5)

    # New window_size = max_lag (5) + differentiation (1) = 6
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 6
    assert forecaster.differentiator.window_size == 6


def test_set_window_features_with_differentiation():
    """
    Test that window_size is correctly updated when differentiation is active with window features.
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )

    # Initial window_size = 4
    assert forecaster.window_size == 4

    # Update window features with size 10
    rolling = RollingFeatures(stats=["mean"], window_sizes=10)
    forecaster.set_window_features(window_features=rolling)

    # New window_size = max(lags=3, window_features=10) + diff(1) = 11
    assert forecaster.max_size_window_features == 10
    assert forecaster.window_size == 11
    assert forecaster.differentiator.window_size == 11
