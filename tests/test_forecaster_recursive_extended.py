# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


def test_predict_quantiles():
    """
    Test predict_quantiles returns expected shape and column names.
    """
    y = pd.Series(np.arange(20, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 3
    quantiles = [0.1, 0.5, 0.9]
    predictions = forecaster.predict_quantiles(
        steps=steps, quantiles=quantiles, n_boot=10
    )

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(quantiles))
    assert list(predictions.columns) == [f"q_{q}" for q in quantiles]
    assert not predictions.isnull().any().any()


def test_predict_dist():
    """
    Test predict_dist returns expected shape and column names.
    """
    y = pd.Series(np.arange(20, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 2
    predictions = forecaster.predict_dist(steps=steps, distribution=norm, n_boot=10)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, 2)  # norm.fit returns (loc, scale)
    assert list(predictions.columns) == ["loc", "scale"]
    assert not predictions.isnull().any().any()


def test_get_feature_importances():
    """
    Test get_feature_importances returns expected DataFrame.
    """
    y = pd.Series(np.arange(20, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    importances = forecaster.get_feature_importances()
    assert isinstance(importances, pd.DataFrame)
    assert "feature" in importances.columns
    assert "importance" in importances.columns
    assert len(importances) == 3  # 3 lags
    assert list(importances["feature"]) == ["lag_1", "lag_2", "lag_3"]


def test_set_lags():
    """
    Test set_lags updates lags correctly.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    assert forecaster.lags.tolist() == [1, 2, 3]

    forecaster.set_lags(lags=5)
    assert forecaster.lags.tolist() == [1, 2, 3, 4, 5]
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5


def test_set_window_features():
    """
    Test set_window_features updates window features correctly.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    assert forecaster.window_features is None

    wf = RollingFeatures(stats="mean", window_sizes=5)
    forecaster.set_window_features(window_features=wf)

    assert forecaster.window_features is not None
    assert forecaster.max_size_window_features == 5
    assert forecaster.window_size == 5  # max(3, 5) is 5


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs updates fit_kwargs.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    # LinearRegression doesn't have many fit_kwargs, but we can test if it's set.
    # We use an empty dict since LinearRegression.fit doesn't take much.
    forecaster.set_fit_kwargs(fit_kwargs={})
    assert forecaster.fit_kwargs == {}
