# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursiveMultiSeries


def test_forecaster_recursive_multiseries_init():
    """
    Test initialization of ForecasterRecursiveMultiSeries.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=5)
    assert forecaster.lags.tolist() == [1, 2, 3, 4, 5]
    assert forecaster.window_size == 5
    assert forecaster.encoding == "ordinal"


def test_forecaster_recursive_multiseries_fit_predict():
    """
    Test fit and predict methods of ForecasterRecursiveMultiSeries.
    """
    # Create dummy data
    series = pd.DataFrame({"series_1": np.arange(20), "series_2": np.arange(20) + 100})
    series.index = pd.date_range(start="2020-01-01", periods=20, freq="D")

    forecaster = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=3)

    forecaster.fit(series=series)
    assert forecaster.is_fitted

    predictions = forecaster.predict(steps=5)

    # ForecasterRecursiveMultiSeries returns long format: columns ['level', 'pred']
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (10, 2)  # 5 steps * 2 levels
    assert list(predictions.columns) == ["level", "pred"]
    assert set(predictions["level"].unique()) == {"series_1", "series_2"}

    # Check index frequency (repeated index)
    # assert predictions.index.freq == 'D' # Index might not have freq if repeated
    assert predictions.index.nunique() == 5


def test_forecaster_recursive_multiseries_predict_subset():
    """
    Test predict method with a subset of levels.
    """
    series = pd.DataFrame({"series_1": np.arange(20), "series_2": np.arange(20) + 100})
    series.index = pd.date_range(start="2020-01-01", periods=20, freq="D")

    forecaster = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=3)

    forecaster.fit(series=series)

    predictions = forecaster.predict(steps=5, levels="series_1")
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (5, 2)  # 5 steps * 1 level, 2 columns (level, pred)
    assert list(predictions.columns) == ["level", "pred"]
    assert predictions["level"].unique().tolist() == ["series_1"]
