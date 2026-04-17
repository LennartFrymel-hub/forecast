# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_predict_output_series():
    """
    Test predict returns a pandas Series with correct index and name.
    """
    y = pd.Series(
        np.arange(10, dtype=float),
        name="y",
        index=pd.date_range("2020-01-01", periods=10),
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    steps = 5
    predictions = forecaster.predict(steps=steps)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == steps
    assert predictions.name == "pred"
    pd.testing.assert_index_equal(
        predictions.index, pd.date_range("2020-01-11", periods=5)
    )


def test_predict_inverse_transform():
    """
    Test predict correctly applies inverse transformation.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    # Scaling y so inverse transform is non-trivial
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=2, transformer_y=StandardScaler()
    )
    forecaster.fit(y=y)

    predictions = forecaster.predict(steps=2)

    # Check that output is in original scale (not scaled)
    assert predictions.max() > 1.0  # Standard scaled 0-9 would be ~ -1.5 to 1.5


def test_predict_differentiation_inverse():
    """
    Test predict correctly applies inverse differentiation.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    # y = [0, 1, 2, ..., 9]
    # diff1 = [1, 1, 1, ..., 1]
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=1, differentiation=1
    )
    forecaster.fit(y=y)

    # Predict 2 steps
    # last y = 9
    # diff prediction should be 1
    # inverse diff: 9 + 1 = 10, 10 + 1 = 11
    predictions = forecaster.predict(steps=2)

    np.testing.assert_allclose(predictions.values, [10.0, 11.0])
