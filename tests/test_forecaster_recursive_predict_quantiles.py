# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_predict_quantiles_output_shape_and_columns():
    """
    Test that predict_quantiles returns a DataFrame with the correct shape and column names.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    quantiles = [0.1, 0.5, 0.9]
    steps = 5
    predictions = forecaster.predict_quantiles(
        steps=steps, quantiles=quantiles, n_boot=100
    )

    expected_columns = [f"q_{q}" for q in quantiles]

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(quantiles))
    assert list(predictions.columns) == expected_columns
    assert predictions.index.name == y.index.name


def test_predict_quantiles_values_logic():
    """
    Test that predicted quantiles follow logical order (q_0.1 <= q_0.5 <= q_0.9).
    """
    y = pd.Series(np.random.normal(loc=10, scale=1, size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    quantiles = [0.1, 0.5, 0.9]
    predictions = forecaster.predict_quantiles(
        steps=5, quantiles=quantiles, n_boot=500  # Higher n_boot for stability
    )

    # Check that q_0.1 <= q_0.5
    assert (predictions["q_0.1"] <= predictions["q_0.5"]).all()
    # Check that q_0.5 <= q_0.9
    assert (predictions["q_0.5"] <= predictions["q_0.9"]).all()


def test_predict_quantiles_no_residuals_error():
    """
    Test that predict_quantiles raises ValueError when residuals are not available.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)

    # Fit WITHOUT storing residuals
    forecaster.fit(y=y, store_in_sample_residuals=False)

    msg = (
        "`forecaster.in_sample_residuals_by_bin_` is either None or empty. Use "
        "`store_in_sample_residuals = True` when fitting the forecaster "
        "or use the `set_in_sample_residuals()` method before predicting."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        forecaster.predict_quantiles(steps=5, use_in_sample_residuals=True)


def test_predict_quantiles_input_validation():
    """
    Test that predict_quantiles validates input arguments via check_interval.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # Invalid quantiles (outside 0-1 range)
    with pytest.raises(ValueError):
        forecaster.predict_quantiles(steps=5, quantiles=[1.5])

    with pytest.raises(ValueError):
        forecaster.predict_quantiles(steps=5, quantiles=[-0.1])


def test_predict_quantiles_reproducibility():
    """
    Test that predict_quantiles is reproducible with the same random_state.
    """
    y = pd.Series(np.random.normal(size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    pred_1 = forecaster.predict_quantiles(steps=5, n_boot=50, random_state=42)
    pred_2 = forecaster.predict_quantiles(steps=5, n_boot=50, random_state=42)
    pred_3 = forecaster.predict_quantiles(steps=5, n_boot=50, random_state=123)

    pd.testing.assert_frame_equal(pred_1, pred_2)
    assert not pred_1.equals(pred_3)
