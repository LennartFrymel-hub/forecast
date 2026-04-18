# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_predict_interval_bootstrapping_output_shape_and_columns():
    """
    Test that predict_interval with method='bootstrapping' returns a DataFrame
    with the correct shape and column names.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 5
    predictions = forecaster.predict_interval(
        steps=steps, method="bootstrapping", interval=[5, 95], n_boot=100
    )

    expected_columns = ["pred", "lower_bound", "upper_bound"]

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(expected_columns))
    assert list(predictions.columns) == expected_columns
    assert predictions.index.name == y.index.name


def test_predict_interval_conformal_output_shape_and_columns():
    """
    Test that predict_interval with method='conformal' returns a DataFrame
    with the correct shape and column names.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 5
    predictions = forecaster.predict_interval(
        steps=steps, method="conformal", interval=[5, 95]
    )

    expected_columns = ["pred", "lower_bound", "upper_bound"]

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(expected_columns))
    assert list(predictions.columns) == expected_columns
    assert predictions.index.name == y.index.name


def test_predict_interval_values_logic():
    """
    Test that lower_bound <= pred <= upper_bound for both methods.
    """
    y = pd.Series(np.random.normal(loc=10, scale=1, size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # Bootstrapping
    pred_boot = forecaster.predict_interval(
        steps=5, method="bootstrapping", interval=[5, 95], n_boot=100
    )
    assert (pred_boot["lower_bound"] <= pred_boot["pred"]).all()
    assert (pred_boot["pred"] <= pred_boot["upper_bound"]).all()

    # Conformal
    pred_conf = forecaster.predict_interval(steps=5, method="conformal", interval=0.9)
    assert (pred_conf["lower_bound"] <= pred_conf["pred"]).all()
    assert (pred_conf["pred"] <= pred_conf["upper_bound"]).all()


def test_predict_interval_invalid_method_error():
    """
    Test that predict_interval raises ValueError when method is invalid.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    msg = "Invalid `method` 'invalid_method'. Choose 'bootstrapping' or 'conformal'."

    with pytest.raises(ValueError, match=msg):
        forecaster.predict_interval(steps=3, method="invalid_method")


def test_predict_interval_conformal_invalid_interval_error():
    """
    Test that predict_interval raises correct error (via check_interval)
    when interval is invalid for conformal (e.g., asymmetric list).
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # Conformal requires symmetric intervals or single float
    # If list is provided, check_interval(ensure_symmetric_intervals=True) is called

    with pytest.raises(ValueError):
        forecaster.predict_interval(steps=3, method="conformal", interval=[10, 80])
