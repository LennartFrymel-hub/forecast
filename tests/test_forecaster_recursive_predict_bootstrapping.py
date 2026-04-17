# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


def test_recursive_predict_bootstrapping_output_shape():
    """
    Test _recursive_predict_bootstrapping returns expected shape (steps, n_boot).
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    steps = 4
    n_boot = 5
    last_window_values = np.array([7.0, 8.0, 9.0])
    sampled_residuals = np.random.normal(size=(steps, n_boot))

    predictions = forecaster._recursive_predict_bootstrapping(
        steps=steps,
        last_window_values=last_window_values,
        sampled_residuals=sampled_residuals,
        use_binned_residuals=False,
        n_boot=n_boot,
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (steps, n_boot)


def test_recursive_predict_bootstrapping_linear_behavior():
    """
    Test _recursive_predict_bootstrapping logic with a linear model.
    """
    # y = lag1 + residual
    y = pd.Series(np.arange(10, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
    forecaster.fit(y=y)

    steps = 2
    n_boot = 3
    last_window_values = np.array([9.0])
    # Residuals: step1: [1, 2, 3], step2: [0.1, 0.2, 0.3]
    sampled_residuals = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])

    predictions = forecaster._recursive_predict_bootstrapping(
        steps=steps,
        last_window_values=last_window_values,
        sampled_residuals=sampled_residuals,
        use_binned_residuals=False,
        n_boot=n_boot,
    )

    # Step 1:
    # lag1 = 9
    # base_pred = 10 (since y=lag1+1)
    # final_pred = 10 + [1, 2, 3] = [11, 12, 13]
    # Step 2:
    # lag1 = [11, 12, 13]
    # base_pred = lag1 + 1 = [12, 13, 14]
    # final_pred = [12, 13, 14] + [0.1, 0.2, 0.3] = [12.1, 13.2, 14.3]

    expected = np.array([[11.0, 12.0, 13.0], [12.1, 13.2, 14.3]])
    np.testing.assert_allclose(predictions, expected, atol=1e-10)


def test_recursive_predict_bootstrapping_with_window_features():
    """
    Test _recursive_predict_bootstrapping with window features.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=1,
        window_features=[RollingFeatures(stats="mean", window_sizes=2)],
    )
    forecaster.fit(y=y)

    steps = 1
    n_boot = 2
    last_window_values = np.array([8.0, 9.0])
    sampled_residuals = np.array([[0.5, -0.5]])

    predictions = forecaster._recursive_predict_bootstrapping(
        steps=steps,
        last_window_values=last_window_values,
        sampled_residuals=sampled_residuals,
        use_binned_residuals=False,
        n_boot=n_boot,
    )

    assert predictions.shape == (1, 2)
    assert not np.isnan(predictions).any()


def test_recursive_predict_bootstrapping_with_exog():
    """
    Test _recursive_predict_bootstrapping with exogenous variables.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    exog = pd.DataFrame({"exog1": np.arange(10, 20, dtype=float)}, index=y.index)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
    forecaster.fit(y=y, exog=exog)

    steps = 1
    n_boot = 3
    last_window_values = np.array([9.0])
    exog_values = np.array([[20.0]])
    sampled_residuals = np.array([[0.0, 0.0, 0.0]])

    predictions = forecaster._recursive_predict_bootstrapping(
        steps=steps,
        last_window_values=last_window_values,
        sampled_residuals=sampled_residuals,
        exog_values=exog_values,
        use_binned_residuals=False,
        n_boot=n_boot,
    )

    assert predictions.shape == (1, 3)
