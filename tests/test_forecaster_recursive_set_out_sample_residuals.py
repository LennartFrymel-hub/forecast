# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.exceptions import NotFittedError
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_set_out_sample_residuals_not_fitted_error():
    """
    Test NotFittedError is raised if forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y_true = np.arange(10)
    y_pred = np.arange(10)

    with pytest.raises(NotFittedError):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_type_error():
    """
    Test TypeError is raised if y_true or y_pred are not array/Series.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    msg = "`y_true` argument must be `numpy ndarray` or `pandas Series`."
    with pytest.raises(TypeError, match=msg):
        forecaster.set_out_sample_residuals(y_true=[1, 2], y_pred=np.array([1, 2]))

    msg = "`y_pred` argument must be `numpy ndarray` or `pandas Series`."
    with pytest.raises(TypeError, match=msg):
        forecaster.set_out_sample_residuals(y_true=np.array([1, 2]), y_pred=[1, 2])


def test_set_out_sample_residuals_length_mismatch():
    """
    Test ValueError is raised if y_true and y_pred have different lengths.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    msg = "`y_true` and `y_pred` must have the same length."
    with pytest.raises(ValueError, match=msg):
        forecaster.set_out_sample_residuals(y_true=np.arange(5), y_pred=np.arange(6))


def test_set_out_sample_residuals_index_mismatch():
    """
    Test ValueError is raised if y_true and y_pred are Series with different indices.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    y_true = pd.Series(np.arange(5), index=np.arange(5))
    y_pred = pd.Series(np.arange(5), index=np.arange(5) + 1)

    msg = "`y_true` and `y_pred` must have the same index."
    with pytest.raises(ValueError, match=msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_success():
    """
    Test successful update of out_sample_residuals.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)

    # Initially None
    assert forecaster.out_sample_residuals_ is None

    y_true = np.array([10, 11, 12])
    # Assume perfect prediction for simplicity of residuals
    y_pred = np.array([10, 11, 12])

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert forecaster.out_sample_residuals_ is not None
    assert len(forecaster.out_sample_residuals_) == 3
    # Residuals should be 0
    np.testing.assert_array_equal(forecaster.out_sample_residuals_, np.zeros(3))


def test_set_out_sample_residuals_append():
    """
    Test appending residuals.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)

    y_true_1 = np.array([10, 11])
    y_pred_1 = np.array([10, 11])

    forecaster.set_out_sample_residuals(y_true=y_true_1, y_pred=y_pred_1)
    assert len(forecaster.out_sample_residuals_) == 2

    y_true_2 = np.array([12, 13])
    y_pred_2 = np.array([12, 13])

    forecaster.set_out_sample_residuals(y_true=y_true_2, y_pred=y_pred_2, append=True)
    assert len(forecaster.out_sample_residuals_) == 4


def test_set_out_sample_residuals_with_differentiation():
    """
    Test set_out_sample_residuals works with differentiation.
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    y = pd.Series(np.arange(20), name="y")
    forecaster.fit(y=y, store_in_sample_residuals=False)

    # When differentiation is used, y_true and y_pred are usually passed in original scale.
    # The method differentiates them before calculating residuals.

    y_true = np.array([10, 12, 14])  # diff -> [2, 2]
    y_pred = np.array([10, 11, 13])  # diff -> [1, 2]
    # Residuals: [2-1, 2-2] -> [1, 0]

    # Note: explicit diff logic inside the method skips the first 'differentiation' elements
    # fit_transform on [10, 12, 14] -> [nan, 2, 2] -> slice [1:] -> [2, 2]

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert forecaster.out_sample_residuals_ is not None
    assert len(forecaster.out_sample_residuals_) == 2
    np.testing.assert_array_equal(
        forecaster.out_sample_residuals_, np.array([1.0, 0.0])
    )
