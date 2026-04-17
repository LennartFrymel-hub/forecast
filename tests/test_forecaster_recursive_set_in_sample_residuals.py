# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.exceptions import NotFittedError
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_set_in_sample_residuals_not_fitted_error():
    """
    Test NotFittedError is raised if forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = pd.Series(np.arange(10), name="y")

    with pytest.raises(NotFittedError):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_invalid_index_error():
    """
    Test IndexError is raised if y index range maintains training range.
    """
    y = pd.Series(np.arange(20), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y[:15])

    # Try to set residuals with different time range (full y)
    msg = "The index range of `y` does not match the range used during training"
    with pytest.raises(IndexError, match=msg):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_feature_mismatch_error():
    """
    Test ValueError is raised if feature names don't match (e.g. different exog).
    """
    y = pd.Series(np.arange(20), name="y")
    exog = pd.Series(np.arange(20), name="exog")

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    # Try to set residuals without exog
    msg = "Feature mismatch detected after matrix creation"
    with pytest.raises(ValueError, match=msg):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_success():
    """
    Test that set_in_sample_residuals successfully updates residuals attributes.
    """
    y = pd.Series(np.random.normal(size=50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Fit without storing residuals initially
    forecaster.fit(y=y, store_in_sample_residuals=False)
    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None

    # Set residuals
    forecaster.set_in_sample_residuals(y=y)

    assert forecaster.in_sample_residuals_ is not None
    assert len(forecaster.in_sample_residuals_) > 0
    assert isinstance(forecaster.in_sample_residuals_, np.ndarray)
    assert forecaster.in_sample_residuals_by_bin_ is not None
    assert isinstance(forecaster.in_sample_residuals_by_bin_, dict)


def test_set_in_sample_residuals_with_differentiation():
    """
    Test set_in_sample_residuals works correctly with differentiation.
    """
    y = pd.Series(np.random.normal(size=50), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )

    forecaster.fit(y=y, store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(y=y)

    assert forecaster.in_sample_residuals_ is not None
    # With differentiation, residuals should still be calculated and stored
    assert len(forecaster.in_sample_residuals_) > 0
