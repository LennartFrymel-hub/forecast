# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from spotforecast2_safe.exceptions import NotFittedError
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


class MockEstimatorImportances(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.feature_importances_ = np.array([0.5, 0.2, 0.3])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MockEstimatorCoef(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = np.array([0.5, 0.2, 0.3])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MockEstimatorNoAttribs(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_get_feature_importances_not_fitted_error():
    """
    Test NotFittedError is raised if forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with pytest.raises(NotFittedError):
        forecaster.get_feature_importances()


def test_get_feature_importances_attribute():
    """
    Test feature importances are returned when estimator has feature_importances_.
    """
    forecaster = ForecasterRecursive(estimator=MockEstimatorImportances(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=False)

    # Manually set X_train_features_names_out_ because MockEstimator doesn't really fit
    forecaster.X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3"]

    importances = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame(
        {"feature": ["lag_1", "lag_2", "lag_3"], "importance": [0.5, 0.2, 0.3]}
    )

    pd.testing.assert_frame_equal(importances, expected)


def test_get_feature_importances_coef():
    """
    Test feature importances are returned when estimator has coef_.
    """
    forecaster = ForecasterRecursive(estimator=MockEstimatorCoef(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=False)
    forecaster.X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3"]

    importances = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame(
        {"feature": ["lag_1", "lag_2", "lag_3"], "importance": [0.5, 0.2, 0.3]}
    )

    pd.testing.assert_frame_equal(importances, expected)


def test_get_feature_importances_sorting():
    """
    Test feature importances are sorted correctly.
    """
    forecaster = ForecasterRecursive(estimator=MockEstimatorImportances(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=False)
    forecaster.X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3"]

    # [0.5, 0.2, 0.3] -> sorted descending: 0.5 (lag_1), 0.3 (lag_3), 0.2 (lag_2)
    importances = forecaster.get_feature_importances(sort_importance=True)

    expected = pd.DataFrame(
        {"feature": ["lag_1", "lag_3", "lag_2"], "importance": [0.5, 0.3, 0.2]}
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(importances.reset_index(drop=True), expected)


def test_get_feature_importances_no_attributes_warning():
    """
    Test warning is raised and None returned when estimator has neither attribute.
    """
    forecaster = ForecasterRecursive(estimator=MockEstimatorNoAttribs(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=False)

    with pytest.warns(UserWarning, match="Impossible to access feature importances"):
        importances = forecaster.get_feature_importances()

    assert importances is None


def test_get_feature_importances_pipeline():
    """
    Test feature importances work with Pipeline estimator.
    """
    pipe = Pipeline([("est", MockEstimatorImportances())])
    forecaster = ForecasterRecursive(estimator=pipe, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=False)
    forecaster.X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3"]

    importances = forecaster.get_feature_importances(sort_importance=False)

    expected = pd.DataFrame(
        {"feature": ["lag_1", "lag_2", "lag_3"], "importance": [0.5, 0.2, 0.3]}
    )

    pd.testing.assert_frame_equal(importances, expected)
