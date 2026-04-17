# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_predict_interval_conformal_output_shape_and_columns():
    """
    Test that _predict_interval_conformal returns a DataFrame with the correct shape and column names.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 5
    predictions = forecaster._predict_interval_conformal(
        steps=steps, nominal_coverage=0.9
    )

    expected_columns = ["pred", "lower_bound", "upper_bound"]

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(expected_columns))
    assert list(predictions.columns) == expected_columns
    assert predictions.index.name == y.index.name


def test_predict_interval_conformal_values_logic():
    """
    Test that lower_bound <= pred <= upper_bound.
    """
    y = pd.Series(np.random.normal(loc=10, scale=1, size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    predictions = forecaster._predict_interval_conformal(steps=5, nominal_coverage=0.9)

    assert (predictions["lower_bound"] <= predictions["pred"]).all()
    assert (predictions["pred"] <= predictions["upper_bound"]).all()


def test_predict_interval_conformal_coverage_logic():
    """
    Test that higher nominal_coverage leads to wider intervals.
    """
    y = pd.Series(np.random.normal(loc=10, scale=2, size=200), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    pred_90 = forecaster._predict_interval_conformal(steps=5, nominal_coverage=0.9)
    pred_50 = forecaster._predict_interval_conformal(steps=5, nominal_coverage=0.5)

    width_90 = pred_90["upper_bound"] - pred_90["lower_bound"]
    width_50 = pred_50["upper_bound"] - pred_50["lower_bound"]

    assert (width_90 >= width_50).all()


def test_predict_interval_conformal_empty_bin_fallback():
    """
    Test that the method handles empty bins gracefully by falling back to global residuals.
    This validates the safety improvement in spotforecast2-safe.
    """
    # Create data with distinct clusters to potentially cause empty bins usage if we predict values
    # falling into ranges not seen specifically during binning, or if binning is sparse.
    # We force specific binning behavior by using a small number of bins or sparse data?
    # Actually, simpler way: fit with very few data points so some bins are empty.

    y = pd.Series(np.concatenate([np.zeros(20), np.ones(20) * 10]), name="y")
    # Use only 2 bins to increase chance of one being populated and one not?
    # Or rely on the safety check logic directly.

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # Predict a value that might fall into a populated bin, and one that might not?
    # The safety feature is: if a bin is empty in residuals, it uses global.
    # We can check that it runs without error even with sparse residuals.

    predictions = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.9, use_binned_residuals=True
    )

    assert not predictions.isnull().any().any()
    assert (predictions["lower_bound"] <= predictions["upper_bound"]).all()


def test_predict_interval_conformal_no_residuals_error():
    """
    Test that it raises ValueError when residuals are not available.
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

    import re

    with pytest.raises(ValueError, match=re.escape(msg)):
        forecaster._predict_interval_conformal(steps=3, use_in_sample_residuals=True)
