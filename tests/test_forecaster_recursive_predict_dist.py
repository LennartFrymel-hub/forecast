# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_predict_dist_output_shape_and_columns():
    """
    Test that predict_dist returns a DataFrame with the correct shape and column names
    when using a normal distribution.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 3
    predictions = forecaster.predict_dist(steps=steps, distribution=norm, n_boot=100)

    expected_columns = ["loc", "scale"]

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (steps, len(expected_columns))
    assert list(predictions.columns) == expected_columns
    assert predictions.index.name == y.index.name


def test_predict_dist_invalid_distribution_error():
    """
    Test that predict_dist raises TypeError when an invalid distribution object is provided.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    class InvalidDist:
        pass

    msg = (
        "`distribution` must be a valid probability distribution object "
        "from scipy.stats, with methods `_pdf` and `fit`."
    )

    with pytest.raises(TypeError, match=msg):
        forecaster.predict_dist(steps=3, distribution=InvalidDist)


def test_predict_dist_values_check():
    """
    Test that the estimated parameters (loc, scale) are reasonable.
    For a naive linear trend, the mean (loc) should closely match the point prediction.
    """
    # Linear trend y = x
    y = pd.Series(np.arange(100, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    steps = 5
    dist_preds = forecaster.predict_dist(
        steps=steps, distribution=norm, n_boot=100, random_state=123
    )

    # Point predictions
    point_preds = forecaster.predict(steps=steps)

    # The 'loc' parameter of the normal distribution should be very close to the point prediction
    # because the residuals are small and symmetric around 0 in this perfect linear case
    pd.testing.assert_series_equal(
        dist_preds["loc"], point_preds, check_names=False, rtol=0.01
    )

    # Scale should be small (non-negative)
    assert (dist_preds["scale"] >= 0).all()


def test_predict_dist_no_residuals_error():
    """
    Test that predict_dist raises ValueError when residuals are not available.
    """
    y = pd.Series(np.arange(50), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)

    # Fit WITHOUT storing residuals
    forecaster.fit(y=y, store_in_sample_residuals=False)

    # We expect the same error logic as in predict_quantiles because it calls predict_bootstrapping
    # The error logic resides in check_residuals_input which is called deep down
    # But usually predict_bootstrapping calls _create_predict_inputs -> check_residuals_input

    msg = (
        "`forecaster.in_sample_residuals_by_bin_` is either None or empty. Use "
        "`store_in_sample_residuals = True` when fitting the forecaster "
        "or use the `set_in_sample_residuals()` method before predicting."
    )

    import re

    with pytest.raises(ValueError, match=re.escape(msg)):
        forecaster.predict_dist(
            steps=3, distribution=norm, use_in_sample_residuals=True
        )
