# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


def test_docstring_example_set_out_sample_residuals():
    """
    Test the docstring example for set_out_sample_residuals.
    """
    # Imports from the example
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    # Example code
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)

    # Create dummy out-of-sample data
    y_true = np.array([20, 21, 22, 23, 24])
    y_pred = np.array([20.1, 20.9, 22.2, 22.8, 24.0])

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    # Verification
    expected_residuals = y_true - y_pred

    print(f"Residuals: {forecaster.out_sample_residuals_}")

    np.testing.assert_array_almost_equal(
        forecaster.out_sample_residuals_, expected_residuals
    )
    assert len(forecaster.out_sample_residuals_) == 5
