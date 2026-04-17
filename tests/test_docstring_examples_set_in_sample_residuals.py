# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


def test_docstring_example_set_in_sample_residuals():
    """
    Test the docstring example for set_in_sample_residuals.
    """
    # Imports from the example
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    # Example code
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(y=pd.Series(np.arange(20)))

    # Verification
    expected_residuals = np.zeros(17)  # 20 - 3 lags = 17 residuals
    # The docstring says 20 zeros, but fit consumes lags.
    # Let's check what happened.
    # If len(y)=20, lags=3. X_train has 17 samples. Predictions are 17. Residuals are 17.
    # The docstring example output shows 20 zeros. This might be incorrect.

    # Let's verify what the actual output is.
    print(f"Residuals shape: {forecaster.in_sample_residuals_.shape}")
    print(f"Residuals: {forecaster.in_sample_residuals_}")

    # Asserting what I suspect is correct (length 17)
    assert len(forecaster.in_sample_residuals_) == 17
    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_, expected_residuals
    )
