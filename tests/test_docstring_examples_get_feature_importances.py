# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


def test_docstring_example_get_feature_importances():
    """
    Test the docstring example for get_feature_importances.
    """
    # Imports from the example
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    # Example code
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20)))
    importances = forecaster.get_feature_importances()

    # Verification
    print(f"Importances:\n{importances}")

    assert isinstance(importances, pd.DataFrame)
    assert "feature" in importances.columns
    assert "importance" in importances.columns
    assert len(importances) == 3  # 3 lags
    assert importances["feature"].tolist() == ["lag_1", "lag_2", "lag_3"]
    # LinearRegression on arange(20) with lags should have specific coefficients.
    # lag_1 should be 1, others 0 ideally?
    # series: 0, 1, 2, ...
    # y = 1*lag_1 + 0*lag_2 + 0*lag_3 + 1 (intercept) roughly?
    # y_t = y_{t-1} + 1
    # So lag_1 coeff ~ 1.

    # Check if sorting works (descending by default)
    assert importances.iloc[0]["feature"] == "lag_1"
