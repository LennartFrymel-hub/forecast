# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


def test_docstring_example_set_window_features():
    """
    Test the docstring example for set_window_features.
    """
    # Imports from the example
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2_safe.preprocessing import RollingFeatures

    # Example code
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    rolling = RollingFeatures(stats=["mean", "std"], window_sizes=[3, 5])
    forecaster.set_window_features(window_features=rolling)

    # Verification
    print(f"Window features names: {forecaster.window_features_names}")
    print(f"Window size: {forecaster.window_size}")

    assert forecaster.window_features_names == [
        "roll_mean_3",
        "roll_std_3",
        "roll_mean_5",
        "roll_std_5",
    ]
    assert forecaster.window_size == 5  # max(3 lags, 5 window_size)
