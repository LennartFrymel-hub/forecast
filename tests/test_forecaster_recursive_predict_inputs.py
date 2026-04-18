# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_create_predict_inputs_return_values():
    """
    Test _create_predict_inputs return values structure.
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    last_window_values, exog_values, prediction_index, steps = (
        forecaster._create_predict_inputs(steps=5)
    )

    assert isinstance(last_window_values, np.ndarray)
    assert exog_values is None
    assert isinstance(prediction_index, pd.Index)
    assert isinstance(steps, int)
    assert steps == 5


def test_create_predict_inputs_steps_as_date():
    """
    Test _create_predict_inputs when steps is a date string.
    """
    y = pd.Series(
        np.arange(10),
        index=pd.date_range(start="2022-01-01", periods=10, freq="D"),
        name="y",
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    # Predict up to 2022-01-15 (5 steps ahead of 2022-01-10)
    last_window_values, exog_values, prediction_index, steps = (
        forecaster._create_predict_inputs(steps="2022-01-15")
    )

    assert steps == 5
    assert len(prediction_index) == 5
    assert prediction_index[0] == pd.Timestamp("2022-01-11")


def test_create_predict_inputs_exog_column_filtration():
    """
    Test _create_predict_inputs filters exog columns to match training.
    """
    y = pd.Series(np.arange(10), name="y")
    exog = pd.DataFrame({"exog_1": np.arange(10), "exog_2": np.arange(10) * 2})
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    # Pass exog with extra columns or different order
    exog_predict = pd.DataFrame(
        {"exog_2": np.arange(5) * 2, "exog_1": np.arange(5), "extra": np.zeros(5)},
        index=pd.RangeIndex(start=10, stop=15),
    )

    _, exog_values, _, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    # Should match training order: exog_1, exog_2
    expected = exog_predict[["exog_1", "exog_2"]].to_numpy()
    np.testing.assert_array_equal(exog_values, expected)


def test_create_predict_inputs_last_window_slicing():
    """
    Test _create_predict_inputs slices last_window to window_size.
    """
    y = pd.Series(np.arange(20), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    # Pass a longer last_window
    last_window = pd.Series(np.arange(10), name="y")
    last_window_values, _, _, _ = forecaster._create_predict_inputs(
        steps=5, last_window=last_window
    )

    # Should only take last 3 values
    np.testing.assert_array_equal(last_window_values, np.array([7, 8, 9]))


def test_create_predict_inputs_differentiation():
    """
    Test _create_predict_inputs with differentiation.
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    forecaster.fit(y=y)

    # last_window: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # window_size is 3 + 1 = 4
    # last 4 values: [6, 7, 8, 9]
    # diff: [1, 1, 1]

    last_window_values, _, _, _ = forecaster._create_predict_inputs(steps=2)

    assert len(last_window_values) == 4
    np.testing.assert_array_equal(last_window_values, np.array([np.nan, 1.0, 1.0, 1.0]))
