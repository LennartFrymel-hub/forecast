import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


def test_recursive_predict_output_type_and_shape():
    """
    Test _recursive_predict returns a numpy array with the correct shape.
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)

    last_window_values = y.iloc[-3:].to_numpy()
    steps = 5
    predictions = forecaster._recursive_predict(
        steps=steps, last_window_values=last_window_values
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (steps,)


def test_recursive_predict_linear_model_optimization():
    """
    Test _recursive_predict with a linear model (uses fast path).
    """
    # Simple linear relationship: y = lag1 + 1
    y = pd.Series(np.arange(10, dtype=float), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
    forecaster.fit(y=y)

    # Expected predictions: 10, 11, 12, 13, 14
    last_window_values = np.array([9.0])
    steps = 5
    predictions = forecaster._recursive_predict(
        steps=steps, last_window_values=last_window_values
    )

    expected = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    np.testing.assert_allclose(predictions, expected, atol=1e-10)


def test_recursive_predict_with_window_features():
    """
    Test _recursive_predict with window features.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    # lags=1, window_features (mean of 3)
    # y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # last_window_values = [7, 8, 9]
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=1,
        window_features=[RollingFeatures(stats="mean", window_sizes=3)],
    )
    forecaster.fit(y=y)

    last_window_values = np.array([7.0, 8.0, 9.0])
    steps = 2
    predictions = forecaster._recursive_predict(
        steps=steps, last_window_values=last_window_values
    )

    # Manual calculation:
    # step 1:
    # lag1 = 9
    # mean3 = (7+8+9)/3 = 8
    # Model was trained on:
    # y_pred | lag1 | mean3
    # 3      | 2    | (0+1+2)/3=1
    # 4      | 3    | (1+2+3)/3=2
    # Regression fit should be y = lag1 + 1 (or something similar depending on weights)
    # Since it's a perfect line, y = lag1 + 1 and mean3 is also correlated.
    # The optimized path uses intercepts and coefs.

    assert len(predictions) == 2
    assert not np.isnan(predictions).any()


def test_recursive_predict_with_exog():
    """
    Test _recursive_predict with exogenous variables.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    exog = pd.DataFrame({"exog1": np.arange(10, 20, dtype=float)}, index=y.index)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
    forecaster.fit(y=y, exog=exog)

    last_window_values = np.array([9.0])
    exog_values = np.array([[20.0], [21.0]])
    steps = 2
    predictions = forecaster._recursive_predict(
        steps=steps, last_window_values=last_window_values, exog_values=exog_values
    )

    assert len(predictions) == 2
    assert not np.isnan(predictions).any()
