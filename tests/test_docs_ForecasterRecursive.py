import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_docs_probabilistic_forecasting():
    # 1. Generate synthetic data with noise
    np.random.seed(123)
    steps = 100
    t = np.arange(steps)
    y = pd.Series(
        data=0.5 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1.5, steps),
        index=pd.date_range(start="2023-01-01", periods=steps, freq="D"),
        name="y",
    )

    # 2. Initialize and Fit
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=12)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # 3. Predict with Intervals (Bootstrapping)
    # We predict the next 10 days with a 95% confidence interval (2.5% to 97.5%)
    results = forecaster.predict_interval(
        steps=10, method="bootstrapping", interval=[5, 95], n_boot=250, random_state=123
    )

    print("Bootstrapping Intervals:")
    print(results.head())

    assert isinstance(results, pd.DataFrame)
    assert results.shape == (10, 3)
    assert set(results.columns) == {"pred", "lower_bound", "upper_bound"}

    # Conformal Prediction Example

    # 1. Split data into training and calibration sets
    # We use the last 20 points for calibration (out-of-sample residuals)
    y_train = y.iloc[:-20]
    y_calibration = y.iloc[-20:]

    forecaster.fit(y=y_train)

    # 2. Compute out-of-sample residuals
    # This is a critical step for conformal prediction
    y_pred = forecaster.predict(steps=len(y_calibration))
    forecaster.set_out_sample_residuals(y_true=y_calibration, y_pred=y_pred)

    # 3. Predict with Intervals (Conformal)
    # We request a 95% confidence level (nominal coverage)
    results_conformal = forecaster.predict_interval(
        steps=10,
        method="conformal",
        interval=0.95,  # 0.95 means 95% coverage
        use_in_sample_residuals=False,  # Use the calibration residuals we just set
    )

    print("\nConformal Prediction Intervals:")
    print(results_conformal.head())

    assert isinstance(results_conformal, pd.DataFrame)
    assert results_conformal.shape == (10, 3)
    assert set(results_conformal.columns) == {"pred", "lower_bound", "upper_bound"}
