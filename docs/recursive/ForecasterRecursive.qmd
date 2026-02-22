# ForecasterRecursive

## Introduction

In modern time series forecasting, the recursive autoregressive approach is a cornerstone strategy. `ForecasterRecursive` is a powerful and flexible class in the `spotforecast2_safe` package that transforms any standard scikit-learn regressor into a multi-step time series forecaster.

The core mechanism of `ForecasterRecursive` involves learning a mapping from past observations (lags) and optional exogenous features to the next value in the series. For multi-step forecasting, it uses its own predictions as inputs for subsequent steps—an approach known as the recursive strategy.

This class is designed with safety and reliability in mind, providing extensive validation, support for feature engineering via window functions, and integrated probabilistic forecasting.

---

## Core Concepts

The `ForecasterRecursive` operates through several key components:

1.  Estimator: Any object compatible with the scikit-learn API (e.g., `LinearRegression`, `RandomForestRegressor`, `XGBRegressor`).
2.  Lags: Specific past time steps used as predictors (e.g., `lags=7` uses the last 7 values; `lags=[1, 7, 14]` uses specific seasonal lags).
3.  Window Features: Automated computation of statistics over rolling windows (e.g., rolling mean, rolling standard deviation) using the `RollingFeatures` class.
4.  Recursive Strategy: During prediction of step $t+k$, the model uses the predicted values from steps $t+1, \dots, t+k-1$ as if they were real historical data.
5.  Exogenous Variables: Support for external factors that influence the target variable (e.g., temperature influencing energy demand).

---

## Safety-Critical Design Patterns

`ForecasterRecursive` includes several features specifically tailored for safety-critical environments:

*   Comprehensive Validation: Strict checks on input data types, indices, and frequencies to prevent "silent failures" or misaligned data.
*   Resilience to Outliers: By using robust estimators like RandomForestRegressor, the forecaster can handle anomalies in the training data.
*   No Training Required: While the underlying estimator needs fitting, the recursive logic is ready to use as soon as the model is trained.
*   Consistent Interface: It follows the familiar scikit-learn pattern of fit and predict.

---

## Functional Examples

### 1. Standard Autoregressive Forecast

A basic example using `LinearRegression` with the last 7 days as predictors.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

# 1. Create a synthetic daily series with a trend and seasonality
index = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.linspace(10, 20, 100) + 5 * np.sin(2 * np.pi * np.arange(100) / 7)
y = pd.Series(data, index=index, name="target")

# 2. Initialize with 7 lags
forecaster = ForecasterRecursive(
    estimator=LinearRegression(),
    lags=7
)

# 3. Fit and Predict
forecaster.fit(y=y)
predictions = forecaster.predict(steps=14)

print("Forecast for the next 14 days:")
print(predictions.head())
```

### 2. Multi-Feature Forecasting (Lags + Window Statistics)

Combining raw lags with rolling means provides the model with both local and global context.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures

# Data preparation
index = pd.date_range(start='2023-01-01', periods=200, freq='h')
y = pd.Series(np.random.normal(50, 5, 200), index=index, name="sensor_reading")

# Forecaster with Lags and Rolling Mean
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=50, random_state=123),
    lags=24,
    window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=24)
)

forecaster.fit(y=y)
predictions = forecaster.predict(steps=12)

print("Predictions with window features:")
print(predictions)
```

### 3. Handling Exogenous Variables

Forecasting energy demand using temperature as an external factor.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

# Target and Exogenous data
index = pd.date_range(start='2023-01-01', periods=100, freq='D')
y = pd.Series(np.random.normal(100, 10, 100), index=index, name="demand")
exog = pd.DataFrame({'temp': np.random.normal(20, 5, 100)}, index=index)

# Define forecaster
forecaster = ForecasterRecursive(estimator=Ridge(), lags=7)

# Fit with exog
forecaster.fit(y=y, exog=exog)

# Future exog values must be known for the prediction horizon
exog_future = pd.DataFrame(
    {'temp': [22, 21, 23]}, 
    index=pd.date_range(start='2023-04-11', periods=3, freq='D')
)

predictions = forecaster.predict(steps=3, exog=exog_future)
print("Demand forecast with temp as exog:")
print(predictions)
```


---

## Expert Reference: Internal Mechanics

### Feature Matrix Construction
`ForecasterRecursive` internally transforms the time series into a supervised learning matrix. If `lags=[1, 2]` is used, the training set $X$ at time $t$ will contain the columns $[y_{t-1}, y_{t-2}]$. If exogenous variables are provided, they are concatenated to this matrix.

### Robustness and Interpretability
The `ForecasterRecursive` transforms time series data into a supervised structure. If `lags=[1, 2]` is used, the training set X at time t will contain columns [y_t-1, y_t-2]. This explicit feature engineering makes it easy to understand what the model is looking at.

### Differentiation and Reconstruction
When `differentiation > 0` is set:
1.  The training series $y$ is differenced $n$ times.
2.  The model is trained to predict the differenced values.
3.  During prediction, the model generates differenced forecasts.
4.  These forecasts are automatically integrated (undifferenced) using the `TimeSeriesDifferentiator` and the `last_window_` to return predictions in the original scale. This process is seamless to the user but critical for handling trends.

### 4. Probabilistic Forecasting (Prediction Intervals)

`ForecasterRecursive` supports probabilistic forecasting, allowing you to estimate prediction intervals. This is crucial for safety-critical applications where quantifying uncertainty is as important as the point forecast itself.

Two methods are available:
*   Bootstrapping: Resamples residuals from the training phase to generate a distribution of possible future paths.
*   Conformal Prediction: Uses a calibration dataset (out-of-sample residuals) to guarantee a statistical coverage rate (e.g., 95%).

#### Bootstrapping Example

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

# 1. Generate synthetic data with noise
np.random.seed(123)
steps = 100
t = np.arange(steps)
y = pd.Series(
    data=0.5 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1.5, steps),
    index=pd.date_range(start='2023-01-01', periods=steps, freq='D'),
    name='y'
)

# 2. Initialize and Fit
forecaster = ForecasterRecursive(
    estimator=LinearRegression(),
    lags=12
)
forecaster.fit(y=y, store_in_sample_residuals=True)

# 3. Predict with Intervals (Bootstrapping)
# We predict the next 10 days with a 95% confidence interval (2.5% to 97.5%)
results = forecaster.predict_interval(
    steps=10,
    method='bootstrapping',
    interval=[5, 95],
    n_boot=250,
    random_state=123
)

print("Bootstrapping Intervals:")
print(results.head())
```

#### Conformal Prediction Example

Conformal prediction often requires out-of-sample residuals for calibration to ensure the coverage guarantee holds.

```python
# ... (Assuming same setup as above)

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
    method='conformal',
    interval=0.95,  # 0.95 means 95% coverage
    use_in_sample_residuals=False # Use the calibration residuals we just set
)

print("\nConformal Prediction Intervals:")
print(results_conformal.head())
```
