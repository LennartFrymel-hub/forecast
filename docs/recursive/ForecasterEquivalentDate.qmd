# ForecasterEquivalentDate

## Introduction

In the realm of time series forecasting, especially within safety-critical systems, robustness and interpretability are paramount. While complex machine learning models like Gradient Boosting or Neural Networks often provide superior accuracy, they can be "black boxes" that behave unpredictably when encountering data distributions different from their training set.

`ForecasterEquivalentDate` is a specialized forecaster designed to provide a highly interpretable, reliable baseline. It operates on a simple yet powerful principle: history repeats itself. By identifying "equivalent dates" in the past (e.g., the same day last week or the same hour yesterday), it generates forecasts that are naturally grounded in observed reality.

This class is part of the `spotforecast2_safe` package, emphasizing its role in building resilient forecasting pipelines where a simple, verifiable fallback is often more valuable than a fragile, complex one.

---

## Core Concepts

The `ForecasterEquivalentDate` relies on three primary parameters to define its behavior:

1.  Offset: Defines how far back in time to look for the "equivalent" data.
    -   Integer: Represents a fixed number of steps (e.g., `offset=24` for hourly data to look at the same hour yesterday).
    -   Pandas DateOffset: Robustly handles calendar logic (e.g., `pd.offsets.Week(1)` to look at the same day/time last week, even accounting for variable frequencies or business days).
2.  n_offsets: The number of historical equivalent periods to consider. Instead of just looking at *one* equivalent date, you can look at the last *N* occurrences (e.g., the last 4 Mondays).
3.  agg_func: The function used to aggregate values if `n_offsets > 1`. Common choices include `np.mean`, `np.median`, `np.max`, or `np.min`.

---

## Safety-Critical Fallback Mechanism

In safety-critical environments (such as energy grid management, medical monitoring, or autonomous industrial processes), a failure in the primary forecasting model can have severe consequences. `ForecasterEquivalentDate` serves as an ideal fallback mechanism due to the following properties:

*   Low Complexity: It has virtually no "model risk." The forecast is a direct reflection of historical data.
*   High Interpretability: Every prediction can be traced back to specific historical dates and values. If a prediction looks wrong, a human operator can immediately see which historical dates were used.
*   Resilience to Outliers: By using `n_offsets > 1` with `agg_func=np.median`, the forecaster becomes extremely robust to historical anomalies.
*   No Training Required: Unlike ML models that require a complex `fit` process on large datasets, this forecaster essentially "remembers" the recent window, making it ready to use almost instantly.
*   Probabilistic Confidence: Through its integration with Conformal Prediction, it provides statistically sound uncertainty intervals, allowing the system to quantify the risk of the baseline forecast.

---

## Functional Examples

### 1. Basic Baseline (Daily Seasonality)

This example demonstrates the simplest usage: using the value from exactly one week ago as the forecast for today.

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate

# 1. Prepare a synthetic daily series with a weekly pattern
index = pd.date_range(start='2023-01-01', periods=21, freq='D')
# Sine wave with 7-day period + noise
data = 10 + 5 * np.sin(2 * np.pi * np.arange(21) / 7) + np.random.normal(0, 0.5, 21)
y = pd.Series(data, index=index, name="energy_consumption")

# 2. Define the forecaster: look back 7 days
forecaster = ForecasterEquivalentDate(offset=7)

# 3. "Fit" the forecaster (stores the necessary history)
forecaster.fit(y=y)

# 4. Predict the next 3 days
predictions = forecaster.predict(steps=3)

print("Predictions for next 3 days:")
print(predictions)

# Verify that the first prediction matches the value from 7 days ago
last_known_equivalent = y.iloc[-7]
print(f"\nValue 7 days ago: {last_known_equivalent:.4f}")
print(f"Prediction for tomorrow: {predictions.iloc[0]:.4f}")
```

### 2. Aggregated Multi-Offset (Robust Weekly Pattern)

For increased safety, we can average the values from the last 3 equivalent dates. This smooths out individual day anomalies.

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate

# Prepare data
index = pd.date_range(start='2023-01-01', periods=35, freq='D')
data = np.arange(35) % 7 # Repeating 0-6 pattern
y = pd.Series(data, index=index)

# Aggregated forecaster: Mean of the last 3 weeks (offset 7, 14, 21)
forecaster = ForecasterEquivalentDate(
    offset=7, 
    n_offsets=3, 
    agg_func=np.mean
)

forecaster.fit(y=y)
predictions = forecaster.predict(steps=7)

print("Forecasted week (should be 0-6):")
print(predictions.values)
```

### 3. Using Calendar Offsets (Handling Business Logic)

In many systems, "one week ago" isn't a fixed number of steps but a calendar concept. `ForecasterEquivalentDate` supports `pd.DateOffset` for these scenarios.

```python
import pandas as pd
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate

# Hourly data with Business Day logic
index = pd.date_range(start='2023-01-01', periods=100, freq='h')
y = pd.Series(range(100), index=index)

# Look back exactly one business week
# This is more robust than offset=168 (24*7) if the data has gaps or irregular holidays
forecaster = ForecasterEquivalentDate(offset=pd.offsets.BusinessDay(5))

forecaster.fit(y=y)
predictions = forecaster.predict(steps=5)

print("Predictions using BusinessDay offset:")
print(predictions)
```

### 4. Advanced: Probabilistic Intervals with Binned Residuals

In safety-critical systems, point forecasts are rarely enough. We need to know how much to trust the fallback.

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate

# Generate data with increasing noise (heteroscedasticity)
n = 200
index = pd.date_range(start='2023-01-01', periods=n, freq='D')
noise = np.random.normal(0, np.linspace(0.1, 2.0, n), n)
y = pd.Series(10 + noise, index=index)

# Forecaster with Residual Binning
# This allows the uncertainty intervals to vary depending on the predicted value
forecaster = ForecasterEquivalentDate(
    offset=7,
    n_offsets=1,
    binner_kwargs={'n_bins': 5}
)

# Store in-sample residuals to calibrate the intervals
forecaster.fit(y=y, store_in_sample_residuals=True)

# Predict intervals (90% confidence)
intervals = forecaster.predict_interval(
    steps=7, 
    interval=[5, 95] 
)

print("Predictions with Confidence Intervals:")
print(intervals)
```

---

## Expert Reference: Internal Mechanics

### Residual Binning and Conformal Prediction
The `ForecasterEquivalentDate` implements a binned approach to conformal prediction. When `fit(store_in_sample_residuals=True)` is called:
1.  The forecaster calculates historical "forecasts" for the training period.
2.  Residuals ($y_{true} - y_{pred}$) are calculated.
3.  Predicted values are split into bins (quantiles) using the `binner`.
4.  Residuals are associated with their corresponding bin based on the predicted value.
5.  When `predict_interval()` is called, the forecaster uses the distribution of residuals from the relevant bin to calculate the bounds. This allows for narrower intervals when the model is historically more accurate and wider intervals when it is less certain.

### Limitations
While excellent as a fallback, this forecaster does not:
*   Account for exogenous variables (e.g., weather, holidays).
*   Capture trends (it assumes the future level will be similar to the past level).
*   Modify its behavior based on recent forecast errors (it's not adaptive in the short term).

For these reasons, it should always be used in tandem with more sophisticated models or as a "sanity check" boundary.
