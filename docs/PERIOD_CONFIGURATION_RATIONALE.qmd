# Period Configuration and Feature Engineering Rationale

This document explains the rationale behind the selected period configurations and feature engineering choices in `spotforecast2-safe`, specifically regarding the encoding of cyclical time features using Radial Basis Functions (RBF).

## Background: Cyclical Feature Encoding

For time series forecasting, encoding cyclical features like "hour of day" or "day of week" is crucial. Standard methods like one-hot encoding or integer encoding have limitations:

- One-Hot Encoding: Explodes dimensionality (e.g., 24 columns for hours, 365 for days of year).
- Integer Encoding: Creates artificial ordinal relationships (e.g., hour 23 is "far" from hour 0).
- Sine-Cosine Encoding: Commonly used, but can introduce high-frequency noise and requires two features per cycle.

We use Repeating Basis Functions (RBF), which effectively project the cyclic feature onto a set of smooth, periodic functions. This allows us to control the resolution (detail) and smoothness of the encoding by adjusting the number of basis functions (`n_periods`).

## Configuration Rationale

The `ConfigEntsoe` class provides default configurations optimized for energy demand forecasting. The choice of `n_periods` relative to the cycle length (e.g., 24 hours) determines the resolution of the encoding.

### 1. Daily Cycle (24 Hours)

- Configuration: `n_periods=12`
- Ratio: 2:1 (24 hours / 12 functions)
- Resolution: ~2 hours
- Rationale: 

  - Hourly data often exhibits noise. A 1:1 ratio (24 basis functions) would provide perfect hourly resolution but might overfit to noise.
  - Reducing `n_periods` to 12 provides a natural smoothing effect, effectively capturing the broader daily demand curve (morning peak, evening peak) while filtering out high-frequency hourly fluctuations.
  - This also reduces feature dimensionality by 50% compared to one-hot encoding, improving model efficiency.

### 2. Weekly Cycle (168 Hours / 7 Days)

- Configuration: Typically `n_periods=7` (or similar for weekly patterns)
- Ratio: 1:1 on a daily scale
- Rationale: 
  - Daily patterns vary significantly between weekdays and weekends.
  - A resolution capturing each day of the week is essential to distinguish between Mondays (start of work week), mid-week, and weekends.

### 3. Yearly Cycle (365 Days)

- Configuration: `n_periods=12`
- Ratio: ~30:1 (365 days / 12 functions)
- Resolution: ~1 month
- Rationale: 
  - The yearly cycle (seasonality) is a low-frequency pattern driven by temperature and daylight.
  - We do not need daily resolution for seasonality. A monthly resolution is sufficient to capture the broad winter-summer trends.
  - Using `n_periods=12` mirrors the months of the year, providing a smooth, continuous representation of seasonality without overfitting to specific day-of-year quirks (which are better handled by holiday features and lag features).

## Summary Table

| Period | N Periods | Range Size | Ratio | Resolution | Goal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Daily | 12 | 24 | 1:2 | 2 Hours | Smoothing hourly noise, reducing dimensionality. |
| Weekly | 7 | 7 | 1:1 | 1 Day | Distinguishing day-of-week patterns. |
| Yearly | 12 | 365 | 1:30 | 1 Month | Capturing broad seasonal trends. |

This configuration balances the need for detail with the benefits of dimensionality reduction and noise smoothing, leading to more robust forecasting models.
