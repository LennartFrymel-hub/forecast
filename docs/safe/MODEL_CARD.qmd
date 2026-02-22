# Model/Method Card: spotforecast2-safe

## 1. System Details

- Name: spotforecast2-safe
- Version: 0.8.0-rc.1
- Type: Deterministic library for time series transformation and feature generation (Preprocessing).
- License: AGPL-3.0-or-later
- Developers: bartzbeielstein
- Repository: https://github.com/sequential-parameter-optimization/spotforecast2-safe
- CPE Identifier (Wildcard): cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:*:*:*:*:*:*:*:*
- CPE Identifier (Current Release): cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:0.8.0-rc.1:*:*:*:*:*:*:*
- Core Dependencies: numpy, pandas (Minimal Dependency Footprint).
- Prohibited Dependencies: plotly, matplotlib, spotoptim, optuna, torch, tensorflow.

## 2. Intended Use
### Primary Use Cases

- **Safety-Critical Forecasting Systems**: Preparation of time series data for regression models in environments requiring auditability (e.g., energy supply, finance).
- **Embedded Systems / Edge AI**: Use in resource-constrained environments where large ML frameworks cannot be installed.
- **Reproducible Research**: Ensuring exact mathematical reproducibility of N-to-1 transformations without hidden stochastics.

### Out-of-Scope

- **Interactive Visualization**: The package deliberately contains no plotting functions to minimize the attack surface.
- **Automated Hyperparameter Tuning**: Optimization (e.g., via spotoptim or Optuna) must take place outside the "Safe Environment".
- **Silent Data Cleaning**: The package does not perform "silent" data imputation. Missing values (NaNs) lead to explicit errors (Fail-Safe), not estimations.

## 3. Algorithm & Logic

The core task `task_n_to_1` implements a deterministic sliding-window transformation.

### Mathematical Description

Given a univariate time series $X = \{x_1, x_2, ..., x_T\}$, the system transforms this into a feature matrix $X_{feat}$ and a target vector $y$ based on the window size $w$ (lags):

$$X_{row, t} = [x_{t-w}, x_{t-w+1}, ..., x_{t-1}] \rightarrow y_t = x_t$$

### Design Objectives

- **Deterministic**: The implementation strives to ensure that the same input always generates the exact same bit-level output.
- **Leakage-Free**: The implementation aims to ensure that the target value $y_t$ is never contained within the input vector $X_{row, t}$.

## 4. Performance & Robustness (Design Goals)

In the absence of "Accuracy" (as no model is trained), the following software metrics are design goals intended to support compliance with standards like IEC 61508 / EU AI Act. **Users must verify these properties**:

### Fail-Safe Behavior

- **Input**: DataFrame with `NaN` or `Inf`.
- **Behavior**: Throws an explicit `ValueError`. No silent processing (Silent Failure).

### Input Validation
- **Strict Checks**: Type hinting and runtime checks for `pd.DataFrame` and `np.ndarray`.

### Cybersecurity Footprint

- **Minimal CVE Surface**: By avoiding complex dependencies (like PyTorch or web server components), the Common Vulnerabilities and Exposures (CVE) attack surface is minimized.

## 5. Compliance & EU AI Act Support

This package is designed to support the development of high-risk AI systems according to the EU AI Act. However, **this package itself is not certified**.

- **Transparency (Art. 13)**: We strive for a fully transparent ("White Box") code structure.
- **Accuracy & Robustness (Art. 15)**: The transformations are designed to be mathematically provable and reproducible, but formal verification is the user's responsibility.
- **Data Governance (Art. 10)**: The package aims to enforce clean data formats by rejecting "dirty" data, assisting in data governance efforts.

## 6. Caveats & Limitations

- **No Extrapolation**: The package prepares data; it does not predict by itself. The quality of the forecast depends on the downstream regressor (e.g., `scikit-learn` LinearRegression).
- **Memory Requirements**: Creating the Lag matrix (N-to-1) can be memory-intensive for extremely large time series ($T > 10^7$) as data is duplicated.

## 7. How to Audit

For auditors who need to validate this package:

1. Check `pyproject.toml` to confirm the absence of unsafe libraries.
2. Run `pytest tests/` to verify the functional correctness of the matrix transformation.
3. Run `pytest tests/test_cpe.py` to verify CPE identifier generation for compliance and SBOM (Software Bill of Materials) tracking.
4. Reference the CPE Identifier from Section 1 to include this package in vulnerability tracking systems and supply chain disclosure documents.
5. Consult the get_cpe_identifier function in `src/spotforecast2_safe/utils/cpe.py` for CPE generation in automated workflows.


## 8. Disclaimer & Liability

**LIMITATION OF LIABILITY**: While this library is designed with safety principles and deterministic logic in mind, it is provided "AS IS" without any warranties. The developers and contributors assume **NO LIABILITY** for any direct or indirect damages, system failures, or financial losses resulting from the use of this software. 

It is the sole responsibility of the system integrator to perform a full system-level safety validation (e.g., as per ISO 26262, IEC 61508, or the EU AI Act) before deploying this software in a production or safety-critical environment.

