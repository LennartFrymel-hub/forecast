# Model/Method Card: spotforecast2-safe

This card follows the Hugging Face Model Card Guidebook taxonomy [@ozon22a].

## 1. Model Details

- **Name**: spotforecast2-safe
- **Version**: 1.0.1
- **Type**: Deterministic library for time series transformation and feature generation (preprocessing + recursive forecasting wrappers).
- **Developed by**: Thomas Bartz-Beielstein. ORCID: [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158).
- **Shared by**: `sequential-parameter-optimization` GitHub organization.
- **Language**: Python 3.13+.
- **License**: AGPL-3.0-or-later.
- **Core dependencies**: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `numba`, `pyarrow`, `requests`, `feature-engine`, `holidays`, `astral`, `tqdm` (minimal dependency footprint).
- **Prohibited dependencies**: `plotly`, `matplotlib`, `spotoptim`, `optuna`, `torch`, `tensorflow`.
- **Repository**: <https://github.com/sequential-parameter-optimization/spotforecast2-safe>
- **Technical report**: `bart26h/index.qmd` (shipped in-tree).
- **CPE Identifier (Wildcard)**: `cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:*:*:*:*:*:*:*:*`
- **CPE Identifier (Current Release)**: `cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:1.0.1:*:*:*:*:*:*:*`

## 2. Uses

### Direct Use

- **Safety-critical forecasting pipelines**: Preparation of time series data for regression models in auditable environments (energy supply, finance, industrial monitoring).
- **Embedded / edge AI**: Runs in resource-constrained environments where heavyweight ML frameworks are not available.
- **Reproducible research**: Bit-level reproducible N-to-1 lag transformations, no hidden stochastics.

### Downstream Use

- Feeding the generated feature matrices into `scikit-learn` regressors, `lightgbm`, or `xgboost` via the bundled `ForecasterRecursiveLGBM` / `ForecasterRecursiveXGB` wrappers.
- Building custom recursive multi-step forecasters on top of the `ForecasterRecursiveModel` base class in `spotforecast2_safe.manager.models`.

### Out-of-Scope Use

- **Interactive visualization**: Deliberately no plotting code ships in this package — no Plotly, no Matplotlib.
- **Automated hyperparameter tuning**: Must run outside the safe environment (e.g., in a separate `spotforecast2` / `spotoptim` / Optuna workflow).
- **Silent data cleaning**: `NaN` / `Inf` values raise `ValueError`; the package does not silently impute.

## 3. Bias, Risks, and Limitations

- **Downstream regressor drift**: `spotforecast2-safe` performs deterministic feature engineering, not forecasting by itself. Accuracy is bounded by the regressor and the training data — concept drift, seasonality shifts, or regime changes in the downstream model will silently degrade forecasts.
- **Lag-feature leakage when bypassing `ExogBuilder`**: Users who construct lag or calendar features outside the provided builders risk leaking target values into the feature row for timestamp *t*. The bundled `ExogBuilder` / `task_n_to_1` paths are leakage-free by construction; hand-rolled pipelines are not.
- **Multi-threaded inference determinism**: The bit-level reproducibility guarantee assumes single-threaded execution or an explicitly pinned `n_jobs=1` on the downstream regressor. Thread-pool schedulers (e.g., LightGBM's default `n_jobs=-1`) may reorder floating-point reductions.
- **Memory for large series**: Creating the lag matrix duplicates the input (T × w). For T > 10⁷ this can exhaust memory; chunking is the caller's responsibility.
- **Living standards**: IEC 61508 / ISO 26262 / EU AI Act references reflect the text as of 2026-04-19. Users must track subsequent amendments themselves.

### Recommendations

- Validate every new deployment against historical ground truth before switching traffic.
- Always use `ExogBuilder` or `task_n_to_1` for feature construction. Do not hand-roll lag matrices.
- Pin `n_jobs=1` (or equivalent) on the downstream regressor when bit-level reproducibility is required.
- For T > 10⁷, process the series in windowed chunks and re-aggregate downstream.

## 4. How to Get Started

```bash
pip install spotforecast2-safe
```

```python
from lightgbm import LGBMRegressor
from spotforecast2_safe import ForecasterRecursiveLGBM, ConfigEntsoe

config = ConfigEntsoe()
forecaster = ForecasterRecursiveLGBM(regressor=LGBMRegressor(n_jobs=1), config=config)
forecaster.fit(y=y_train, exog=exog_train)
predictions = forecaster.predict(steps=config.forecast_horizon, exog=exog_future)
```

A full end-to-end reference workflow (baseline + covariates + LightGBM vs. ground truth) is registered as a console script:

```bash
uv run spotforecast-safe-demo
```

The demo source lives in `src/spotforecast2_safe/tasks/task_safe_demo.py`.

## 5. Technical Specifications

The core task `task_n_to_1` implements a deterministic sliding-window transformation.

### Mathematical Description

Given a univariate time series $X = \{x_1, x_2, \ldots, x_T\}$ and a window size $w$ (lags), the system produces a feature matrix $X_{feat}$ and target vector $y$ via

$$X_{row, t} = [x_{t-w}, x_{t-w+1}, \ldots, x_{t-1}] \rightarrow y_t = x_t.$$

### Design Objectives

- **Deterministic**: Same input → same bit-level output.
- **Leakage-free**: The target $y_t$ is never contained in the corresponding input vector $X_{row, t}$.
- **Fail-safe**: Invalid input raises an explicit exception; the library never silently repairs bad data.

### Architecture (layered)

`forecaster/` (low-level estimator wrappers) → `preprocessing/` (deterministic transformers: `ExogBuilder`, `RepeatingBasisFunction`, `QuantileBinner`, `TimeSeriesDifferentiator`) → `model_selection/` (time-aware CV: `TimeSeriesFold`, `OneStepAheadFold`, `backtesting_forecaster`) → `manager/` (orchestration: `ForecasterRecursiveLGBM`, `ForecasterRecursiveXGB`, `ConfigEntsoe`) → `processing/` (high-level pipelines) → `tasks/` (console-script entry points).

## 6. Evaluation

No training step runs inside `spotforecast2-safe` itself, so classical "accuracy" metrics do not apply to the library. The evaluation targets are software-quality metrics that support compliance with IEC 61508 / EU AI Act.

### Testing Data

- Docstring examples in `src/` (executed via `tests/test_docstring_examples_*.py`).
- Unit fixtures and integration data in `tests/` and `src/spotforecast2_safe/datasets/csv/`.
- The bundled ENTSO-E demo set (`DemoConfig.data_path`) for end-to-end task tests.

### Factors

- Input dtype (numpy vs. pandas, int vs. float vs. datetime index).
- Presence of `NaN` / `Inf` (must fail loudly).
- Series length (smoke-tested up to 10⁶ rows).
- Lag-window size and forecast horizon.

### Metrics

- Functional correctness of the lag-matrix transformation (unit tests).
- CPE identifier generation (`tests/test_cpe.py`).
- Determinism: identical input must yield identical output bytes.
- Coverage: ≥ 80 % line coverage on new code (see `CONTRIBUTING.md`).

### Results

- **Fail-safe behavior**: DataFrames containing `NaN` or `Inf` raise `ValueError`. Public loaders (`load_timeseries`, `load_timeseries_forecast`, `WeatherService.get_dataframe`) refuse to return silently-imputed values by default. Callers must opt in to legacy forward/back-fill via `on_missing='ffill_bfill'` (loaders) or `fill_missing=True` (weather client) to restore pre-1.0 behavior.
- **Input validation**: Strict type hinting plus runtime checks for `pd.DataFrame` and `np.ndarray`.
- **Cybersecurity footprint**: The prohibited-dependency policy minimizes the Common Vulnerabilities and Exposures (CVE) attack surface. No web server, no deep-learning runtime, no plotting backend.

## 7. Environmental Impact

The library itself performs no training and requires no GPU. Runtime cost is dominated by (a) NumPy/Pandas vector ops during feature engineering and (b) whatever downstream regressor the caller passes in. A typical `ForecasterRecursiveLGBM` fit on a 10⁵-row series with 168 lags completes in seconds on a single commodity CPU core; the per-inference carbon cost of the safe layer is effectively negligible next to the regressor's own cost. No pretrained weights are shipped, so there are no embedded-training emissions to report.

## 8. Compliance & EU AI Act Support

This package is designed to support the development of high-risk AI systems according to the EU AI Act. **The package itself is not certified**; the system integrator owns full-system certification.

- **Art. 10 (Data Governance)**: The package rejects dirty data (`NaN` / `Inf`) by default, supporting governance requirements for training and inference data quality.
- **Art. 11 (Technical Documentation)**: This card plus the `bart26h/` technical report form the technical-documentation baseline. The CPE identifier in §1 feeds SBOM and vulnerability-tracking pipelines.
- **Art. 12 (Automatic Logging)**: `spotforecast2_safe.manager.logger` provides a dual-handler (console + persistent file) logger; tasks emit timestamped logs under `~/spotforecast2_safe_models/logs/` for audit retention.
- **Art. 13 (Transparency)**: Code is "white-box" — no compiled inference kernels, no opaque model weights.
- **Art. 15 (Accuracy & Robustness)**: Transformations are mathematically provable and bit-level reproducible. Formal verification remains the user's responsibility.

See the compliance table in the accompanying technical report (`bart26h/index.qmd`, section *Compliance Mapping*) for the authoritative mapping to IEC 61508, ISO 26262, ISA/IEC 62443, and the EU AI Act articles.

## 9. Glossary

- **AGPL** — Affero General Public License; copyleft license requiring source availability even for network-deployed use.
- **ASIL** — Automotive Safety Integrity Level (ISO 26262).
- **CPE** — Common Platform Enumeration; standardized identifier for software products in vulnerability-tracking systems.
- **CVE** — Common Vulnerabilities and Exposures; public catalogue of known software vulnerabilities.
- **EU AI Act** — Regulation (EU) 2024/1689 on artificial intelligence, in force since 2024-08-01.
- **IEC 61508** — International Electrotechnical Commission standard for functional safety of electrical / electronic / programmable electronic safety-related systems.
- **ISA/IEC 62443** — Industrial automation and control systems security standard series.
- **ISO 26262** — Road-vehicle functional-safety standard.
- **SBOM** — Software Bill of Materials; machine-readable inventory of a product's components.
- **SDL** — Security Development Lifecycle.
- **SIL** — Safety Integrity Level (IEC 61508).

## 10. Citation

```bibtex
@misc{spotforecast2safe,
  author       = {Bartz-Beielstein, Thomas},
  title        = {{spotforecast2-safe}: Safety-critical Subset of {spotforecast2}},
  year         = {2026},
  howpublished = {\url{https://github.com/sequential-parameter-optimization/spotforecast2-safe}},
  note         = {AGPL-3.0-or-later}
}
```

**APA**: Bartz-Beielstein, T. (2026). *spotforecast2-safe: Safety-critical subset of spotforecast2* (Version 1.0.1) [Computer software]. https://github.com/sequential-parameter-optimization/spotforecast2-safe

The accompanying technical report (`bart26h/index.qmd`) is the long-form reference for design rationale, compliance mapping, and evaluation protocol.

## 11. Model Card Authors & Contact

- Thomas Bartz-Beielstein — ORCID [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158) — `bartzbeielstein@gmail.com`

This card follows the Hugging Face Model Card Guidebook taxonomy [@ozon22a].

## 12. How to Audit

For auditors who need to validate this package:

1. Check `pyproject.toml` to confirm the absence of prohibited libraries (`plotly`, `matplotlib`, `spotoptim`, `optuna`, `torch`, `tensorflow`).
2. Run `uv run pytest tests/` to verify functional correctness of the matrix transformation and the full test suite.
3. Run `uv run pytest tests/test_cpe.py` to verify CPE identifier generation for compliance and SBOM tracking.
4. Reference the CPE identifiers from §1 in vulnerability tracking systems and supply-chain disclosure documents.
5. Consult `get_cpe_identifier` in `src/spotforecast2_safe/utils/cpe.py` for CPE generation in automated workflows.
6. Run `uv run reuse lint` to confirm SPDX/REUSE licensing compliance.

## 13. Disclaimer & Liability

**LIMITATION OF LIABILITY**: While this library is designed with safety principles and deterministic logic in mind, it is provided "AS IS" without any warranties. The developers and contributors assume **NO LIABILITY** for any direct or indirect damages, system failures, or financial losses resulting from the use of this software.

It is the sole responsibility of the system integrator to perform a full system-level safety validation (e.g., as per ISO 26262, IEC 61508, or the EU AI Act) before deploying this software in a production or safety-critical environment.
