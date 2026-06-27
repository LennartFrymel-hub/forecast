<!--
SPDX-FileCopyrightText: 2026 Lennart Frymel
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Model/Method Card: spotforecast2-safe
## Load Forecasting (Germany)

*Complete 13-section model card based on the uploaded ZIP repository and the provided project card structure*

| **Audit basis**     | **Result**                                                                                                                                                                                                              |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Repository package  | Korrekturmöglichkeit.zip / spotforecast2-safe                                                                                                                                                                           |
| Files inspected     | 947 non-.git repository files indexed; source code, scripts, tests, docs, configs, output CSVs and project metadata analysed. Binary caches, .git internals and generated pyc files were not used as semantic evidence. |
| Core source modules | src/spotforecast2_safe: data, downloader, forecaster, manager, preprocessing, processing, tasks, utils, weather                                                                                                         |
| Operational scripts | scripts/01 to 11, including live refresh, gap diagnosis, self-healing redownload, weather-cache validation and rolling-origin backtesting                                                                               |
| Testing footprint   | 255 files in tests/, 127 test\_\*.py files, approximately 1441 test functions                                                                                                                                           |

This card follows the structure of the project-specific model card with 13 numbered main sections. It combines the existing model-card content with the current operational Germany load-forecasting code contained in the uploaded ZIP repository.

# 1. Model Details

spotforecast2-safe is a deterministic, safety-oriented Python framework used here as a Germany load-forecasting ensemble for 24-hour day-ahead electrical demand prediction. The repository combines a reusable safe forecasting library with operational scripts for ENTSO-E load data, multi-city weather covariates, model training, live refresh, backtesting, post-processing and security audit support.

| **Field**              | **Project-specific value**                                                                                                  |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Name                   | spotforecast2-safe (Configuration: Germany Load Forecasting Ensemble)                                                       |
| Version                | 3.0.0, from pyproject.toml                                                                                                  |
| Type                   | Deterministic time-series feature-engineering and recursive forecasting framework for production-oriented load forecasting  |
| Language               | Python 3.13+                                                                                                                |
| License                | AGPL-3.0-or-later                                                                                                           |
| Repository             | https://github.com/sequential-parameter-optimization/spotforecast2-safe                                                     |
| Documentation          | https://sequential-parameter-optimization.github.io/spotforecast2-safe/                                                     |
| Developer / Maintainer | Thomas Bartz-Beielstein / sequential-parameter-optimization; project implementation: Germany load-forecasting configuration |
| Primary target region  | Germany, ENTSO-E control-area use case                                                                                      |
| Forecast horizon       | 24 hours, day-ahead; issued in scripts as hourly Europe/Berlin forecasts                                                    |
| Core dependencies      | astral, feature-engine, holidays, lightgbm, numba, pandas, pyarrow, requests, scikit-learn, tqdm                            |
| CPE wildcard           | cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:\*:\*:\*:\*:\*:\*:\*:\*                                      |
| CPE current release    | cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:3.0.0:\*:\*:\*:\*:\*:\*:\*                                   |

- The safe package emphasizes deterministic logic, explicit input validation, fail-safe behavior and a minimal dependency footprint.

- The Germany configuration extends the library into an operational pipeline using ENTSO-E actual load, weather covariates and LightGBM-based recursive forecasting.

- Production visualization dependencies are intentionally excluded from the core package; plotting is isolated in reporting scripts such as the single-day backtest visualizer.

# 2. Intended Use

The intended use is day-ahead forecasting of German electricity demand on an hourly grid. The pipeline is designed for transparent, reproducible and auditable forecasting rather than opaque end-to-end automated control.

| **Use category**                   | **Description**                                                                                                                       |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Direct operational use             | Generate a 24-hour German actual-load forecast for the next calendar day or next operational horizon.                                 |
| Energy utility planning            | Support portfolio planning, procurement, load-flow preparation, capacity planning and short-term demand assessment.                   |
| Critical infrastructure support    | Provide auditable forecasts for KRITIS-related energy workflows where traceability and validation are required.                       |
| Research and teaching              | Benchmark recursive forecasting, exogenous feature engineering, weather aggregation, backtesting and safety-oriented software design. |
| Compliance-oriented ML engineering | Demonstrate model-card documentation, CPE/SBOM support, supply-chain controls, reproducible tests and audit logs.                     |

- Human operators should interpret forecasts and compare them with baselines such as weekly persistence and ENTSO-E forecasts where available.

- Forecasts should be treated as decision-support signals, not as autonomous grid-control instructions.

- The system is suitable for hourly day-ahead planning, not for sub-minute or protection-level control.

# 3. Out-of-Scope Use

The project is not designed for every possible time-series or grid-control scenario. The following uses are explicitly outside the intended scope:

- Minute-level or real-time closed-loop control of electrical assets.

- Autonomous switching, dispatching, grid protection or safety-critical actuation without human oversight.

- Use outside Germany without retraining, recalibration and local holiday/weather validation.

- Silent data cleaning of corrupted input. Missing or invalid values must be handled explicitly; default safe paths raise errors or require opt-in imputation.

- Deep-learning, GPU-based, black-box forecasting where model internals and dependency footprint cannot be audited.

- Forecasting tasks where the target variable is not comparable to historical hourly electrical load or where data availability differs materially from the ENTSO-E/weather setup.

# 4. Data

The operational Germany use case combines historical load data, forecast-history comparison data, weather data, generated calendar features and derived post-processing tables. The central target is the ENTSO-E Actual Load time series.

| **Data source / file group**                   | **Observed repository content and role**                                                                                                                               |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_entsoe/interim/energy_load.csv            | Merged hourly/quarter-hourly load file with Time (UTC), Forecasted Load and Actual Load. This is the main target-data source for training and evaluation.              |
| data_entsoe/raw/entsoe_load\_\*.csv            | Raw ENTSO-E download chunks covering historical periods from 2022 onward and refresh windows into 2026.                                                                |
| Weather cache                                  | Multi-location Open-Meteo weather cache under weather_cache/de_multi; validated by outputs/weather_cache_validation.csv.                                               |
| outputs/backtest\_\*.csv                       | Generated evaluation artefacts: comparison tables, full-day metrics, horizon metrics, local-hour metrics, edge-group metrics, bias tables and peak-correction outputs. |
| src/spotforecast2_safe/datasets/csv/demo\*.csv | Small bundled demonstration datasets used by tests and examples.                                                                                                       |
| Holiday data                                   | Generated by holidays/fetch_holiday_data and aligned to hourly grids; default country DE and state NW in pipeline helpers.                                             |

| **Target columns** | **Meaning**                                                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Time (UTC)         | Timestamp, parsed to timezone-aware datetime and converted/validated as needed.                                                              |
| Actual Load        | Ground-truth target variable for German electricity demand.                                                                                  |
| Forecasted Load    | ENTSO-E forecast-history field, used for comparison and diagnostics; live script 07 explicitly disables ENTSO-E forecast as a model feature. |

- Data home is configurable via SPOTFORECAST2_DATA; cache/model storage is configurable via SPOTFORECAST2_CACHE.

- The live refresh script uses FULL_REFRESH_START = 2022-01-01 00:00 and an incremental refresh buffer of 7 days.

- Training for the Germany load pipeline starts at 2023-01-01 00:00 UTC in the live and rolling-origin scripts.

- A practical data-availability delay is modeled: the rolling-origin backtest uses DATA_AVAILABILITY_LAG_HOURS = 4, while live refresh applies ENTSOE_SAFETY_DELAY_HOURS = 3 for completed actual-load data.

- Missing values can be handled through explicit contracts: raise, ffill_bfill or passthrough. Safe default behavior rejects invalid NaN/Inf data unless the caller opts into a defined repair path.

# 5. Features

Feature engineering is the main value layer of the project. It transforms historical load and exogenous signals into a leakage-aware supervised forecasting matrix.

| **Feature family**          | **Implementation details**                                                                                                                                                                         |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Autoregressive lags         | Production lag set: 1, 2, 3; 22-26; 46-50; 166-170; 335-337; 503-505; 671-673 hours. This captures immediate, daily, two-day, weekly and multi-week periodicity around the exact seasonal offsets. |
| Rolling windows             | 72 h, 168 h and 720 h moving windows; used to capture short-term, weekly and 30-day trend structures.                                                                                              |
| Calendar features           | month, week, day_of_week and hour from a regular hourly DatetimeIndex.                                                                                                                             |
| Cyclical encoding           | month, week, day_of_week, hour, sunrise_hour and sunset_hour can be converted into sine/cosine representations.                                                                                    |
| Day/night features          | sunrise_hour, sunset_hour, daylight_hours and is_daylight via astral LocationInfo.                                                                                                                 |
| Holiday features            | DE/NW public-holiday indicator aligned to the forecast grid.                                                                                                                                       |
| Weather features            | temperature_2m, relative_humidity_2m, precipitation, rain, snowfall, weather_code, pressure_msl, surface_pressure, cloud cover and wind variables.                                                 |
| Weather aggregation         | Eight German cities are used with equal weights; the live configuration sets add_weighted_weather_average=True and keep_regional_weather_features=False.                                           |
| Interaction features        | Optional bilinear interactions between cyclical calendar features, weather-window columns and holiday indicators.                                                                                  |
| Additional exogenous inputs | The pipeline accepts additional numeric, fully-covered exogenous DataFrames and validates missing timestamps, duplicate columns and NaN entries.                                                   |

| **Weather location** | **Latitude** | **Longitude** | **Weight** |
|----------------------|--------------|---------------|------------|
| hamburg              | 53.5511      | 9.9937        | 1.0        |
| berlin               | 52.5200      | 13.4050       | 1.0        |
| koeln                | 50.9375      | 6.9603        | 1.0        |
| frankfurt            | 50.1109      | 8.6821        | 1.0        |
| stuttgart            | 48.7758      | 9.1829        | 1.0        |
| muenchen             | 48.1372      | 11.5756       | 1.0        |
| leipzig              | 51.3397      | 12.3731       | 1.0        |
| hannover             | 52.3759      | 9.7320        | 1.0        |

# 6. Models

The project separates deterministic data transformation from the downstream estimator. The Germany forecast uses recursive forecasting around LightGBM regressors and supports scikit-learn compatible estimators.

| **Model component**         | **Role**                                                                                                                                                                                                 |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ForecasterRecursive         | Core recursive multi-step forecasting wrapper. It creates lagged predictors and recursively feeds predictions forward.                                                                                   |
| ForecasterRecursiveLGBM     | Manager-level wrapper around LightGBM for recursive forecasting.                                                                                                                                         |
| ForecasterRecursiveXGB      | Alternative wrapper for XGBoost-style use cases where available.                                                                                                                                         |
| ForecasterEquivalentDate    | Baseline/equivalent-date forecaster and residual-binning support for prediction intervals.                                                                                                               |
| LightGBM tuned model        | Operational Germany scripts use deterministic LightGBM with n_estimators=1059, learning_rate about 0.0419, num_leaves=212, min_child_samples=54, subsample about 0.501 and colsample_bytree about 0.608. |
| Weekly persistence baseline | Evaluation comparator that shifts actual load by one week for benchmark comparison.                                                                                                                      |

- The live and backtest scripts set deterministic=True and force_col_wise=True for LightGBM. Some configurations use n_jobs=1 for reproducibility; the default estimator helper uses n_jobs=-1 for speed.

- Model persistence uses joblib-style saving/loading through manager.persistence. Cached models may be reused when force_train=False.

- The N-to-1 pipeline aggregates multiple forecast components using configurable weights; default weights are stored as a named DEFAULT_WEIGHTS constant.

- Prediction intervals and conformal-style behaviour are covered through residual storage, QuantileBinner and in-sample/out-of-sample residual binning tests.

# 7. Forecasting Workflow

The repository contains a complete operational workflow. The numbering below follows the actual script names in the ZIP.

| **Step** | **Script**                                                              | **Purpose**                                                                                                                                          |
|----------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1        | 01_download_de_load.py                                                  | Download ENTSO-E load data for Germany using ENTSOE_API_KEY and SPOTFORECAST2_DATA.                                                                  |
| 2        | 02_check_de_load.py                                                     | Check and load ENTSO-E Actual Load and Forecasted Load with explicit missing-data behaviour.                                                         |
| 3        | 03_forecast_de_load.py                                                  | Reference Germany forecast with multi-city weather, lags and LightGBM.                                                                               |
| 4        | 04_backtest_de_load.py                                                  | Backtest the configured approach using the same issue-time assumptions and lag/window setup.                                                         |
| 5        | 05_diagnose_entsoe_alignment.py                                         | Diagnose possible time alignment and phase-shift issues in ENTSO-E data.                                                                             |
| 6        | 06_check_entsoe_forecast_history.py                                     | Inspect ENTSO-E forecast-history coverage and consistency.                                                                                           |
| 7        | 06b_compare_de_vs_de_lu.py                                              | Compare Germany-only DE and Germany-Luxembourg DE_LU market-zone data.                                                                               |
| 8        | 06c_diagnose_entsoe_forecast_gaps.py                                    | Detect missing ENTSO-E forecast/load periods and cluster contiguous gaps.                                                                            |
| 9        | 06d_redownload_entsoe_forecast_gaps.py                                  | Redownload missing gap windows using API calls and safety buffers.                                                                                   |
| 10       | 07_refresh_and_forecast_next_24h.py                                     | Production-style refresh: update actual-load data, validate weather cache, train/load model, generate next 24-hour forecast and post-process output. |
| 11       | 08_check_weather_cache.py / 08_prepare_weather_cache.py                 | Prepare and validate the multi-city weather cache with future buffer.                                                                                |
| 12       | 09_rolling_origin_backtest_de_load.py                                   | Rolling-origin backtest over 90 days with daily folds and full metric export.                                                                        |
| 13       | 10_visualize_single_day_backtest.py / 11_build_postprocessing_tables.py | Visual reporting and construction of bias/peak post-processing tables.                                                                               |

```bash
python scripts/08_prepare_weather_cache.py
python scripts/01_download_de_load.py
python scripts/06c_diagnose_entsoe_forecast_gaps.py
python scripts/06d_redownload_entsoe_forecast_gaps.py
python scripts/07_refresh_and_forecast_next_24h.py
python scripts/09_rolling_origin_backtest_de_load.py
```

- The live forecast returns 24 output hours and writes comparison/forecast artefacts under outputs/.

- The model intentionally does not use ENTSO-E forecasted load as a live model feature in script 07; this reduces dependency on a competing forecast and keeps the model focused on actual load plus weather/calendar covariates.

- Post-processing can apply bias correction and peak correction with configurable strength 0.5 using tables built from previous backtest folds.

# 8. Evaluation

Evaluation combines software-quality tests with mathematical forecast validation. The project contains a large automated pytest suite and generated backtest outputs.

| **Evaluation layer**      | **Repository evidence**                                                                                                                            |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Unit/integration tests    | 255 files in tests/, including 127 test\_\*.py files and about 1441 test functions.                                                                |
| Forecasting tests         | Coverage for recursive fitting, prediction, intervals, quantiles, bootstrapping, residuals, feature importances and multiseries behaviour.         |
| Data tests                | ENTSO-E downloader/merge validation, fetch_data, weather data, holiday data, missing values, outliers and imputation weights.                      |
| Security/compliance tests | CPE generation, SBOM enrichment, prohibited-dependency guard and logger schema tests.                                                              |
| Backtesting               | Rolling-origin backtest script with FORECAST_HORIZON=24, BACKTEST_DAYS=90, FOLD_STEP_DAYS=1, EXCLUDE_LATEST_DAYS=7 and TRAIN_START=2023-01-01 UTC. |

| **Metric**    | **Definition in project use**                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------------------------|
| MAE           | Mean absolute error in MW.                                                                                          |
| RMSE          | Root mean squared error; penalizes larger errors.                                                                   |
| MAPE_percent  | Mean absolute percentage error, computed only where actual load is non-zero.                                        |
| sMAPE_percent | Symmetric percentage error using actual and predicted magnitudes.                                                   |
| Bias          | Mean signed error; positive indicates overprediction, negative indicates underprediction depending on model column. |
| MaxAbsError   | Largest absolute error in the evaluated horizon.                                                                    |
| UPR_percent   | Under-prediction rate used in segmented diagnostic output.                                                          |

| **Example output file**                      | **Model**          | **n_points** | **MAE** | **RMSE** | **MAPE %** | **Bias** | **MaxAbsError** |
|----------------------------------------------|--------------------|--------------|---------|----------|------------|----------|-----------------|
| outputs/backtest_metrics_24h.csv             | spotforecast       | 24           | 1084.98 | 1199.47  | 2.12       | -260.93  | 2180.09         |
| outputs/backtest_metrics_24h.csv             | entsoe_forecast    | 24           | 2426.99 | 2931.32  | 4.47       | 1673.57  | 5717.70         |
| outputs/backtest_metrics_24h_entsoe_exog.csv | weekly_persistence | 24           | 1000.82 | 1245.56  | 1.86       | -216.09  | 2432.02         |

- The numbers above are generated artefacts from the ZIP outputs and should be interpreted as example evaluation snapshots, not as a universal production guarantee.

- Backtests also export diagnostics by horizon, local hour and edge group, making systematic time-of-day weaknesses visible.

- Weather cache validation in outputs/weather_cache_validation.csv reports each of the eight weather caches as present, with matching start/end coverage and zero missing values after alignment for the inspected range.

# 9. Risks and Limitations

The following risks were identified from the code, scripts and existing project model card.

| **Risk**                                | **Impact**                                                                                        | **Mitigation**                                                                         |
|-----------------------------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Weather aggregation                     | Equal-weight national weather averages can hide local extreme events.                             | Monitor regional residuals; optionally keep regional features or tune weather weights. |
| Market-zone ambiguity                   | DE versus DE_LU data can introduce systematic bias.                                               | Use 06b_compare_de_vs_de_lu.py and document the selected country code.                 |
| External API dependency                 | ENTSO-E or weather-service outage can delay or degrade forecasts.                                 | Use cache validation, incremental refresh, gap diagnosis and redownload scripts.       |
| DST/time-zone phase shifts              | Hour misalignment can create large systematic forecast errors.                                    | Run 05_diagnose_entsoe_alignment.py and inspect Europe/Berlin to UTC conversions.      |
| Concept drift                           | Industrial demand, electrification, holidays, weather sensitivity or market behaviour may change. | Use rolling-origin backtests and continuous error monitoring.                          |
| Silent leakage from hand-built features | External feature creation can accidentally include future target data.                            | Use the provided ExogBuilder / n_to_1 paths and pipeline validation.                   |
| Memory use for long series              | Lag matrices duplicate data and can exhaust memory for very large T.                              | Use windowing/chunking or reduce feature set.                                          |
| Multi-thread determinism                | Parallel estimators may reorder floating-point reductions.                                        | Pin n_jobs=1 where bit-level reproducibility is more important than speed.             |

# 10. EU AI Act Compliance

The package is not itself a certified high-risk AI system. It is a technical component that supports compliance-oriented development. Full-system certification remains the responsibility of the integrator.

| **EU AI Act / assurance aspect** | **Support in repository**                                                                                                                 |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Data governance                  | Fail-safe NaN/Inf handling, explicit missing-data contracts, timestamp checks, imputation weights and validation tests.                   |
| Technical documentation          | MODEL_CARD.md, README.md, docs/\*.qmd, API reference files and this 13-section model card.                                                |
| Logging and auditability         | manager.logger and audit_log_schema.json; tasks create persistent logs under user cache/model directories.                                |
| Transparency                     | White-box Python implementation, deterministic transformations, documented scripts and readable configuration constants.                  |
| Accuracy and robustness          | Rolling-origin backtesting, extensive test suite, baseline comparisons, edge-hour segmentation and post-processing diagnostics.           |
| Cybersecurity and supply chain   | CPE generator, CycloneDX SBOM enrichment, Sigstore release verification guidance, Dependabot, CodeQL, Safety and Bandit workflow support. |
| Human oversight                  | Designed as decision support; out-of-scope section excludes autonomous control.                                                           |

- CI includes REUSE compliance, prohibited-dependency checks, audit-log schema gates, pytest coverage, linting and security scans.

- Security policy requires private vulnerability reporting, defines response timelines and documents supported versions.

- Release artefacts are intended to include wheel, source distribution, CycloneDX SBOM and Sigstore bundles, enabling tamper-evidence and vulnerability tracking.

# 11. Environmental Impact

The framework is comparatively lightweight. It is CPU-oriented and avoids GPU-heavy deep-learning stacks.

- Incremental data refresh reduces repeated downloads by updating only recent actual-load windows after local data already exists.

- Weather data is cached locally per city and validated before training, reducing redundant API calls.

- Feature generation is based on pandas, NumPy, pyarrow and numba-style vectorized operations rather than large neural networks.

- LightGBM training is efficient on commodity CPUs and model persistence avoids unnecessary retraining when cache reuse is enabled.

- The production dependency footprint deliberately excludes large deep-learning frameworks such as torch and tensorflow, reducing installation size and attack surface.

The environmental footprint is dominated by feature-matrix creation and LightGBM training. For daily 24-hour forecasting on standard electricity-load data, the project can be operated without dedicated GPU infrastructure.

# 12. Glossary

| **Term**                  | **Meaning**                                                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| AGPL                      | GNU Affero General Public License; copyleft license used by the project.                                                         |
| API                       | Application Programming Interface; here mainly ENTSO-E and weather-service access.                                               |
| Backtest                  | Historical simulation of forecasting decisions using only information that would have been available at the forecast issue time. |
| Bias                      | Mean signed forecast error; detects systematic over- or underestimation.                                                         |
| CPE                       | Common Platform Enumeration identifier used for vulnerability tracking.                                                          |
| CycloneDX                 | SBOM format used to describe software components and dependencies.                                                               |
| DE / DE_LU                | ENTSO-E country/market-zone codes; Germany-only versus Germany-Luxembourg.                                                       |
| ENTSO-E                   | European Network of Transmission System Operators for Electricity; provides load and forecast data.                              |
| KRITIS                    | German term for critical infrastructure.                                                                                         |
| Lag                       | Past target value used as a predictor, e.g. lag 24 = same hour previous day.                                                     |
| LightGBM                  | Gradient boosting framework used as primary downstream regressor.                                                                |
| MAE / RMSE / MAPE / sMAPE | Error metrics for absolute, squared and percentage forecast deviations.                                                          |
| SBOM                      | Software Bill of Materials; machine-readable inventory of software components.                                                   |
| Sigstore                  | Keyless signing ecosystem used for release artefact verification.                                                                |
| UPR                       | Under-prediction rate; share of evaluated timestamps where prediction is below actual load.                                      |

# 13. Authors and Contact

| **Role**               | **Name / Contact**                                                                                                                             |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Framework author       | Thomas Bartz-Beielstein                                                                                                                        |
| GitHub organization    | sequential-parameter-optimization                                                                                                              |
| Repository             | https://github.com/sequential-parameter-optimization/spotforecast2-safe                                                                        |
| Documentation          | https://sequential-parameter-optimization.github.io/spotforecast2-safe/                                                                        |
| Security reporting     | GitHub Private Security Advisory or bartzbeielstein@users.noreply.github.com with subject \[SECURITY\] spotforecast2-safe Vulnerability Report |
| Project implementation | Germany load-forecasting configuration maintained in the uploaded ZIP project state                                                            |

## Citation

> @misc{spotforecast2safe,  
> author = {Bartz-Beielstein, Thomas},  
> title = {{spotforecast2-safe}: Safety-critical time-series forecasting for production},  
> year = {2026},  
> howpublished = {https://github.com/sequential-parameter-optimization/spotforecast2-safe},  
> note = {AGPL-3.0-or-later}  
> }

## Audit Instructions

1.  Verify pyproject.toml: package name spotforecast2-safe, version 3.0.0, Python \>=3.13 and AGPL-3.0-or-later.

2.  Run uv lock and confirm prohibited dependencies are absent: plotly, matplotlib, spotoptim, optuna, torch and tensorflow.

3.  Run pytest tests/ and check coverage output for the src/spotforecast2_safe package.

4.  Run the CPE tests and inspect src/spotforecast2_safe/utils/cpe.py plus scripts/enrich_sbom_with_cpe.py.

5.  Validate weather cache using scripts/08_check_weather_cache.py or the validate_multi_weather_cache helper.

6.  Run scripts/09_rolling_origin_backtest_de_load.py before deployment and archive the produced outputs/\*.csv metrics.

7.  Inspect outputs by local hour, horizon and edge group for operationally relevant weakness patterns.

8.  Repeat time-zone alignment checks after daylight-saving-time changes and after data-provider changes.

## Disclaimer and Liability

This software and the Germany load-forecasting configuration are provided as-is. Forecasts are probabilistic/empirical decision-support artefacts and can be wrong. The developers and contributors assume no liability for direct or indirect damages, operational losses or safety incidents resulting from use. Before production deployment, the system integrator must perform full system-level validation, risk assessment, monitoring design and regulatory review.
