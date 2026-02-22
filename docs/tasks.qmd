# Task Scripts

`spotforecast2-safe` provides command-line task scripts for safety-critical forecasting workflows. These scripts are registered as console entry points and can be invoked directly via `uv run` or after package installation.

## Available Commands

| Command | Description |
|---------|-------------|
| `spotforecast-safe-demo` | Demo task comparing baseline, covariate, and custom LightGBM forecasts |
| `spotforecast-safe-n2o1-cov-df` | N-to-1 forecasting with exogenous covariates and DataFrame input |

---

## Demo Task

The `spotforecast-safe-demo` command runs a comprehensive comparison of three forecasting approaches:

1. **Baseline**: Standard N-to-1 recursive forecaster
2. **Covariate-enhanced**: Includes weather, holidays, and cyclical features
3. **Custom LightGBM**: Optimized hyperparameters with safety-critical configuration

### Usage

```bash
# Run with default settings (force training)
uv run spotforecast-safe-demo

# Skip training (use cached models if available)
uv run spotforecast-safe-demo --force_train false

# Specify custom data path
uv run spotforecast-safe-demo --data_path /path/to/data.csv

# Enable logging
uv run spotforecast-safe-demo --logging true
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General failure |
| 2 | Data loading error |
| 3 | Model training error |

!!! warning "Safety-Critical Consideration"
    The demo task logs all execution steps and errors. In production, always enable logging with `--logging true` for auditability.

---

## N-to-1 with Covariates and DataFrame

The `spotforecast-safe-n2o1-cov-df` command executes the full N-to-1 forecasting pipeline with exogenous covariates and flexible data input.

### Features

- **Automatic feature engineering**: Weather, holidays, cyclical time features
- **Weighted aggregation**: Combine multiple forecasts with configurable weights
- **DataFrame input**: Pass custom data or use default data fetcher
- **Comprehensive logging**: Safety-critical execution tracing

### Usage

```bash
# Run with default settings
uv run spotforecast-safe-n2o1-cov-df

# Custom forecast horizon
uv run spotforecast-safe-n2o1-cov-df --forecast_horizon 48

# Enable verbose output
uv run spotforecast-safe-n2o1-cov-df --verbose true

# Enable logging to file
uv run spotforecast-safe-n2o1-cov-df --logging true --log_dir ~/logs
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--forecast_horizon` | 24 | Number of steps ahead to forecast |
| `--contamination` | 0.01 | Outlier detection threshold |
| `--window_size` | 72 | Rolling window size for features |
| `--lags` | 24 | Number of lag features |
| `--train_ratio` | 0.8 | Train/validation split ratio |
| `--verbose` | False | Enable detailed output |
| `--logging` | False | Enable file logging |

---

## Configuration

All tasks use sensible defaults but can be customized via:

- **Command-line arguments** (use `--help` for details)
- **Environment variables** for API keys and paths
- **Configuration files** stored in `~/spotforecast2_data/`

```bash
# View available options for any command
uv run spotforecast-safe-demo --help
uv run spotforecast-safe-n2o1-cov-df --help
```

---

## Model Persistence

Trained models are saved to `~/spotforecast2_models/<task_name>/` by default. This allows:

- **Incremental retraining**: Only retrain when models are stale
- **Reproducibility**: Models are versioned by task and timestamp
- **Auditability**: Full training logs are stored alongside models

!!! tip "Best Practice"
    For production deployments, always verify model checksums and training timestamps before using cached models.

---

## Logging

Safety-critical tasks support comprehensive logging:

```bash
# Enable logging to default directory
uv run spotforecast-safe-demo --logging true

# Specify custom log directory
uv run spotforecast-safe-n2o1-cov-df --logging true --log_dir /var/log/spotforecast
```

Log files include:

- Execution timestamps
- Parameter configurations
- Model training metrics
- Error tracebacks (if any)
