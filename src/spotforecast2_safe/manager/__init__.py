# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Manager module for model persistence, prediction, and training.

This module provides utilities for:
- Logging configuration for safety-critical systems
- Model persistence (save/load)
- Model prediction management
- Model training and retraining workflows
- Model evaluation metrics
- Dataset configurations (see manager.datasets submodule)
- CLI argument parsing utilities
- Exogenous feature engineering (see manager.exo submodule)
"""

from spotforecast2_safe.manager.logger import setup_logging
from spotforecast2_safe.manager.metrics import calculate_metrics
from spotforecast2_safe.manager.persistence import (
    _ensure_model_dir,
    _get_model_filepath,
    _load_forecasters,
    _model_directory_exists,
    _save_forecasters,
)
from spotforecast2_safe.manager.predictor import get_model_prediction
from spotforecast2_safe.manager.tools import _parse_bool
from spotforecast2_safe.manager.trainer import get_last_model
from spotforecast2_safe.manager.exo import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_features,
    get_weather_features,
)

__all__ = [
    # Logger
    "setup_logging",
    # Metrics
    "calculate_metrics",
    # Persistence
    "_ensure_model_dir",
    "_get_model_filepath",
    "_save_forecasters",
    "_load_forecasters",
    "_model_directory_exists",
    # Predictor
    "get_model_prediction",
    # Trainer
    "get_last_model",
    # Tools
    "_parse_bool",
    # Exo feature engineering
    "get_calendar_features",
    "get_day_night_features",
    "get_holiday_features",
    "get_weather_features",
]
