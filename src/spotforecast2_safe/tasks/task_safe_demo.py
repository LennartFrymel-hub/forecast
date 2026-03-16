# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Task demo: compare baseline, covariate, and custom LightGBM forecasts against ground truth.

This script executes the baseline N-to-1 task, the covariate-enhanced N-to-1
pipeline, and a custom LightGBM model with optimized hyperparameters, then loads
the ground truth from a specified data directory.
Logging Mechanism: This script uses a dual-handler logging system designed for safety-critical MLOps:

* Console Handler:
    Provides real-time progress updates to `stdout`.
* File Handler:
    Persists all log messages (including debug/tracebacks) to a timestamped file in `{model_root}/logs/`.
* Log File Location:
    By default, logs are saved to `~/spotforecast2_safe_models/logs/task_safe_demo_YYYYMMDD_HHMMSS.log`.
* Safety-Critical Features:
    * Persistent file-based logging for auditability.
    * Path management using pathlib for cross-platform reliability.
    * Explicit input validation and existence checks.
    * Comprehensive error handling with traceback logging.
    * Deterministic random seeding where applicable.
    * Minimal dependency footprint (no plotting libraries).

Examples:
    # Run with default settings (force training)
    >>> uv run spotforecast-safe-demo

    # Skip training (use cached models if available)
    >>> uv run spotforecast-safe-demo --force_train false

    # Specify custom data path
    >>> uv run spotforecast-safe-demo --data_path /path/to/data.csv

    # Enable logging
    >>> uv run spotforecast-safe-demo --logging true
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

from lightgbm import LGBMRegressor

from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict import n2n_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)
from spotforecast2_safe.manager.logger import setup_logging
from spotforecast2_safe.manager.tools import _parse_bool
from spotforecast2_safe.manager.metrics import calculate_metrics
from spotforecast2_safe.manager.datasets import DemoConfig, load_actual_combined


def main(
    force_train: bool = True,
    data_path: Optional[Path] = None,
    logging_enabled: bool = False,
) -> int:
    """
    Main execution entry point.
    Returns 0 on success, non-zero on failure.
    """
    # Initialize configuration first to get log path
    default_config = DemoConfig()
    config = DemoConfig(
        data_path=data_path if data_path else default_config.data_path,
        model_root=default_config.model_root,
        log_root=default_config.log_root,
        forecast_horizon=default_config.forecast_horizon,
        contamination=default_config.contamination,
        window_size=default_config.window_size,
        lags=default_config.lags,
        train_ratio=default_config.train_ratio,
        random_seed=default_config.random_seed,
        weights=default_config.weights,
    )

    # Setup Logging if enabled
    log_file = None
    if logging_enabled:
        logger, log_file = setup_logging(log_dir=config.log_root)
    else:
        logger = logging.getLogger("task_safe_demo")
        logger.addHandler(logging.NullHandler())

    # --- Fail-Fast Check: Validate ground truth availability before compute ---
    logger.info(f"Validating ground truth availability: {config.data_path}")
    if not config.data_path.is_file():
        logger.error(f"FAIL-FAST: Ground truth file not found at {config.data_path}")
        logger.error(
            "Please ensure the file exists or specify a valid path via --data_path"
        )
        return 1
    logger.info("Ground truth file verified.")

    try:
        # --- Baseline predictions ---
        logger.info("Executing baseline forecast...")
        baseline_predictions, _ = n2n_predict(
            columns=None,
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            verbose=False,  # Use our own logging
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_baseline",
        )

        baseline_combined = agg_predict(baseline_predictions, weights=config.weights)
        logger.info(f"Baseline combined generated. Shape: {baseline_combined.shape}")

        # --- Covariate-enhanced predictions ---
        logger.info("Executing covariate-enhanced forecast...")
        cov_predictions, _, _ = n2n_predict_with_covariates(
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            lags=config.lags,
            train_ratio=config.train_ratio,
            verbose=False,
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_covariates",
        )

        covariates_combined = agg_predict(cov_predictions, weights=config.weights)
        logger.info(
            f"Covariates combined generated. Shape: {covariates_combined.shape}"
        )

        # --- Custom LightGBM predictions ---
        logger.info("Executing custom LightGBM forecast...")
        custom_lgbm = LGBMRegressor(
            n_estimators=1059,
            learning_rate=0.04191323446625026,
            num_leaves=212,
            min_child_samples=54,
            subsample=0.5014650987802548,
            colsample_bytree=0.6080926628683118,
            random_state=config.random_seed,
            verbose=-1,
        )
        custom_lgbm_predictions, _, _ = n2n_predict_with_covariates(
            forecast_horizon=config.forecast_horizon,
            contamination=config.contamination,
            window_size=config.window_size,
            lags=config.lags,
            train_ratio=config.train_ratio,
            timezone="UTC",
            estimator=custom_lgbm,
            verbose=False,
            show_progress=True,
            force_train=force_train,
            model_dir=config.model_root / "task_demo_custom_lgbm",
        )
        custom_lgbm_combined = agg_predict(
            custom_lgbm_predictions, weights=config.weights
        )
        logger.info(
            f"Custom LightGBM combined generated. Shape: {custom_lgbm_combined.shape}"
        )

        # --- Comparative Metrics ---
        logger.info("Comparing models...")
        idx_match = (baseline_combined.index == covariates_combined.index).all()
        logger.info(f"Indices align between baseline and covariates: {idx_match}")

        # --- Ground truth and Evaluation ---
        # Note: We checked for file existence above, but load_actual_combined
        # also performs schema validation.
        try:
            columns = list(baseline_predictions.columns)
            actual_combined = load_actual_combined(config=config, columns=columns)
            actual_combined = actual_combined.reindex(baseline_combined.index)

            # Use fillna(0) or dropna() based on safety policy; here we assume test data should be complete
            if actual_combined.isnull().any():
                logger.warning("Actual combined series contains NaNs after reindexing.")
                actual_combined = actual_combined.dropna()
                common_idx = actual_combined.index.intersection(baseline_combined.index)
                actual_combined = actual_combined.loc[common_idx]
                baseline_combined = baseline_combined.loc[common_idx]
                covariates_combined = covariates_combined.loc[common_idx]
                custom_lgbm_combined = custom_lgbm_combined.loc[common_idx]

            metrics_baseline = calculate_metrics(actual_combined, baseline_combined)
            metrics_cov = calculate_metrics(actual_combined, covariates_combined)
            metrics_lgbm = calculate_metrics(actual_combined, custom_lgbm_combined)

            logger.info("\n=== EVALUATION RESULTS ===")
            logger.info(f"Baseline:       {metrics_baseline}")
            logger.info(f"Covariates:     {metrics_cov}")
            logger.info(f"Custom LGBM:    {metrics_lgbm}")
            logger.info("==========================\n")

        except Exception as e:
            logger.error(f"Error during evaluation (checking data again): {e}")

        logger.info("--- task_safe_demo completed successfully ---")
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 0

    except Exception as _e:  # noqa: F841
        logger.critical("Task failed with an unexpected error:")
        logger.critical(traceback.format_exc())
        if log_file:
            print(f"Finalized logging info saved to: {log_file}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the spotforecast2_safe demo task."
    )
    parser.add_argument(
        "--force_train",
        type=_parse_bool,
        default=True,
        help="Force training (true/false, case-insensitive).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the ground truth CSV file.",
    )
    parser.add_argument(
        "--logging",
        type=_parse_bool,
        default=False,
        help="Enable logging (both console and file).",
    )
    args = parser.parse_args()

    # Convert string path to Path object if provided
    specified_data_path = Path(args.data_path) if args.data_path else None

    sys.exit(
        main(
            force_train=args.force_train,
            data_path=specified_data_path,
            logging_enabled=args.logging,
        )
    )
