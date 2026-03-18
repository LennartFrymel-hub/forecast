# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for managing model predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from spotforecast2_safe.manager.trainer import get_last_model

logger = logging.getLogger(__name__)


def get_model_prediction(
    model_name: str,
    model_dir: Optional[Union[str, Path]] = None,
    predict_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the prediction package from the latest trained model.
    This function retrieves the latest iteration of a specified model from the
    cache and calls its `package_prediction` method to obtain a comprehensive
    set of predictions and metrics.

    Args:
        model_name: Name of the model to use (e.g., 'lgbm', 'xgb').
        model_dir: Directory where models are stored. If None, defaults to
            the library's cache home.
        predict_size: Optional override for the prediction horizon.

    Returns:
        A dictionary containing predictions and metrics if a model is found and
        successfully executes `package_prediction`. Returns None otherwise.

    Notes:
         `predict_size` is accepted by `get_model_prediction()` but only has effect if the concrete model's `package_prediction()` accepts it.
         The original `ForecasterRecursiveModel.package_prediction()` does not — so this parameter is currently forward-looking API design, not yet wired end-to-end.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.predictor import get_model_prediction
        >>> from joblib import dump
        >>>
        >>> # Example 1: No model found scenario
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     result = get_model_prediction('lgbm', model_dir=tmpdir)
        ...     print(f"Result when no model exists: {result}")
        Result when no model exists: None
        >>>
        >>> # Example 2: Model found but no package_prediction method
        >>> class SimpleModel:  # doctest: +SKIP
        ...     '''Simple model without package_prediction method'''
        ...     def __init__(self):
        ...         self.name = 'simple'
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     simple_model = SimpleModel()
        ...     dump(simple_model, model_dir / "test_forecaster_1.joblib")
        ...     result = get_model_prediction('test', model_dir=model_dir)
        ...     print(f"Result without package_prediction: {result}")
        Result without package_prediction: None
        >>>
        >>> # Example 3: Successful prediction package retrieval
        >>> class ForecastModel:  # doctest: +SKIP
        ...     '''Model with package_prediction method'''
        ...     def __init__(self):
        ...         self.name = 'xgb'
        ...     def package_prediction(self):
        ...         return {
        ...             'predictions': [1.0, 2.0, 3.0],
        ...             'metrics': {'mse': 0.05, 'mae': 0.02}
        ...         }
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     forecast_model = ForecastModel()
        ...     dump(forecast_model, model_dir / "xgb_forecaster_1.joblib")
        ...     result = get_model_prediction('xgb', model_dir=model_dir)
        ...     print(f"Predictions available: {'predictions' in result}")
        ...     print(f"Metrics available: {'metrics' in result}")
        Predictions available: True
        Metrics available: True
        >>>
        >>> # Example 4: Safety-critical - verify prediction integrity
        >>> class SafetyModel:  # doctest: +SKIP
        ...     '''Safety model with validation'''
        ...     def __init__(self):
        ...         self.name = 'safety_forecaster'
        ...     def package_prediction(self):
        ...         return {
        ...             'predictions': [10.5, 11.2],
        ...             'confidence_intervals': [(10.0, 11.0), (10.8, 11.6)],
        ...             'validation_passed': True
        ...         }
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     safety_model = SafetyModel()
        ...     dump(safety_model, model_dir / "safety_forecaster_forecaster_2.joblib")
        ...     pkg = get_model_prediction('safety_forecaster', model_dir=model_dir)
        ...     if pkg:
        ...         print(f"Validation status: {pkg['validation_passed']}")
        Validation status: True
    """
    n_iteration, model = get_last_model(model_name, model_dir)

    if n_iteration < 0 or model is None:
        logger.error(
            "No trained model found for '%s'. Please train a model first.", model_name
        )
        return None

    logger.info(
        "Making predictions using %s model (iteration %d)...",
        model_name,
        n_iteration,
    )

    if not hasattr(model, "package_prediction"):
        logger.error(
            "Model '%s' (iteration %d) does not implement 'package_prediction' method.",
            model_name,
            n_iteration,
        )
        return None

    try:
        prediction_package = model.package_prediction(predict_size=predict_size)
        return prediction_package
    except Exception as e:
        logger.error(
            "Error occurred while generating prediction package for '%s': %s",
            model_name,
            e,
            exc_info=True,
        )
        return None
