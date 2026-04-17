# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Any

from lightgbm import LGBMRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from .forecaster_recursive_model import ForecasterRecursiveModel

logger = logging.getLogger(__name__)


class ForecasterRecursiveLGBM(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using LightGBM.

    Examples:
        >>> from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import ForecasterRecursiveLGBM
        >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
        >>> model = ForecasterRecursiveLGBM(iteration=0)
        >>> model.name
        'lgbm'
        >>> isinstance(model, ForecasterRecursiveModel)
        True
        >>> model.forecaster is not None
        True
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the LGBM Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.
        """
        super().__init__(iteration, name="lgbm", **kwargs)
        self.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(
                n_jobs=-1, verbose=-1, random_state=self.random_state
            ),
            lags=lags,
        )
