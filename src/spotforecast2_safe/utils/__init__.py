# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for spotforecast."""

from spotforecast2_safe.utils.data_transform import (
    date_to_index_position,
    expand_index,
    input_to_frame,
    transform_dataframe,
)
from spotforecast2_safe.utils.forecaster_config import (
    check_select_fit_kwargs,
    initialize_lags,
    initialize_weights,
)
from spotforecast2_safe.utils.generate_holiday import create_holiday_df
from spotforecast2_safe.utils.validation import (
    DataTypeWarning,
    MissingValuesWarning,
    check_exog,
    check_exog_dtypes,
    check_interval,
    check_predict_input,
    check_residuals_input,
    check_y,
    get_exog_dtypes,
    set_cpu_gpu_device,
)

__all__ = [
    "check_y",
    "check_exog",
    "get_exog_dtypes",
    "check_interval",
    "MissingValuesWarning",
    "DataTypeWarning",
    "input_to_frame",
    "initialize_lags",
    "expand_index",
    "initialize_weights",
    "check_select_fit_kwargs",
    "check_exog_dtypes",
    "check_predict_input",
    "transform_dataframe",
    "create_holiday_df",
    "check_residuals_input",
    "date_to_index_position",
    "set_cpu_gpu_device",
]
