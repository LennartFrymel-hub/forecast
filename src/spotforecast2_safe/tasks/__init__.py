# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from .task_safe_demo import main as demo_main
from .task_safe_n_to_1_with_covariates_and_dataframe import main as n2o1_cov_df_main

__all__ = ["demo_main", "n2o1_cov_df_main"]
