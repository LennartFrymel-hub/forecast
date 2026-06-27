# SPDX-FileCopyrightText: 2026 Lennart Frymel
# SPDX-License-Identifier: AGPL-3.0-or-later

from lightgbm import LGBMRegressor


def build_default_lgbm_regressor(random_state: int = 1234) -> LGBMRegressor:
    return LGBMRegressor(
        random_state=random_state,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        reg_alpha=0.05,
        reg_lambda=0.2,
        subsample=0.9,
        colsample_bytree=0.9,
    )