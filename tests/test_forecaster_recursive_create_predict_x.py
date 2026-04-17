# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures


def test_create_predict_X_output_shape_and_columns():
    """
    Test create_predict_X returns a DataFrame with the correct shape and columns.
    """
    y = pd.Series(np.arange(10), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        window_features=[RollingFeatures(stats="mean", window_sizes=2)],
    )
    forecaster.fit(y=y)

    steps = 5
    X_predict = forecaster.create_predict_X(steps=steps)

    assert isinstance(X_predict, pd.DataFrame)
    assert X_predict.shape == (steps, 4)  # 3 lags + 1 window feature
    assert list(X_predict.columns) == forecaster.X_train_features_names_out_
    assert len(X_predict) == steps


def test_create_predict_X_with_exog_categorical():
    """
    Test create_predict_X with categorical exogenous variables.
    """
    y = pd.Series(np.arange(10, dtype=float), name="y")
    exog = pd.DataFrame(
        {
            "exog_cat": pd.Series([0, 1] * 5, dtype="category"),
            "exog_num": np.arange(10, dtype=float),
        },
        index=y.index,
    )

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
    forecaster.fit(y=y, exog=exog)

    steps = 2
    exog_predict = pd.DataFrame(
        {"exog_cat": pd.Series([0, 1], dtype="category"), "exog_num": [10.0, 11.0]},
        index=pd.RangeIndex(start=10, stop=12),
    )

    X_predict = forecaster.create_predict_X(steps=steps, exog=exog_predict)

    assert X_predict["exog_cat"].dtype == "category"
    assert X_predict["exog_num"].dtype == float
    assert X_predict.shape == (steps, 3)  # 1 lag + 2 exog


def test_create_predict_X_transformation_warning():
    """
    Test create_predict_X raises a warning when transformations are used.
    """
    from spotforecast2_safe.exceptions import DataTransformationWarning

    y = pd.Series(np.arange(10, dtype=float), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=2, differentiation=1
    )
    forecaster.fit(y=y)

    with pytest.warns(
        DataTransformationWarning, match="The output matrix is in the transformed scale"
    ):
        _ = forecaster.create_predict_X(steps=3)
