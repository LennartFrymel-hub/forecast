# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_create_train_X_y_exog_full_length_alignment():
    """
    Test _create_train_X_y with exog matching full y length and aligned index.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    exog = pd.DataFrame({"exog": np.arange(10)}, index=y.index)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    X_train, y_train, exog_names_in_, _, exog_names_out_, _, _, _ = (
        forecaster._create_train_X_y(y=y, exog=exog)
    )

    assert len(X_train) == 7  # 10 - 3
    assert (X_train.index == y.index[3:]).all()
    assert (X_train["exog"] == exog["exog"].iloc[3:]).all()
    assert exog_names_in_ == ["exog"]
    assert exog_names_out_ == ["exog"]


def test_create_train_X_y_exog_train_length_alignment():
    """
    Test _create_train_X_y with exog matching train_index length and aligned index.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    train_index = y.index[3:]
    exog = pd.DataFrame({"exog": np.arange(7)}, index=train_index)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    X_train, y_train, _, _, _, _, _, _ = forecaster._create_train_X_y(y=y, exog=exog)

    assert len(X_train) == 7
    assert (X_train.index == train_index).all()
    assert (X_train["exog"] == exog["exog"]).all()


def test_create_train_X_y_exog_length_mismatch_error():
    """
    Test _create_train_X_y raises ValueError if exog length matches neither y nor train_index.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    exog = pd.DataFrame(
        {"exog": np.arange(5)}, index=pd.date_range("2024-01-01", periods=5, freq="D")
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with pytest.raises(ValueError, match="Length mismatch for exogenous variables"):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_exog_full_length_index_mismatch_error():
    """
    Test _create_train_X_y raises ValueError if exog full-length index is not aligned with y.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    exog = pd.DataFrame(
        {"exog": np.arange(10)}, index=pd.date_range("2024-01-02", periods=10, freq="D")
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with pytest.raises(
        ValueError, match="the index of `exog` must be aligned with the index of `y`"
    ):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_exog_train_length_index_mismatch_error():
    """
    Test _create_train_X_y raises ValueError if exog train-length index is not aligned with train_index.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    exog = pd.DataFrame(
        {"exog": np.arange(7)}, index=pd.date_range("2024-01-05", periods=7, freq="D")
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with pytest.raises(
        ValueError,
        match="its index must be aligned with the index of `y` starting from `window_size`",
    ):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_differentiation():
    """
    Test _create_train_X_y with differentiation.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )

    X_train, y_train, _, _, _, _, _, _ = forecaster._create_train_X_y(y=y)

    # Differentiation order 1 means first value of differentiated y is NaN.
    # TimeSeriesDifferentiator.transform pads with NaNs.
    # window_size = max(lags, window_features_size) + differentiation = 3 + 1 = 4.
    # train_index = y_index[4:] -> length 6.
    # _create_lags creates lags from y_values using window_size=4.
    assert X_train.shape == (6, 3)
    # The first row of X_train will have NaNs if differentiation is used
    # and not handled properly (shifted).
    # But skforecast usually expects cleaned data or handles NaNs.
    # our TimeSeriesDifferentiator pads with NaNs.


def test_create_train_X_y_public_api():
    """
    Test public create_train_X_y returns only 2 items.
    """
    y = pd.Series(
        np.arange(10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y"
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    output = forecaster.create_train_X_y(y=y)

    assert len(output) == 2
    assert isinstance(output[0], pd.DataFrame)
    assert isinstance(output[1], pd.Series)
    assert output[0].shape == (7, 3)
    assert output[1].shape == (7,)
