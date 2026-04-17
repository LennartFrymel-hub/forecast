# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_set_in_sample_residuals_success():
    """
    Test successful execution of set_in_sample_residuals.
    """
    forecaster = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 2})
    y = pd.Series(
        np.arange(21, dtype=float),
        index=pd.date_range("2022-01-01", periods=21, freq="D"),
    )
    forecaster.fit(y=y, store_in_sample_residuals=False)

    assert forecaster.in_sample_residuals_ is None

    forecaster.set_in_sample_residuals(y=y, random_state=123)

    assert forecaster.in_sample_residuals_ is not None
    assert len(forecaster.in_sample_residuals_) == 14  # 21 - 7
    assert forecaster.in_sample_residuals_by_bin_ is not None
    assert len(forecaster.in_sample_residuals_by_bin_) == 2


def test_set_in_sample_residuals_not_fitted_raises_error():
    """
    Test NotFittedError is raised if forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(np.arange(10, dtype=float))

    with pytest.raises(NotFittedError, match="is not fitted yet"):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_index_mismatch_raises_error():
    """
    Test IndexError is raised if y index does not match training range.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y)

    y_mismatch = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-02-01", periods=14, freq="D"),
    )

    with pytest.raises(IndexError, match="index range of `y` does not match"):
        forecaster.set_in_sample_residuals(y=y_mismatch)


def test_set_in_sample_residuals_random_state():
    """
    Test random_state effect on sampling (if data > 10,000).
    For this class, the sampling logic is inside _binning_in_sample_residuals.
    Since we don't have > 10,000 points here, we just verify it runs with random_state.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2022-01-01", periods=20, freq="D"),
    )
    forecaster.fit(y=y)

    forecaster.set_in_sample_residuals(y=y, random_state=123)
    res_1 = forecaster.in_sample_residuals_.copy()

    forecaster.set_in_sample_residuals(y=y, random_state=123)
    res_2 = forecaster.in_sample_residuals_.copy()

    np.testing.assert_array_equal(res_1, res_2)
