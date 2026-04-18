# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate
from spotforecast2_safe.forecaster.recursive._warnings import ResidualsUsageWarning


def test_set_out_sample_residuals_success():
    """
    Test successful execution of set_out_sample_residuals.
    """
    forecaster = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 2})
    y = pd.Series(
        np.arange(21, dtype=float),
        index=pd.date_range("2022-01-01", periods=21, freq="D"),
    )
    forecaster.fit(y=y)

    y_true = np.array([10, 11, 12])
    y_pred = np.array([10.5, 10.8, 12.2])

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert forecaster.out_sample_residuals_ is not None
    assert len(forecaster.out_sample_residuals_) == 3
    assert forecaster.out_sample_residuals_by_bin_ is not None
    assert len(forecaster.out_sample_residuals_by_bin_) == 2


def test_set_out_sample_residuals_append():
    """
    Test successful execution of set_out_sample_residuals with append=True.
    """
    forecaster = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 2})
    y = pd.Series(
        np.arange(21, dtype=float),
        index=pd.date_range("2022-01-01", periods=21, freq="D"),
    )
    forecaster.fit(y=y)

    y_true_1 = np.array([10, 11])
    y_pred_1 = np.array([10.5, 10.8])
    forecaster.set_out_sample_residuals(y_true=y_true_1, y_pred=y_pred_1)

    y_true_2 = np.array([12])
    y_pred_2 = np.array([12.2])
    forecaster.set_out_sample_residuals(y_true=y_true_2, y_pred=y_pred_2, append=True)

    assert len(forecaster.out_sample_residuals_) == 3


def test_set_out_sample_residuals_not_fitted_raises_error():
    """
    Test NotFittedError is raised if forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y_true = np.array([10])
    y_pred = np.array([10.5])

    with pytest.raises(NotFittedError, match="is not fitted yet"):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_length_mismatch_raises_error():
    """
    Test ValueError is raised if y_true and y_pred have different lengths.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(np.arange(14, dtype=float))
    forecaster.fit(y=y)

    y_true = np.array([10, 11])
    y_pred = np.array([10.5])

    with pytest.raises(ValueError, match="must have the same length"):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_index_mismatch_raises_error():
    """
    Test ValueError is raised if y_true and y_pred are pd.Series with different indexes.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y)

    idx1 = pd.date_range("2022-02-01", periods=2, freq="D")
    idx2 = pd.date_range("2022-02-02", periods=2, freq="D")
    y_true = pd.Series([10, 11], index=idx1)
    y_pred = pd.Series([10.5, 10.8], index=idx2)

    with pytest.raises(ValueError, match="must have the same index"):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_warning_empty_bins():
    """
    Test ResidualsUsageWarning is raised if some bins are empty.
    """
    forecaster = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 10})
    y = pd.Series(
        np.arange(100, dtype=float),
        index=pd.date_range("2022-01-01", periods=100, freq="D"),
    )
    forecaster.fit(y=y)

    # Use very few points to ensure some bins are empty
    y_true = np.array([10])
    y_pred = np.array([10.5])

    with pytest.warns(
        ResidualsUsageWarning, match="bins have no out of sample residuals"
    ):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
