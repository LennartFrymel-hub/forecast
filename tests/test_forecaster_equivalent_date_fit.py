# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import Day

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_fit_integer_offset():
    """
    Test fit with an integer offset.
    """
    forecaster = ForecasterEquivalentDate(offset=7, n_offsets=2)
    y = pd.Series(
        np.arange(20), index=pd.date_range("2022-01-01", periods=20, freq="D")
    )
    forecaster.fit(y=y)

    assert forecaster.is_fitted
    assert forecaster.window_size == 14
    assert forecaster.index_freq_ == "D"
    assert forecaster.index_type_ == pd.DatetimeIndex
    assert isinstance(forecaster.last_window_, pd.Series)
    assert len(forecaster.last_window_) == 20
    np.testing.assert_array_equal(
        forecaster.training_range_.to_numpy(), y.index[[0, -1]].to_numpy()
    )


def test_fit_dateoffset():
    """
    Test fit with a pandas DateOffset.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(7), n_offsets=1)
    y = pd.Series(
        np.arange(20), index=pd.date_range("2022-01-01", periods=20, freq="D")
    )
    forecaster.fit(y=y)

    assert forecaster.is_fitted
    assert forecaster.window_size == 7
    assert forecaster.index_freq_ == "D"
    assert forecaster.index_type_ == pd.DatetimeIndex


def test_fit_frequency_inference():
    """
    Test fit correctly infers frequency if missing from index.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(1), n_offsets=1)
    idx = pd.date_range("2022-01-01", periods=10, freq="D")
    idx.freq = None  # Remove frequency
    y = pd.Series(np.arange(10), index=idx)

    # In spotforecast2-safe, this should trigger frequency inference
    forecaster.fit(y=y)

    assert forecaster.is_fitted
    assert forecaster.index_freq_ == "D"


def test_fit_error_insufficient_data_integer_offset():
    """
    Test ValueError is raised if y is too short for integer offset.
    """
    forecaster = ForecasterEquivalentDate(offset=10, n_offsets=1)
    y = pd.Series(np.arange(5))  # Length 5 < window_size 10

    msg = "Length of `y` must be greater than the maximum window size"
    with pytest.raises(ValueError, match=msg):
        forecaster.fit(y=y)


def test_fit_error_insufficient_data_dateoffset():
    """
    Test ValueError is raised if y is too short for DateOffset.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(10), n_offsets=1)
    y = pd.Series(np.arange(5), index=pd.date_range("2022-01-01", periods=5, freq="D"))

    msg = "The length of `y` .* must be greater than or equal to the window size"
    with pytest.raises(ValueError, match=msg):
        forecaster.fit(y=y)


def test_fit_error_invalid_y_type():
    """
    Test TypeError is raised if y is not a pandas Series.
    """
    forecaster = ForecasterEquivalentDate(offset=1)
    with pytest.raises(TypeError, match="`y` must be a pandas Series"):
        forecaster.fit(y=[1, 2, 3])


def test_fit_resets_fitted_state():
    """
    Test fit resets attributes if called multiple times.
    """
    forecaster = ForecasterEquivalentDate(offset=10)
    y1 = pd.Series(np.arange(15))
    forecaster.fit(y=y1)
    assert forecaster.is_fitted

    y2 = pd.Series(np.arange(5))  # Too short, should fail and reset is_fitted
    with pytest.raises(ValueError):
        forecaster.fit(y=y2)

    assert not forecaster.is_fitted
    assert forecaster.last_window_ is None


def test_fit_stores_residuals_binned():
    """
    Test that _binning_in_sample_residuals is executed correctly.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    y = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2022-01-01", periods=10, freq="D"),
    )

    forecaster.fit(y=y, store_in_sample_residuals=True)

    assert hasattr(forecaster, "in_sample_residuals_")
    assert forecaster.in_sample_residuals_ is not None
    assert len(forecaster.in_sample_residuals_) == 9  # 10 - 1 (offset)
    assert isinstance(forecaster.in_sample_residuals_by_bin_, dict)
