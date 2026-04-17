# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import sys

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate
from spotforecast2_safe.preprocessing import QuantileBinner


def test_init_with_int_offset():
    """
    Test initialization with an integer offset.
    """
    forecaster = ForecasterEquivalentDate(offset=7, n_offsets=2)
    assert forecaster.offset == 7
    assert forecaster.n_offsets == 2
    assert forecaster.window_size == 14
    assert forecaster.agg_func == np.mean
    assert forecaster.is_fitted is False
    assert forecaster.forecaster_id is None
    assert forecaster.estimator is None
    assert forecaster.differentiation is None
    assert forecaster.differentiation_max is None


def test_init_with_dateoffset():
    """
    Test initialization with a pandas DateOffset.
    """
    offset = pd.tseries.offsets.Day(7)
    forecaster = ForecasterEquivalentDate(offset=offset, n_offsets=3)
    assert forecaster.offset == offset
    assert forecaster.n_offsets == 3
    # window_size calculation should work for DateOffset * int
    assert forecaster.window_size == offset * 3
    assert forecaster.agg_func == np.mean


def test_init_error_wrong_type_offset():
    """
    Test TypeError is raised when offset is not int or DateOffset.
    """
    with pytest.raises(
        TypeError, match="`offset` must be an integer greater than 0 or a"
    ):
        ForecasterEquivalentDate(offset="7")


def test_init_default_binner_kwargs():
    """
    Test default binner_kwargs are set correctly.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    expected_kwargs = {
        "n_bins": 10,
        "method": "linear",
        "subsample": 200000,
        "random_state": 789654,
        "dtype": np.float64,
    }
    assert forecaster.binner_kwargs == expected_kwargs
    assert isinstance(forecaster.binner, QuantileBinner)
    assert forecaster.binner.n_bins == 10


def test_init_custom_binner_kwargs():
    """
    Test custom binner_kwargs are respected.
    """
    custom_kwargs = {"n_bins": 5, "method": "median_unbiased"}
    forecaster = ForecasterEquivalentDate(offset=7, binner_kwargs=custom_kwargs)
    assert forecaster.binner_kwargs == custom_kwargs
    assert forecaster.binner.n_bins == 5


def test_init_spotforecast_tags():
    """
    Test __spotforecast_tags__ content and keys for parity.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    tags = forecaster.__spotforecast_tags__
    assert tags["library"] == "spotforecast"
    assert tags["forecaster_name"] == "ForecasterEquivalentDate"
    assert tags["forecasting_strategy"] == "recursive"
    assert tags["supports_exog"] is False
    assert tags["handles_binned_residuals"] is True


def test_init_versions():
    """
    Test version and python version attributes.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    assert hasattr(forecaster, "spotforecast_version")
    assert forecaster.python_version == sys.version.split(" ")[0]
