# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_binning_in_sample_residuals_integer_offset():
    """
    Test _binning_in_sample_residuals with an integer offset.
    """
    forecaster = ForecasterEquivalentDate(
        offset=1, n_offsets=1, binner_kwargs={"n_bins": 2}
    )
    y = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2022-01-01", periods=10, freq="D"),
    )

    forecaster.fit(y=y, store_in_sample_residuals=True)

    assert hasattr(forecaster, "in_sample_residuals_")
    assert hasattr(forecaster, "in_sample_residuals_by_bin_")
    assert len(forecaster.in_sample_residuals_) == 9
    assert len(forecaster.in_sample_residuals_by_bin_) == 2
    for bin_residuals in forecaster.in_sample_residuals_by_bin_.values():
        assert isinstance(bin_residuals, np.ndarray)


def test_binning_in_sample_residuals_dateoffset():
    """
    Test _binning_in_sample_residuals with a pandas DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
        offset=Day(1), n_offsets=1, binner_kwargs={"n_bins": 2}
    )
    y = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2022-01-01", periods=10, freq="D"),
    )

    forecaster.fit(y=y, store_in_sample_residuals=True)

    assert len(forecaster.in_sample_residuals_) == 9
    assert len(forecaster.in_sample_residuals_by_bin_) == 2


def test_binning_in_sample_residuals_sampling_limit_per_bin():
    """
    Test that the number of residuals per bin is limited.
    """
    n_bins = 2
    max_sample_per_bin = 10_000 // n_bins
    # Create enough data to exceed the limit per bin
    # 20 elements, 2 bins -> 10 per bin. Let's make it larger.
    # To really test 10_000 // n_bins, we would need 10,001 residuals in a bin.
    # For efficiency in testing, let's just mock the limit or use a smaller large number.
    # However, the code has a hardcoded 10_000.

    forecaster = ForecasterEquivalentDate(
        offset=1, n_offsets=1, binner_kwargs={"n_bins": n_bins}
    )
    y = pd.Series(np.arange(12000, dtype=float))

    forecaster.fit(y=y, store_in_sample_residuals=True)

    for k, v in forecaster.in_sample_residuals_by_bin_.items():
        assert len(v) <= max_sample_per_bin


def test_binning_in_sample_residuals_total_sampling_limit():
    """
    Test that the total number of stored residuals is limited to 10,000.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    y = pd.Series(np.arange(15000, dtype=float))

    forecaster.fit(y=y, store_in_sample_residuals=True)

    assert len(forecaster.in_sample_residuals_) == 10000


def test_binning_in_sample_residuals_empty_bins_handling():
    """
    Test that empty bins are filled with a random sample of residuals.
    """
    # Create data where one bin will be empty
    # By default QuantileBinner uses quantiles, so it's hard to get an empty bin
    # unless we have duplicate values or extreme distributions.
    # Let's use 10 bins and very few data points.
    forecaster = ForecasterEquivalentDate(
        offset=1, n_offsets=1, binner_kwargs={"n_bins": 5}
    )
    y = pd.Series(
        np.array([1, 1, 1, 1, 10, 10, 10, 10], dtype=float)
    )  # Discrete values

    # Fit should result in some bins having 0 residuals initially before filling
    forecaster.fit(y=y, store_in_sample_residuals=True)

    for k, v in forecaster.in_sample_residuals_by_bin_.items():
        assert v.size > 0  # Should have been filled


def test_binning_in_sample_residuals_no_storage():
    """
    Test that intervals are stored even if residuals are not.
    """
    forecaster = ForecasterEquivalentDate(
        offset=1, n_offsets=1, binner_kwargs={"n_bins": 2}
    )
    y = pd.Series(np.arange(10, dtype=float))

    forecaster.fit(y=y, store_in_sample_residuals=False)

    assert hasattr(forecaster, "binner_intervals_")
    assert forecaster.binner_intervals_ is not None
    assert (
        not hasattr(forecaster, "in_sample_residuals_by_bin_")
        or forecaster.in_sample_residuals_by_bin_ is None
    )
    # in_sample_residuals_ is not stored when store_in_sample_residuals=False
    assert forecaster.in_sample_residuals_ is None
