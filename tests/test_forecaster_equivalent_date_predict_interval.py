# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import Day

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_predict_interval_conformal_integer_offset_binned_residuals():
    """
    Test predict_interval with conformal method, integer offset, and binned residuals.
    """
    forecaster = ForecasterEquivalentDate(
        offset=7, n_offsets=1, binner_kwargs={"n_bins": 2}
    )
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y, store_in_sample_residuals=True)

    predictions = forecaster.predict_interval(
        steps=3, interval=[5, 95], use_binned_residuals=True
    )

    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["pred", "lower_bound", "upper_bound"]
    assert len(predictions) == 3
    assert not predictions.isnull().any().any()
    # Check that upper_bound is >= pred and lower_bound is <= pred
    assert (predictions["upper_bound"] >= predictions["pred"]).all()
    assert (predictions["lower_bound"] <= predictions["pred"]).all()


def test_predict_interval_conformal_dateoffset_global_residuals():
    """
    Test predict_interval with conformal method, DateOffset, and global residuals.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(7), n_offsets=1)
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y, store_in_sample_residuals=True)

    predictions = forecaster.predict_interval(
        steps=3, interval=0.9, use_binned_residuals=False
    )

    assert len(predictions) == 3
    assert (predictions["upper_bound"] >= predictions["pred"]).all()


def test_predict_interval_conformal_out_sample_residuals():
    """
    Test predict_interval using out-of-sample residuals.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y)

    # Set mock out-of-sample residuals
    out_sample_y_true = np.array([10, 11, 12])
    out_sample_y_pred = np.array([10.5, 10.8, 12.2])
    forecaster.set_out_sample_residuals(
        y_true=out_sample_y_true, y_pred=out_sample_y_pred
    )

    predictions = forecaster.predict_interval(
        steps=2, use_in_sample_residuals=False, use_binned_residuals=False
    )

    assert len(predictions) == 2
    assert forecaster.out_sample_residuals_ is not None


def test_predict_interval_unsupported_method_raises_error():
    """
    Test ValueError is raised for unsupported method.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(np.arange(10, dtype=float))
    forecaster.fit(y=y)

    with pytest.raises(ValueError, match="Method 'invalid' is not supported"):
        forecaster.predict_interval(steps=1, method="invalid")


def test_predict_interval_asymmetric_interval_raises_error():
    """
    Test ValueError is raised for asymmetric interval in conformal method.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(np.arange(10, dtype=float))
    # Must store residuals to reach the interval symmetry check
    forecaster.fit(y=y, store_in_sample_residuals=True)

    # [5, 90] is not symmetric around 50
    with pytest.raises(ValueError, match="must be symmetric"):
        forecaster.predict_interval(steps=1, interval=[5, 90])
