# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import Day

from spotforecast2_safe.exceptions import MissingValuesWarning
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_predict_integer_offset_n_offsets_1():
    """
    Test prediction with an integer offset and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(offset=7, n_offsets=1)
    y = pd.Series(
        np.arange(14), index=pd.date_range("2022-01-01", periods=14, freq="D")
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=3)

    expected_index = pd.date_range("2022-01-15", periods=3, freq="D")
    expected_values = np.array([7, 8, 9])

    assert isinstance(predictions, pd.Series)
    pd.testing.assert_index_equal(predictions.index, expected_index)
    np.testing.assert_array_equal(predictions.to_numpy(), expected_values)
    assert predictions.name == "pred"


def test_predict_integer_offset_n_offsets_2():
    """
    Test prediction with an integer offset and n_offsets=2 (aggregation).
    """
    forecaster = ForecasterEquivalentDate(offset=7, n_offsets=2, agg_func=np.mean)
    y = pd.Series(
        np.arange(21), index=pd.date_range("2022-01-01", periods=21, freq="D")
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=3)

    # Offset 1: 14, 15, 16
    # Offset 2: 7, 8, 9
    # Mean: 10.5, 11.5, 12.5
    expected_values = np.array([10.5, 11.5, 12.5])

    np.testing.assert_array_equal(predictions.to_numpy(), expected_values)


def test_predict_dateoffset_n_offsets_1():
    """
    Test prediction with a pandas DateOffset and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(7), n_offsets=1)
    y = pd.Series(
        np.arange(14), index=pd.date_range("2022-01-01", periods=14, freq="D")
    )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=3)

    expected_index = pd.date_range("2022-01-15", periods=3, freq="D")
    expected_values = np.array([7, 8, 9])

    pd.testing.assert_index_equal(predictions.index, expected_index)
    np.testing.assert_array_equal(predictions.to_numpy(), expected_values)


def test_predict_dateoffset_no_equivalent_dates_raises_error():
    """
    Test ValueError is raised if no equivalent dates are found for DateOffset.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(10))
    # Fit with enough data
    y = pd.Series(
        np.arange(20), index=pd.date_range("2022-01-01", periods=20, freq="D")
    )
    forecaster.fit(y=y)

    # Use a custom last_window that doesn't contain the equivalent date
    custom_lw = pd.Series(
        np.arange(5), index=pd.date_range("2023-01-01", periods=5, freq="D")
    )

    msg = "All equivalent values are missing"
    with pytest.raises(ValueError, match=msg):
        # check_inputs=False allows using a shorter/mismatched last_window
        forecaster.predict(steps=1, last_window=custom_lw, check_inputs=False)


def test_predict_dateoffset_incomplete_offsets_raises_warning():
    """
    Test MissingValuesWarning is raised if some equivalent dates are missing.
    """
    forecaster = ForecasterEquivalentDate(offset=Day(7), n_offsets=2)
    # Fit with valid data
    y = pd.Series(
        np.arange(20), index=pd.date_range("2022-01-01", periods=20, freq="D")
    )
    forecaster.fit(y=y)

    # last_window for predict starts at 2022-01-10
    # Predict step 1 (2022-01-21):
    # Offset 1 (21-7=14): present in lw
    # Offset 2 (14-7=7): missing in lw (starts at 10)
    custom_lw = y.iloc[9:].copy()

    with pytest.warns(MissingValuesWarning, match="are calculated with less than"):
        forecaster.predict(steps=1, last_window=custom_lw, check_inputs=False)


def test_predict_last_window_override():
    """
    Test prediction using a custom last_window.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(14), index=pd.date_range("2022-01-01", periods=14, freq="D")
    )
    forecaster.fit(y=y)

    custom_lw = pd.Series(
        np.arange(10, 24), index=pd.date_range("2022-02-01", periods=14, freq="D")
    )
    predictions = forecaster.predict(steps=1, last_window=custom_lw)

    # Custom last_window ends at Feb 14.
    # Prediction for Feb 15 should be value from Feb 8 (idx 7 in lw, which is 17)
    assert predictions.iloc[0] == 17
    assert predictions.index[0] == pd.Timestamp("2022-02-15")


def test_predict_check_inputs_false():
    """
    Test predict when check_inputs=False (skips validation).
    """
    forecaster = ForecasterEquivalentDate(offset=1)
    y = pd.Series(np.arange(5))
    forecaster.fit(y=y)

    # Should work without error if checks are skipped even if freq is missing (as int offset allows it)
    predictions = forecaster.predict(steps=1, check_inputs=False)
    assert len(predictions) == 1
