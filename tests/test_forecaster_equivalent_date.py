import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.exceptions import MissingValuesWarning
from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_init_invalid_offset():
    """
    Test initialization with invalid offset type.
    """
    with pytest.raises(TypeError, match="`offset` must be an integer greater than 0"):
        ForecasterEquivalentDate(offset="invalid")


def test_fit_invalid_y_type():
    """
    Test fit with invalid y type.
    """
    forecaster = ForecasterEquivalentDate(offset=1)
    with pytest.raises(TypeError, match="`y` must be a pandas Series"):
        forecaster.fit(y=[1, 2, 3])


def test_fit_dateoffset_no_datetimeindex():
    """
    Test fit with DateOffset but no DatetimeIndex in y.
    """
    forecaster = ForecasterEquivalentDate(offset=pd.DateOffset(days=1))
    y = pd.Series(np.arange(10))
    with pytest.raises(TypeError, match="If `offset` is a pandas DateOffset"):
        forecaster.fit(y=y)


def test_predict_simple_integer_offset():
    """
    Test predict with simple integer offset.
    """
    y = pd.Series(np.arange(20), name="y")
    forecaster = ForecasterEquivalentDate(offset=5)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=3)

    expected_index = pd.RangeIndex(start=20, stop=23)
    expected_values = np.array(
        [15, 16, 17]
    )  # Values from 5 steps ago: 20-5=15, 21-5=16, 22-5=17

    pd.testing.assert_series_equal(
        predictions, pd.Series(expected_values, index=expected_index, name="pred")
    )


def test_predict_dateoffset():
    """
    Test predict with DateOffset.
    """
    y = pd.Series(
        np.arange(20),
        index=pd.date_range(start="2020-01-01", periods=20, freq="D"),
        name="y",
    )
    # Offset 1 week
    forecaster = ForecasterEquivalentDate(offset=pd.DateOffset(weeks=1))
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=3)

    expected_index = pd.date_range(start="2020-01-21", periods=3, freq="D")
    # Values from 1 week ago (7 days)
    # 2020-01-21 -> 2020-01-14 (index 13, value 13)
    # 2020-01-22 -> 2020-01-15 (index 14, value 14)
    # 2020-01-23 -> 2020-01-16 (index 15, value 15)
    expected_values = np.array(
        [13, 14, 15], dtype=float
    )  # Note: floats because of potential NaNs in reindexing logic usually

    pd.testing.assert_series_equal(
        predictions, pd.Series(expected_values, index=expected_index, name="pred")
    )


def test_predict_aggregation():
    """
    Test predict with multiple offsets and aggregation.
    """
    y = pd.Series(np.arange(20), name="y")
    # offset=2, n_offsets=2, agg_func=mean
    # For step 1 (index 20): values at (20-2)=18 and (20-4)=16. Mean = 17.
    # For step 2 (index 21): values at (21-2)=19 and (21-4)=17. Mean = 18.
    forecaster = ForecasterEquivalentDate(offset=2, n_offsets=2, agg_func=np.mean)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=2)

    expected_values = np.array([17.0, 18.0])
    expected_index = pd.RangeIndex(start=20, stop=22)

    pd.testing.assert_series_equal(
        predictions, pd.Series(expected_values, index=expected_index, name="pred")
    )


def test_predict_interval_conformal():
    """
    Test predict_interval with conformal method.
    """
    y = pd.Series(np.random.normal(loc=10, scale=1, size=100), name="y")
    forecaster = ForecasterEquivalentDate(offset=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)

    intervals = forecaster.predict_interval(steps=5, interval=[5, 95])

    assert isinstance(intervals, pd.DataFrame)
    assert intervals.columns.tolist() == ["pred", "lower_bound", "upper_bound"]
    assert len(intervals) == 5
    assert (intervals["upper_bound"] >= intervals["lower_bound"]).all()


def test_missing_values_warning_dateoffset():
    """
    Test that MissingValuesWarning is raised when DateOffset results in missing values.
    We need n_offsets > 1 to allow partial matches (otherwise it raises ValueError).
    And we use check_inputs=False to bypass the strict window_size check, simulating
    a scenario where we might have just enough data for recent offsets but not older ones.
    """
    y = pd.Series(
        np.arange(100), index=pd.date_range(start="2020-01-01", periods=100, freq="D")
    )
    # Offset 10 days, n_offsets=2. Logic needs lookback of 10 and 20 days.
    forecaster = ForecasterEquivalentDate(offset=pd.DateOffset(days=10), n_offsets=2)
    forecaster.fit(y=y)

    # Create a short last_window that has data for 10 days ago but NOT 20 days ago.
    # Current prediction time T.
    # We need T-10 to be in last_window.
    # We need T-20 to NOT be in last_window.
    # If last_window covers [T-15, T], then T-10 is there, T-20 is missing.

    last_window = y.iloc[-15:]  # Last 15 days

    with pytest.warns(MissingValuesWarning, match="are calculated with less than"):
        forecaster.predict(steps=1, last_window=last_window, check_inputs=False)


def test_repr():
    """
    Test __repr__ output.
    """
    forecaster = ForecasterEquivalentDate(offset=5)
    repr_str = repr(forecaster)
    assert "ForecasterEquivalentDate" in repr_str
    assert "Offset: 5" in repr_str
    assert "Window size: 5" in repr_str


def test_predict_with_last_window():
    """
    Test predict using an external last_window.
    """
    y_train = pd.Series(np.arange(20))
    forecaster = ForecasterEquivalentDate(offset=1)
    forecaster.fit(y=y_train)

    # last_window continues the sequence: 20, 21, 22, 23, 24
    last_window = pd.Series(np.arange(20, 25), index=pd.RangeIndex(start=20, stop=25))

    predictions = forecaster.predict(steps=3, last_window=last_window)

    expected_values = np.array([24, 24, 24])  # offset=1 means we repeat the last value

    pd.testing.assert_series_equal(
        predictions,
        pd.Series(expected_values, index=pd.RangeIndex(start=25, stop=28), name="pred"),
    )
