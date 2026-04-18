# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from spotforecast2_safe.utils import date_to_index_position


def test_date_to_index_position_prediction():
    """
    Test date_to_index_position for prediction (steps).
    """
    index = pd.date_range("2024-01-01", periods=5, freq="D")

    # Test with date later than last index
    steps = date_to_index_position(index, "2024-01-08", method="prediction")
    assert steps == 3  # 6, 7, 8 (3 steps)

    # Test with Timestamp
    steps = date_to_index_position(
        index, pd.Timestamp("2024-01-06"), method="prediction"
    )
    assert steps == 1

    # Test with int
    assert date_to_index_position(index, 5, method="prediction") == 5


def test_date_to_index_position_validation():
    """
    Test date_to_index_position for validation (initial_train_size).
    """
    index = pd.date_range("2024-01-01", periods=10, freq="D")

    # Test with date within index
    size = date_to_index_position(index, "2024-01-05", method="validation")
    assert size == 5  # Includes first 5 dates

    # Test with first date
    size = date_to_index_position(index, "2024-01-01", method="validation")
    assert size == 1


def test_date_to_index_position_errors():
    """
    Test error cases for date_to_index_position.
    """
    index = pd.date_range("2024-01-01", periods=5, freq="D")

    # Date earlier than last index in prediction
    with pytest.raises(ValueError, match="must be greater than the last date"):
        date_to_index_position(index, "2024-01-04", method="prediction")

    # RangeIndex but date input
    range_index = pd.RangeIndex(0, 5)
    with pytest.raises(TypeError, match="Index must be a pandas DatetimeIndex"):
        date_to_index_position(range_index, "2024-01-08")

    # Unsupported type
    with pytest.raises(
        TypeError, match="must be an integer, string, or pandas Timestamp"
    ):
        date_to_index_position(index, [1, 2])
