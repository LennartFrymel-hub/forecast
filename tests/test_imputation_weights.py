import pytest
import pandas as pd
import numpy as np
import pickle
from spotforecast2_safe.preprocessing.imputation import (
    WeightFunction,
    get_missing_weights,
    custom_weights,
)


def test_weight_function_basic():
    """Verify WeightFunction basic functionality and pickling."""
    index = pd.Index([10, 11, 12, 13])
    weights = pd.Series([1.0, 1.0, 0.0, 1.0], index=index)
    wf = WeightFunction(weights)

    # Check single value access
    assert wf(10) == 1.0
    assert wf(12) == 0.0

    # Check array access
    res = wf(pd.Index([10, 11]))
    np.testing.assert_array_equal(res, np.array([1.0, 1.0]))

    # Check serialization
    pickled = pickle.dumps(wf)
    unpickled = pickle.loads(pickled)
    assert unpickled(12) == 0.0
    np.testing.assert_array_equal(unpickled(pd.Index([10, 13])), np.array([1.0, 1.0]))


def test_get_missing_weights_gaps():
    """Verify that gaps result in zero weights according to window_size."""
    # Create data with a gap
    # Index 0-9: good
    # Index 5: missing
    # window_size = 2
    # Expected bad indices: 5, 6, 7 (total 3 points including original gap and window_size after)
    data = pd.DataFrame({"y": range(10)}, index=pd.RangeIndex(0, 10))
    data.loc[5, "y"] = np.nan

    filled_data, weights = get_missing_weights(data, window_size=2)

    # Check that data is filled (using ffill/bfill)
    assert not filled_data.isnull().any().any()
    assert filled_data.loc[5, "y"] == 4.0  # ffill from 4

    # Check weights
    # weights[5] = 0 (missing)
    # weights[6] = 0 (within window_size=2 after gap)
    # weights[7] = 0 (within window_size=2 after gap)
    # weights[8] = 1 (outside window_size)
    assert weights.loc[4] == 1.0
    assert weights.loc[5] == 0.0
    assert weights.loc[6] == 0.0
    assert weights.loc[7] == 0.0
    assert weights.loc[8] == 1.0
    assert weights.loc[0] == 1.0


def test_get_missing_weights_multiple_gaps():
    """Verify multiple gaps with overlapping windows."""
    data = pd.DataFrame({"y": range(20)}, index=pd.RangeIndex(0, 20))
    data.loc[5, "y"] = np.nan
    data.loc[8, "y"] = np.nan

    # window_size = 3
    # Gap 1 (5): affects 5, 6, 7, 8
    # Gap 2 (8): affects 8, 9, 10, 11
    # Total affected: 5, 6, 7, 8, 9, 10, 11
    filled_data, weights = get_missing_weights(data, window_size=3)

    affected = [5, 6, 7, 8, 9, 10, 11]
    for i in range(20):
        if i in affected:
            assert weights.loc[i] == 0.0, f"Index {i} should have weight 0"
        else:
            assert weights.loc[i] == 1.0, f"Index {i} should have weight 1"


def test_custom_weights_index_not_found():
    """Verify that ValueError is raised if index is missing from weights_series."""
    weights = pd.Series([1.0], index=[0])

    with pytest.raises(ValueError, match="Index not found in weights_series"):
        custom_weights(1, weights)

    with pytest.raises(ValueError, match="Index not found in weights_series"):
        custom_weights(pd.Index([0, 1]), weights)


def test_imputation_empty_data():
    """Verify checks for empty data and invalid window_size."""
    with pytest.raises(ValueError, match="Input data is empty"):
        get_missing_weights(pd.DataFrame())

    data = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        get_missing_weights(data, window_size=0)

    with pytest.raises(
        ValueError, match="window_size must be smaller than the number of rows"
    ):
        get_missing_weights(data, window_size=3)
