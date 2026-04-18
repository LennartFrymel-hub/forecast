import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.repeating_basis_function import (
    RepeatingBasisFunction,
)


def test_rbf_transform_dataframe():
    """Test RBF transformation with a DataFrame input."""
    X = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
    rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))
    features = rbf.fit_transform(X)

    assert features.shape == (5, 4)
    assert np.all(features >= 0)
    assert np.all(features <= 1)
    # At hour 0, the first basis function should be at its maximum
    assert features[0, 0] > features[0, 1]
    assert features[0, 0] > features[0, 2]
    assert features[0, 0] > features[0, 3]


def test_rbf_transform_series():
    """Test RBF transformation with a Series input."""
    X = pd.Series([0, 6, 12, 18, 23], name="hour")
    rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))
    features = rbf.transform(X)

    assert features.shape == (5, 4)


def test_rbf_missing_column():
    """Test that missing column raises ValueError."""
    X = pd.DataFrame({"day": [1, 2, 3]})
    rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))

    with pytest.raises(ValueError, match="Column hour not found in input"):
        rbf.transform(X)


def test_rbf_docstring_example():
    """Test the example provided in the docstring."""
    # The docstring example:
    # >>> X = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
    # >>> rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))
    # >>> features = rbf.fit_transform(X)
    # >>> features.shape
    # (5, 4)

    X = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
    rbf = RepeatingBasisFunction(n_periods=4, column="hour", input_range=(0, 23))
    features = rbf.fit_transform(X)
    assert features.shape == (5, 4)


def test_rbf_periodicity():
    """Test that RBF handles periodicity correctly (wraparound)."""
    # Use 2 periods: one at 0, one at 0.5 (normalized)
    rbf = RepeatingBasisFunction(n_periods=2, column="val", input_range=(0, 10))

    # Value 0 and 10 should produce very similar features because of wraparound
    X_0 = pd.DataFrame({"val": [0]})
    X_10 = pd.DataFrame({"val": [10]})

    feat_0 = rbf.transform(X_0)
    feat_10 = rbf.transform(X_10)

    np.testing.assert_allclose(feat_0, feat_10, atol=1e-5)
