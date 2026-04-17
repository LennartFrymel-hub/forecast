import pytest

from spotforecast2_safe.data.data import Period


def test_period_instantiation():
    """Test correctly instantiating Period."""
    period = Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))
    assert period.name == "hour"
    assert period.n_periods == 24
    assert period.column == "hour"
    assert period.input_range == (0, 23)


def test_period_invalid_n_periods():
    """Test that non-positive n_periods raises ValueError."""
    with pytest.raises(ValueError, match="n_periods must be positive"):
        Period(name="test", n_periods=0, column="test", input_range=(0, 10))

    with pytest.raises(ValueError, match="n_periods must be positive"):
        Period(name="test", n_periods=-1, column="test", input_range=(0, 10))


def test_period_invalid_input_range():
    """Test that invalid input_range raises ValueError."""
    with pytest.raises(
        ValueError, match=r"input_range\[0\] must be less than input_range\[1\]"
    ):
        Period(name="test", n_periods=10, column="test", input_range=(10, 10))

    with pytest.raises(
        ValueError, match=r"input_range\[0\] must be less than input_range\[1\]"
    ):
        Period(name="test", n_periods=10, column="test", input_range=(11, 10))


def test_period_docstring_example():
    """Test the example provided in the docstring."""
    period = Period(name="hour", n_periods=24, column="hour", input_range=(0, 23))
    assert period.name == "hour"
