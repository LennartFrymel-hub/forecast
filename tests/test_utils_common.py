import warnings

import pytest

from spotforecast2_safe.model_selection.utils_common import (
    OneStepAheadValidationWarning,
)


def test_is_subclass_of_userwarning():
    assert issubclass(OneStepAheadValidationWarning, UserWarning)


def test_message_is_stored():
    msg = "Test warning"
    w = OneStepAheadValidationWarning(msg)
    assert w.message == msg


def test_str_returns_message_and_suppression():
    msg = "Test warning"
    w = OneStepAheadValidationWarning(msg)
    s = str(w)
    assert msg in s
    assert "You can suppress this warning using:" in s


def test_warn_emits_warning_and_message():
    msg = "Test warning"
    with pytest.warns(OneStepAheadValidationWarning) as record:
        warnings.warn(msg, OneStepAheadValidationWarning)
    assert any(msg in str(w.message) for w in record)


def test_warning_can_be_suppressed():
    msg = "Test warning"
    warnings.simplefilter("ignore", category=OneStepAheadValidationWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.warn(msg, OneStepAheadValidationWarning)
        assert len(w) == 0
    warnings.resetwarnings()
