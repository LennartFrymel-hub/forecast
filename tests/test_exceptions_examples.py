# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings

import pytest

from spotforecast2_safe.exceptions import MissingValuesWarning


def test_missing_values_warning_example():
    """
    Test the example from MissingValuesWarning docstring.
    """
    with pytest.warns(
        MissingValuesWarning, match="Missing values detected in input data."
    ):
        warnings.warn("Missing values detected in input data.", MissingValuesWarning)
