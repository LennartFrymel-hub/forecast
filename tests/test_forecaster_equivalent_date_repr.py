# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_repr_not_fitted():
    """
    Test __repr__ output when forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(
        offset=7, n_offsets=1, forecaster_id="test_id"
    )
    repr_str = repr(forecaster)

    assert "ForecasterEquivalentDate" in repr_str
    assert "Offset: 7" in repr_str
    assert "Number of offsets: 1" in repr_str
    assert "Aggregation function: mean" in repr_str
    assert "Window size: 7" in repr_str
    assert "Series name: None" in repr_str
    assert "Training range: None" in repr_str
    assert "Training index type: None" in repr_str
    assert "Training index frequency: None" in repr_str
    assert "Forecaster id: test_id" in repr_str
    assert "spotforecast version:" in repr_str
    assert "Python version:" in repr_str


def test_repr_fitted():
    """
    Test __repr__ output when forecaster is fitted.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    data = pd.Series(
        data=np.arange(20),
        index=pd.date_range(start="2022-01-01", periods=20, freq="D"),
        name="my_series",
    )
    forecaster.fit(y=data)
    repr_str = repr(forecaster)

    assert "ForecasterEquivalentDate" in repr_str
    assert "Offset: 7" in repr_str
    assert "Series name: my_series" in repr_str
    assert "Training range: [Timestamp('2022-01-01 00:00:00')," in repr_str
    assert "Training index type: DatetimeIndex" in repr_str
    assert "Training index frequency: D" in repr_str
    assert "Last fit date:" in repr_str
    assert "spotforecast version:" in repr_str


def test_repr_header_length():
    """
    Test that the header length matches the class name length.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    repr_str = repr(forecaster)
    lines = repr_str.split("\n")
    header_border = lines[0].strip()
    class_name = lines[1].strip()

    assert len(header_border) == len(class_name)
    assert header_border == "=" * len("ForecasterEquivalentDate")
