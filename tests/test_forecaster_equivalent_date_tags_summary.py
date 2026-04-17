# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_get_tags():
    """
    Test get_tags method.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    tags = forecaster.get_tags()

    assert isinstance(tags, dict)
    assert tags["library"] == "spotforecast"
    assert tags["forecaster_name"] == "ForecasterEquivalentDate"
    assert tags["forecaster_task"] == "regression"
    assert tags["forecasting_scope"] == "single-series"
    assert tags["supports_probabilistic"] is True
    assert "conformal" in tags["probabilistic_methods"]


def test_summary(capsys):
    """
    Test summary method.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    forecaster.summary()
    captured = capsys.readouterr()

    assert "ForecasterEquivalentDate" in captured.out
    assert "Offset: 7" in captured.out
    assert "Aggregation function: mean" in captured.out
    assert "Window size: 7" in captured.out
    assert "Training range: None" in captured.out


def test_summary_fitted(capsys):
    """
    Test summary method after fitting.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    y = pd.Series(
        np.arange(14, dtype=float),
        index=pd.date_range("2022-01-01", periods=14, freq="D"),
    )
    forecaster.fit(y=y)
    forecaster.summary()
    captured = capsys.readouterr()

    assert "ForecasterEquivalentDate" in captured.out
    assert (
        "Training range: [Timestamp('2022-01-01 00:00:00'), Timestamp('2022-01-14 00:00:00')]"
        in captured.out
    )
    assert "Training index frequency: D" in captured.out
