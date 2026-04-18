# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
import pandas as pd

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


def test_repr_html_not_fitted():
    """
    Test _repr_html_ output when forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(
        offset=7, n_offsets=1, forecaster_id="test_id"
    )
    html_str = forecaster._repr_html_()

    assert "ForecasterEquivalentDate" in html_str
    assert "Offset:</strong> 7" in html_str
    assert "Number of offsets:</strong> 1" in html_str
    assert "Aggregation function:</strong> mean" in html_str
    assert "Window size:</strong> 7" in html_str
    assert "Training range:</strong> Not fitted" in html_str
    assert "Training index type:</strong> Not fitted" in html_str
    assert "Training index frequency:</strong> Not fitted" in html_str
    assert "Forecaster id:</strong> test_id" in html_str
    assert "spotforecast version:</strong>" in html_str
    assert "Python version:</strong>" in html_str
    # Check for closing tags
    assert "</ul>" in html_str
    assert "</details>" in html_str
    assert html_str.count("</details>") == 2
    assert html_str.endswith("</div>")


def test_repr_html_fitted():
    """
    Test _repr_html_ output when forecaster is fitted.
    """
    forecaster = ForecasterEquivalentDate(offset=7)
    data = pd.Series(
        data=np.arange(20),
        index=pd.date_range(start="2022-01-01", periods=20, freq="D"),
        name="my_series",
    )
    forecaster.fit(y=data)
    html_str = forecaster._repr_html_()

    assert "ForecasterEquivalentDate" in html_str
    assert "Offset:</strong> 7" in html_str
    assert "Training range:</strong> [Timestamp('2022-01-01 00:00:00')," in html_str
    assert "Training index type:</strong> DatetimeIndex" in html_str
    assert "Training index frequency:</strong> D" in html_str
    assert "Last fit date:</strong>" in html_str
    assert "spotforecast version:</strong>" in html_str
    assert html_str.count("</details>") == 2
    assert html_str.strip().endswith("</div>")
