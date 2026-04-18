import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


@pytest.fixture
def forecaster():
    return ForecasterRecursive(
        estimator=LinearRegression(), lags=3, forecaster_id="html_test"
    )


def test_repr_html_contains_general_info(forecaster):
    y = pd.Series(np.random.randn(20), name="y")
    forecaster.fit(y)
    html = forecaster._repr_html_()
    assert "<strong>Estimator:</strong> LinearRegression" in html
    assert "<strong>Lags:</strong> [1 2 3]" in html
    assert "<strong>Window size:</strong> 3" in html
    assert "<strong>Series name:</strong> y" in html
    assert "<strong>Exogenous included:</strong> False" in html
    assert "<strong>Weight function included:</strong> False" in html
    assert "<strong>Differentiation order:</strong> None" in html
    assert "<strong>Creation date:</strong>" in html
    assert "<strong>Last fit date:</strong>" in html
    assert "<strong>spotforecast version:</strong>" in html
    assert "<strong>Python version:</strong>" in html
    assert "<strong>Forecaster id:</strong> html_test" in html


def test_repr_html_exog_section():
    y = pd.Series(np.random.randn(20), name="target")
    exog = pd.DataFrame({"temp": np.random.randn(20)}, index=y.index)
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, forecaster_id="html_exog"
    )
    forecaster.fit(y, exog)
    html = forecaster._repr_html_()
    assert "<summary>Exogenous Variables</summary>" in html
    assert "temp" in html
    assert "<strong>Exogenous included:</strong> True" in html


def test_repr_html_transformers():
    from sklearn.preprocessing import StandardScaler

    y = pd.Series(np.random.randn(20), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        forecaster_id="html_transformer",
    )
    forecaster.fit(y)
    html = forecaster._repr_html_()
    assert "<strong>Transformer for y:</strong> StandardScaler" in html
    assert "<strong>Transformer for exog:</strong> StandardScaler" in html
    assert "<strong>Forecaster id:</strong> html_transformer" in html
