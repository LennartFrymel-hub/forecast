from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_repr_basic():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    output = str(forecaster)
    # Check for key lines in the output
    assert "ForecasterRecursive" in output
    assert "Estimator: LinearRegression" in output
    assert "Lags:" in output
    assert "Window features:" in output
    assert "Window size:" in output
    assert "Series name:" in output
    assert "Exogenous included:" in output
    assert "Exogenous names:" in output
    assert "Transformer for y:" in output
    assert "Transformer for exog:" in output
    assert "Weight function included:" in output
    assert "Differentiation order:" in output
    assert "Training range:" in output
    assert "Training index type:" in output
    assert "Training index frequency:" in output
    assert "Estimator parameters:" in output
    assert "fit_kwargs:" in output
    assert "Creation date:" in output
    assert "Last fit date:" in output
    assert "spotforecast version:" in output
    assert "Python version:" in output
    assert "Forecaster id:" in output
