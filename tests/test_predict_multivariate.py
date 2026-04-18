import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.forecaster.utils import predict_multivariate


def test_predict_multivariate_recursive():
    """
    Test predict_multivariate with ForecasterRecursive.
    """
    y1 = pd.Series(np.arange(10, dtype=float), name="target1")
    y2 = pd.Series(np.arange(10, dtype=float) * 2, name="target2")

    f1 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f2 = ForecasterRecursive(estimator=LinearRegression(), lags=2)

    f1.fit(y=y1)
    f2.fit(y=y2)

    forecasters = {"target1": f1, "target2": f2}
    predictions = predict_multivariate(forecasters, steps_ahead=3)

    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["target1", "target2"]
    assert len(predictions) == 3
    # Linear trend: 10, 11, 12... for target1; 20, 22, 24... for target2
    np.testing.assert_allclose(predictions["target1"], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(predictions["target2"], [20.0, 22.0, 24.0])


def test_predict_multivariate_single_target():
    """
    Test predict_multivariate with a single target.
    """
    y = pd.Series(np.arange(10, dtype=float), name="target")
    f = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f.fit(y=y)

    forecasters = {"target": f}
    predictions = predict_multivariate(forecasters, steps_ahead=2)

    assert len(predictions) == 2
    assert list(predictions.columns) == ["target"]


def test_predict_multivariate_empty_dict():
    """
    Test predict_multivariate with an empty dictionary.
    """
    predictions = predict_multivariate({}, steps_ahead=5)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.empty


def test_predict_multivariate_mismatched_index():
    """
    Test predict_multivariate with targets having different initial indices.
    The output should still align them if possible, but usually forecasters
    predict following their own window. concat(axis=1) handles this.
    """
    y1 = pd.Series(np.arange(10, dtype=float), index=pd.RangeIndex(0, 10), name="t1")
    y2 = pd.Series(np.arange(10, dtype=float), index=pd.RangeIndex(10, 20), name="t2")

    f1 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f2 = ForecasterRecursive(estimator=LinearRegression(), lags=2)

    f1.fit(y=y1)
    f2.fit(y=y2)

    forecasters = {"t1": f1, "t2": f2}
    predictions = predict_multivariate(forecasters, steps_ahead=2)

    # f1 predicts for 10, 11
    # f2 predicts for 20, 21
    # pd.concat(axis=1) will create a 4-row DataFrame with NaNs
    assert len(predictions) == 4
    assert "t1" in predictions.columns
    assert "t2" in predictions.columns
    assert predictions.loc[10, "t1"] == 10.0
    assert np.isnan(predictions.loc[10, "t2"])
    assert predictions.loc[20, "t2"] == 10.0
    assert np.isnan(predictions.loc[20, "t1"])


def test_predict_multivariate_zero_steps():
    """
    Test predict_multivariate with zero steps.
    Expect ValueError from validation.
    """
    y = pd.Series(np.arange(10, dtype=float), name="target")
    f = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f.fit(y=y)

    forecasters = {"target": f}
    with pytest.raises(
        ValueError, match="`steps` must be an integer greater than or equal to 1"
    ):
        predict_multivariate(forecasters, steps_ahead=0)
