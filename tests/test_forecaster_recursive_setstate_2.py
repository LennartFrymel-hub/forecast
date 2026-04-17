import pickle

from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_setstate_tags_created_on_unpickle():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    # Remove __spotforecast_tags__ to simulate old pickle
    if hasattr(forecaster, "__spotforecast_tags__"):
        del forecaster.__spotforecast_tags__
    pickled = pickle.dumps(forecaster)
    unpickled = pickle.loads(pickled)
    assert hasattr(unpickled, "__spotforecast_tags__")
    tags = unpickled.__spotforecast_tags__
    assert tags["library"] == "spotforecast"
    assert tags["forecaster_name"] == "ForecasterRecursive"
    assert tags["forecasting_strategy"] == "recursive"
    assert tags["supports_exog"] is True


def test_setstate_tags_not_overwritten():
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.__spotforecast_tags__ = {"library": "custom", "extra": "value"}
    pickled = pickle.dumps(forecaster)
    unpickled = pickle.loads(pickled)
    # Should not overwrite tags if present
    assert unpickled.__spotforecast_tags__["library"] == "custom"
    assert unpickled.__spotforecast_tags__["extra"] == "value"
