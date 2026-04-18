import pandas as pd
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.model_selection import TimeSeriesFold
from spotforecast2_safe.model_selection.validation import _backtesting_forecaster


def test_backtesting_forecaster_example():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    cv = TimeSeriesFold(steps=2, initial_train_size=6)
    metric_values, backtest_predictions = _backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_squared_error",
        show_progress=False,
    )
    # Check output types and shapes
    assert isinstance(metric_values, pd.DataFrame)
    assert isinstance(backtest_predictions, pd.DataFrame)
    assert not metric_values.empty
    assert not backtest_predictions.empty
