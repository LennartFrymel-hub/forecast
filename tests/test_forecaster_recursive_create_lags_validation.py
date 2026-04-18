import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive


def test_create_lags_basic():
    """
    Test _create_lags with basic input.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = np.arange(10)
    X_data, y_data = forecaster._create_lags(y=y)

    # window_size is 3
    # X_data should have 7 rows (10 - 3) and 3 columns (lags)
    assert X_data.shape == (7, 3)
    assert y_data.shape == (7,)

    # Lag 1 is t-1, Lag 2 is t-2, Lag 3 is t-3
    # Row 0 of X_data should be [2, 1, 0] (for y[3]=3)
    np.testing.assert_array_equal(X_data[0], [2, 1, 0])
    assert y_data[0] == 3


def test_create_lags_as_pandas():
    """
    Test _create_lags with X_as_pandas=True.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = np.arange(10)
    train_index = pd.RangeIndex(start=3, stop=10)
    X_data, y_data = forecaster._create_lags(
        y=y, X_as_pandas=True, train_index=train_index
    )

    assert isinstance(X_data, pd.DataFrame)
    assert X_data.shape == (7, 3)
    pd.testing.assert_index_equal(X_data.index, train_index)
    assert list(X_data.columns) == forecaster.lags_names


def test_create_lags_value_error_missing_index():
    """
    Test _create_lags raises ValueError when X_as_pandas=True but train_index is None.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = np.arange(10)
    with pytest.raises(
        ValueError, match="If `X_as_pandas` is True, `train_index` must be provided"
    ):
        forecaster._create_lags(y=y, X_as_pandas=True, train_index=None)


def test_create_lags_value_error_insufficient_y():
    """
    Test _create_lags raises ValueError when y is too short.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    y = np.arange(3)  # window_size is 3
    with pytest.raises(
        ValueError, match="Length of `y` must be greater than the maximum window size"
    ):
        forecaster._create_lags(y=y)


def test_create_train_X_y_safety_validation():
    """
    Test _create_train_X_y with exogenous variables and check the safety-critical validation.
    """
    y = pd.Series(np.arange(20), name="y")
    # window_size = 3
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Case 1: exog length matches y length (20)
    exog_full = pd.DataFrame({"exog": np.random.randn(20)}, index=y.index)
    X_train, y_train, *_ = forecaster._create_train_X_y(y=y, exog=exog_full)
    assert len(X_train) == 17  # 20 - 3

    # Case 2: exog length matches training size (17)
    exog_trimmed = pd.DataFrame({"exog": np.random.randn(17)}, index=y.index[3:])
    X_train, y_train, *_ = forecaster._create_train_X_y(y=y, exog=exog_trimmed)
    assert len(X_train) == 17

    # Case 3: Case that should fail (length 18)
    exog_fail = pd.DataFrame({"exog": np.random.randn(18)}, index=y.index[2:])
    with pytest.raises(ValueError, match="Length mismatch for exogenous variables"):
        forecaster._create_train_X_y(y=y, exog=exog_fail)


def test_differentiation_initialization():
    """
    Test that differentiation is correctly initialized in ForecasterRecursive.
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    assert forecaster.differentiation == 1
    # window_size should be lags (3) + differentiation (1) == 4
    assert forecaster.window_size == 4
    assert forecaster.differentiator is not None


class MockWindowFeature:
    def __init__(self, name, window_size=3):
        self.features_names = [name]
        self.window_sizes = window_size

    def transform_batch(self, y):
        # Return a DataFrame with shifted values to simulate window features
        df = pd.DataFrame(
            {self.features_names[0]: y.iloc[self.window_sizes :].values},
            index=y.index[self.window_sizes :],
        )
        return df

    def transform(self, y):
        return np.mean(y)


def test_create_window_features_basic():
    """
    Test _create_window_features with basic input.
    """
    y = pd.Series(np.arange(10), index=pd.RangeIndex(10), name="y")
    wf = MockWindowFeature(name="rolling_mean", window_size=3)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), window_features=[wf])

    train_index = y.index[3:]
    X_train_window_features, feature_names = forecaster._create_window_features(
        y=y, train_index=train_index, X_as_pandas=True
    )

    assert len(X_train_window_features) == 1
    assert isinstance(X_train_window_features[0], pd.DataFrame)
    assert feature_names == ["rolling_mean"]
    pd.testing.assert_index_equal(X_train_window_features[0].index, train_index)


def test_create_window_features_type_error():
    """
    Test _create_window_features raises TypeError if transform_batch doesn't return DataFrame.
    """

    class BadWindowFeature:
        def __init__(self):
            self.features_names = ["bad"]
            self.window_sizes = 3

        def transform_batch(self, y):
            return np.array([1, 2, 3])

        def transform(self, y):
            return np.mean(y)

    y = pd.Series(np.arange(10))
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), window_features=[BadWindowFeature()]
    )
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        forecaster._create_window_features(y=y, train_index=y.index[3:])


def test_create_window_features_value_error_length():
    """
    Test _create_window_features raises ValueError if length mismatch.
    """

    class ShortWindowFeature:
        def __init__(self):
            self.features_names = ["short"]
            self.window_sizes = 3

        def transform_batch(self, y):
            return pd.DataFrame({"short": [1, 2, 3]}, index=y.index[:3])

        def transform(self, y):
            return np.mean(y)

    y = pd.Series(np.arange(10))
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), window_features=[ShortWindowFeature()]
    )
    with pytest.raises(
        ValueError, match="must return a DataFrame with the same number of rows"
    ):
        forecaster._create_window_features(y=y, train_index=y.index[3:])


def test_create_window_features_value_error_index():
    """
    Test _create_window_features raises ValueError if index mismatch.
    """

    class WrongIndexWindowFeature:
        def __init__(self):
            self.features_names = ["wrong"]
            self.window_sizes = 3

        def transform_batch(self, y):
            return pd.DataFrame({"wrong": np.arange(7)}, index=pd.RangeIndex(10, 17))

        def transform(self, y):
            return np.mean(y)

    y = pd.Series(np.arange(10))
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), window_features=[WrongIndexWindowFeature()]
    )
    with pytest.raises(ValueError, match="must return a DataFrame with the same index"):
        forecaster._create_window_features(y=y, train_index=y.index[3:])


def test_data_transformation_warning():
    """
    Test that DataTransformationWarning is raised when differentiation is used in create_predict_inputs.
    """
    from spotforecast2_safe.exceptions import DataTransformationWarning

    y = pd.Series(np.arange(30), name="y")
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, differentiation=1
    )
    forecaster.fit(y=y)

    with pytest.warns(
        DataTransformationWarning, match="The output matrix is in the transformed scale"
    ):
        forecaster._create_predict_inputs(steps=5)


def test_residuals_usage_warning():
    """
    Test that ResidualsUsageWarning is raised when set_out_sample_residuals has empty bins.
    """
    from spotforecast2_safe.exceptions import ResidualsUsageWarning

    y = pd.Series(np.arange(100), name="y")
    # Binner with 10 bins
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, binner_kwargs={"n_bins": 10}
    )
    forecaster.fit(y=y)

    # y_true and y_pred that only fall in some bins
    y_true = np.arange(10)
    y_pred = np.arange(10)

    with pytest.warns(
        ResidualsUsageWarning,
        match="The following bins have no out of sample residuals",
    ):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
