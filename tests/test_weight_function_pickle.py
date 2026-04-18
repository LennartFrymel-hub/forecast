"""Test WeightFunction class pickling and integration with ForecasterRecursive."""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import WeightFunction


class TestWeightFunction:
    """Test the WeightFunction class."""

    def test_weight_function_callable(self):
        """Test that WeightFunction is callable and returns correct weights."""
        weights = pd.Series(
            [1.0, 0.9, 0.8, 0.7], index=pd.date_range("2024-01-01", periods=4, freq="h")
        )
        weight_func = WeightFunction(weights)

        # Test single index
        result = weight_func(weights.index[0])
        assert result == 1.0

        # Test multiple indices
        result = weight_func(weights.index[:2])
        np.testing.assert_array_equal(result, np.array([1.0, 0.9]))

    def test_weight_function_pickle(self):
        """Test that WeightFunction can be pickled and unpickled."""
        weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights)

        # Pickle and unpickle
        pickled = pickle.dumps(weight_func)
        unpickled = pickle.loads(pickled)

        # Verify it still works
        result = unpickled(pd.Index([0, 1]))
        np.testing.assert_array_equal(result, np.array([1.0, 0.9]))

    def test_weight_function_joblib(self):
        """Test that WeightFunction can be saved and loaded with joblib."""
        weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
            temp_file = f.name

        try:
            # Save and load with joblib
            dump(weight_func, temp_file)
            loaded = load(temp_file)

            # Verify it still works
            result = loaded(pd.Index([0, 1]))
            np.testing.assert_array_equal(result, np.array([1.0, 0.9]))
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_forecaster_with_weight_function_pickle(self):
        """Test that ForecasterRecursive with WeightFunction can be pickled."""
        weights = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=range(5))
        weight_func = WeightFunction(weights)

        forecaster = ForecasterRecursive(
            estimator=RandomForestRegressor(n_estimators=2, random_state=42),
            lags=2,
            weight_func=weight_func,
        )

        # Pickle and unpickle
        pickled = pickle.dumps(forecaster)
        unpickled = pickle.loads(pickled)

        # Verify weight_func is still there and functional
        assert unpickled.weight_func is not None
        result = unpickled.weight_func(pd.Index([0, 1]))
        np.testing.assert_array_equal(result, np.array([1.0, 0.9]))

    def test_forecaster_with_weight_function_joblib(self):
        """Test that ForecasterRecursive with WeightFunction can be saved/loaded with joblib."""
        weights = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=range(5))
        weight_func = WeightFunction(weights)

        forecaster = ForecasterRecursive(
            estimator=RandomForestRegressor(n_estimators=2, random_state=42),
            lags=2,
            weight_func=weight_func,
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
            temp_file = f.name

        try:
            # Save and load with joblib
            dump(forecaster, temp_file)
            loaded = load(temp_file)

            # Verify weight_func is still there and functional
            assert loaded.weight_func is not None
            result = loaded.weight_func(pd.Index([0, 1]))
            np.testing.assert_array_equal(result, np.array([1.0, 0.9]))
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_trained_forecaster_with_weight_function_persistence(self):
        """Test that a trained ForecasterRecursive with WeightFunction can be persisted."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        y = pd.Series(np.random.randn(100).cumsum(), index=dates, name="target")

        # Create weights (simulating missing data weights)
        weights = pd.Series(np.ones(100), index=dates)
        weights.iloc[20:30] = 0.5  # Lower weights for some period

        weight_func = WeightFunction(weights)

        forecaster = ForecasterRecursive(
            estimator=RandomForestRegressor(n_estimators=2, random_state=42),
            lags=3,
            weight_func=weight_func,
        )

        # Train the forecaster
        forecaster.fit(y)

        # Make a prediction before saving
        pred_before = forecaster.predict(steps=5)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
            temp_file = f.name

        try:
            # Save the trained forecaster
            dump(forecaster, temp_file)

            # Load the trained forecaster
            loaded = load(temp_file)

            # Make a prediction with loaded forecaster
            pred_after = loaded.predict(steps=5)

            # Predictions should be identical
            pd.testing.assert_series_equal(pred_before, pred_after)

            # Verify weight_func is still functional
            assert loaded.weight_func is not None
            test_weights = loaded.weight_func(dates[:5])
            np.testing.assert_array_equal(test_weights, weights.iloc[:5].values)

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
