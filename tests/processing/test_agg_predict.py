import pandas as pd
import pytest

from spotforecast2_safe.processing.agg_predict import agg_predict


class TestAggPredict:
    def test_agg_predict_default_weights(self):
        """Test aggregation with default weights."""
        # Create a dataframe with generic columns
        columns = [f"col_{i}" for i in range(5)]

        # Assign value 10.0 to all
        data = {col: [10.0] for col in columns}
        df = pd.DataFrame(data, index=[0])

        # Calculation:
        # Sum of all columns: 10.0 * 5 = 50.0

        result = agg_predict(df)
        assert result.iloc[0] == 50.0

    def test_agg_predict_custom_weights(self):
        """Test aggregation with custom weights."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        weights = {"A": 0.5, "B": 2.0}

        # Expected:
        # 0: 1*0.5 + 3*2.0 = 0.5 + 6.0 = 6.5
        # 1: 2*0.5 + 4*2.0 = 1.0 + 8.0 = 9.0

        result = agg_predict(df, weights=weights)
        assert result.iloc[0] == 6.5
        assert result.iloc[1] == 9.0

    def test_agg_predict_missing_columns(self):
        """Test that ValueError is raised if columns are missing."""
        df = pd.DataFrame({"A": [1, 2]})
        weights = {"A": 1.0, "Missing": 1.0}

        with pytest.raises(
            ValueError, match="Missing columns in predictions dataframe"
        ):
            agg_predict(df, weights=weights)

    def test_agg_predict_extra_columns(self):
        """Test that extra columns in dataframe are ignored."""
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        weights = {"A": 1.0, "B": 1.0}

        result = agg_predict(df, weights=weights)
        assert result.iloc[0] == 3.0
