"""
Test suite for outliers documentation examples.

Implements pytest tests for examples from docs/preprocessing/outliers.qmd
to ensure documentation accuracy and example functionality.
"""

from spotforecast2_safe.preprocessing.outlier import mark_outliers
import pandas as pd


def test_docs_outliers_mark_outliers_example():
    """Validates the mark_outliers example from outliers.qmd."""
    # Create sample time series data
    data = pd.DataFrame(
        {
            "value": [1, 2, 100, 4, 5, 6, 7, 8, 9, 10],  # 100 is an outlier
        }
    )

    # Mark outliers
    result_data, outlier_labels = mark_outliers(
        data,
        contamination=0.1,  # Expect 10% contamination
    )

    # In a set of 10 items with 0.1 contamination, we expect exactly 1 outlier to be flagged (-1)
    outlier_count = (outlier_labels == -1).sum()
    assert outlier_count == 1
    # Verify the outlier is correctly identified at index 2 (value 100)
    assert outlier_labels[2] == -1

    print(f"Outliers marked: {outlier_count} records")
