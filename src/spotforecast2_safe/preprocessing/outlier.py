# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Optional, Dict

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


def mark_outliers(
    data: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 1234,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Marks outliers as NaN in the dataset using Isolation Forest.

    Args:
        data (pd.DataFrame):
            The input dataset.
        contamination (float):
            The (estimated) proportion of outliers in the dataset.
        random_state (int):
            Random seed for reproducibility. Default is 1234.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the modified dataset with outliers marked as NaN and the outlier labels.

    Examples:
        ```{python}
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        from spotforecast2_safe.preprocessing.outlier import mark_outliers
        path_demo = get_package_data_home() / "demo02.csv"
        data = fetch_data(filename=path_demo)
        print(data.head())
        cleaned_data, outlier_labels = mark_outliers(data, contamination=0.1, random_state=42, verbose=True)
        print(cleaned_data.head())
        print(outlier_labels[:10])
        ```
    """
    for col in data.columns:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        # Fit and predict (-1 for outliers, 1 for inliers)
        outliers = iso.fit_predict(data[col].values.reshape(-1, 1))

        # Mark outliers as NaN
        data.loc[outliers == -1, col] = np.nan

        pct_outliers = (outliers == -1).mean() * 100
        if verbose:
            print(
                f"Column '{col}': Marked {pct_outliers:.4f}% of data points as outliers."
            )
    return data, outliers


def manual_outlier_removal(
    data: pd.DataFrame,
    column: str,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Manual outlier removal function.
    Args:
        data (pd.DataFrame):
            The input dataset.
        column (str):
            The column name in which to perform manual outlier removal.
        lower_threshold (float | None):
            The lower threshold below which values are considered outliers.
            If None, no lower threshold is applied.
        upper_threshold (float | None):
            The upper threshold above which values are considered outliers.
            If None, no upper threshold is applied.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, int]: A tuple containing the modified dataset with outliers marked as NaN and the number of outliers marked.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2_safe.preprocessing.outlier import manual_outlier_removal
        >>> data = fetch_data()
        >>> data, n_manual_outliers = manual_outlier_removal(
        ...     data,
        ...     column='ABC',
        ...     lower_threshold=50,
        ...     upper_threshold=700,
        ...     verbose=True
    """
    if lower_threshold is None and upper_threshold is None:
        if verbose:
            print(f"No thresholds provided for {column}; no outliers marked.")
        return data, 0

    if lower_threshold is not None and upper_threshold is not None:
        mask = (data[column] > upper_threshold) | (data[column] < lower_threshold)
    elif lower_threshold is not None:
        mask = data[column] < lower_threshold
    else:
        mask = data[column] > upper_threshold

    n_manual_outliers = mask.sum()

    data.loc[mask, column] = np.nan

    if verbose:
        if lower_threshold is not None and upper_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} or < {lower_threshold} as outliers in {column}."
            )
        elif lower_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values < {lower_threshold} as outliers in {column}."
            )
        else:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} as outliers in {column}."
            )
    return data, n_manual_outliers


def get_outliers(
    data: pd.DataFrame,
    data_original: Optional[pd.DataFrame] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
) -> Dict[str, pd.Series]:
    """Detect outliers in each column using Isolation Forest.

    This function uses scikit-learn's IsolationForest algorithm to detect outliers
    in each column of the input DataFrame. The original data (before any NaN values
    were introduced) can be provided to identify which values were marked as NaN due
    to outlier detection.

    Args:
        data: The input DataFrame to check for outliers.
        data_original: Optional original DataFrame before outlier marking. If provided,
            helps identify which values became NaN due to outlier detection.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.

    Returns:
        A dictionary mapping column names to Series of outlier values.
        For columns without outliers, an empty Series is returned.

    Raises:
        ValueError: If data is empty or contains no columns.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.preprocessing.outlier import get_outliers
        >>>
        >>> # Create sample data with outliers
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({
        ...     'A': np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
        ...     'B': np.concatenate([np.random.normal(5, 2, 100), [100, 110, 120]])
        ... })
        >>> data_original = data.copy()
        >>>
        >>> # Detect outliers
        >>> outliers = get_outliers(data_original, contamination=0.03)
        >>> for col, outlier_vals in outliers.items():
        ...     print(f"{col}: {len(outlier_vals)} outliers detected")
    """
    if data.empty:
        raise ValueError("Input data is empty")
    if len(data.columns) == 0:
        raise ValueError("Input data contains no columns")

    outliers_dict = {}

    for col in data.columns:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = iso.fit_predict(data[col].values.reshape(-1, 1))

        # Get outlier values
        if data_original is not None:
            # Use original data to identify outlier values
            outlier_mask = predictions == -1
            outliers_dict[col] = data_original.loc[outlier_mask, col]
        else:
            # Use current data
            outlier_mask = predictions == -1
            outliers_dict[col] = data.loc[outlier_mask, col]

    return outliers_dict
