# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from spotforecast2_safe.preprocessing import RollingFeatures


def test_rolling_features_transform_2d():
    """
    Test RollingFeatures.transform with 2D input (vectorized bootstrap case).
    """
    rf = RollingFeatures(stats=["mean", "std"], window_sizes=[3, 5])
    # Create 2D array (window_size, n_boot)
    X = np.random.randn(10, 3)

    # Expected output shape: (n_boot, n_features)
    # n_features = 2 stats * 2 window_sizes = 4
    features = rf.transform(X)

    assert features.shape == (3, 4)
    # Check if first column is mean of last 3 rows for each boot
    for i in range(3):
        np.testing.assert_almost_equal(features[i, 0], np.mean(X[-3:, i]))
        np.testing.assert_almost_equal(features[i, 1], np.std(X[-3:, i], ddof=1))
        np.testing.assert_almost_equal(features[i, 2], np.mean(X[-5:, i]))
        np.testing.assert_almost_equal(features[i, 3], np.std(X[-5:, i], ddof=1))


def test_rolling_features_transform_1d():
    """
    Test RollingFeatures.transform with 1D input (last window case).
    """
    rf = RollingFeatures(stats="mean", window_sizes=3)
    X = np.arange(10).astype(float)

    features = rf.transform(X)

    # transform() on 1D input returns (n_features,) - the stats of the last window
    assert features.shape == (1,)
    np.testing.assert_almost_equal(features[0], (7.0 + 8.0 + 9.0) / 3.0)


def test_rolling_features_new_stats():
    """
    Test RollingFeatures with ratio_min_max, coef_variation, and ewm.
    """
    rf = RollingFeatures(
        stats=["ratio_min_max", "coef_variation", "ewm"], window_sizes=3
    )
    X = np.array([10.0, 20.0, 30.0, 40.0])

    features = rf.transform(X)
    # 1D input -> (3,)
    assert features.shape == (3,)

    # Window=3 -> [20, 30, 40]
    # ratio_min_max: 20/40 = 0.5
    # coef_variation: std([20, 30, 40]) / mean([20, 30, 40])
    window = X[-3:]
    expected_ratio = np.min(window) / np.max(window)
    expected_cv = np.std(window, ddof=1) / np.mean(window)

    np.testing.assert_almost_equal(features[0], expected_ratio)
    np.testing.assert_almost_equal(features[1], expected_cv)
    # ewm is more complex, but let's check it's not NaN
    assert not np.isnan(features[2])
