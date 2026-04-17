# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
QuantileBinner class for binning data into quantile-based bins.

This module contains the QuantileBinner class which bins data into quantile-based bins
using numpy.percentile with optimized performance using numpy.searchsorted.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from spotforecast2_safe.exceptions import IgnoredArgumentWarning


class QuantileBinner(BaseEstimator, TransformerMixin):
    """
    Bin data into quantile-based bins using numpy.percentile.

    This class is similar to sklearn's KBinsDiscretizer but optimized for
    performance using numpy.searchsorted for fast bin assignment. Bin intervals
    are defined following the convention: bins[i-1] <= x < bins[i]. Values
    outside the range are clipped to the first or last bin.

    Args:
        n_bins: The number of quantile-based bins to create. Must be >= 2.
        method: The method used to compute quantiles, passed to numpy.percentile.
            Default is 'linear'. Valid values: "inverse_cdf",
            "averaged_inverse_cdf", "closest_observation",
            "interpolated_inverse_cdf", "hazen", "weibull", "linear",
            "median_unbiased", "normal_unbiased".
        subsample: Maximum number of samples for computing quantiles. If dataset
            has more samples, a random subset is used. Default 200000.
        dtype: Data type for bin indices. Default is numpy.float64.
        random_state: Random seed for subset generation. Default 789654.

    Attributes:
        n_bins (int): Number of bins to create.
        method (str): Quantile computation method.
        subsample (int): Maximum samples for quantile computation.
        dtype (type): Data type for bin indices.
        random_state (int): Random seed.
        n_bins_ (int): Actual number of bins after fitting (may differ from n_bins
            if duplicate edges are found).
        bin_edges_ (np.ndarray): Edges of the bins learned during fitting.
        internal_edges_ (np.ndarray): Internal edges for optimized bin assignment.
        intervals_ (dict): Mapping from bin index to (lower, upper) interval bounds.

    Examples:
        >>> import numpy as np
        >>> from spotforecast2_safe.preprocessing import QuantileBinner
        >>>
        >>> # Basic usage: create 3 quantile bins
        >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> binner = QuantileBinner(n_bins=3)
        >>> _ = binner.fit(X)
        >>> result = binner.transform(np.array([1.5, 5.5, 9.5]))
        >>> print(result)
        [0. 1. 2.]
        >>>
        >>> # Check bin intervals
        >>> print(binner.n_bins_)
        3
        >>> assert len(binner.intervals_) == 3
        >>>
        >>> # Use fit_transform for one-step operation
        >>> X2 = np.array([10, 20, 30, 40, 50])
        >>> binner2 = QuantileBinner(n_bins=2)
        >>> bins = binner2.fit_transform(X2)
        >>> print(bins)
        [0. 0. 1. 1. 1.]
    """

    def __init__(
        self,
        n_bins: int,
        method: str = "linear",
        subsample: int = 200000,
        dtype: type = np.float64,
        random_state: int = 789654,
    ) -> None:

        self._validate_params(n_bins, method, subsample, dtype, random_state)

        self.n_bins = n_bins
        self.method = method
        self.subsample = subsample
        self.dtype = dtype
        self.random_state = random_state
        self.n_bins_ = None
        self.bin_edges_ = None
        self.internal_edges_ = None
        self.intervals_ = None

    def _validate_params(
        self, n_bins: int, method: str, subsample: int, dtype: type, random_state: int
    ):
        """
        Validate parameters passed to the class initializer.

        Args:
            n_bins: Number of quantile-based bins. Must be int >= 2.
            method: Quantile computation method for numpy.percentile.
            subsample: Number of samples for computing quantiles. Must be int >= 1.
            dtype: Data type for bin indices. Must be a valid numpy dtype.
            random_state: Random seed for subset generation. Must be int >= 0.

        Raises:
            ValueError: If n_bins < 2, method is invalid, subsample < 1,
                random_state < 0, or dtype is not a valid type.

        Examples:
            >>> import numpy as np
            >>> from spotforecast2_safe.preprocessing import QuantileBinner
            >>>
            >>> # Valid parameters work fine
            >>> binner = QuantileBinner(n_bins=5, method='linear')
            >>> assert binner.n_bins == 5
            >>>
            >>> # Invalid n_bins raises ValueError
            >>> try:
            ...     binner = QuantileBinner(n_bins=1)
            ... except ValueError as e:
            ...     assert 'greater than 1' in str(e)
            >>>
            >>> # Invalid method raises ValueError
            >>> try:
            ...     binner = QuantileBinner(n_bins=3, method='invalid')
            ... except ValueError as e:
            ...     assert 'must be one of' in str(e)
        """

        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError(f"`n_bins` must be an int greater than 1. Got {n_bins}.")

        valid_methods = [
            "inverse_cdf",
            "averaged_inverse_cdf",
            "closest_observation",
            "interpolated_inverse_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
        ]
        if method not in valid_methods:
            raise ValueError(f"`method` must be one of {valid_methods}. Got {method}.")
        if not isinstance(subsample, int) or subsample < 1:
            raise ValueError(
                f"`subsample` must be an integer greater than or equal to 1. "
                f"Got {subsample}."
            )
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError(
                f"`random_state` must be an integer greater than or equal to 0. "
                f"Got {random_state}."
            )
        if not isinstance(dtype, type):
            raise ValueError(f"`dtype` must be a valid numpy dtype. Got {dtype}.")

    def fit(self, X: np.ndarray, y: object = None) -> object:
        """
        Learn bin edges based on quantiles from training data.

        Computes quantile-based bin edges using numpy.percentile. If the dataset
        contains more samples than `subsample`, a random subset is used. Duplicate
        edges (which can occur with repeated values) are removed automatically.

        Args:
            X: Training data (1D numpy array) for computing quantiles.
            y: Ignored.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data X is empty.

        Examples:
            >>> import numpy as np
            >>> from spotforecast2_safe.preprocessing import QuantileBinner
            >>>
            >>> # Fit with basic data
            >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> binner = QuantileBinner(n_bins=3)
            >>> _ = binner.fit(X)
            >>> print(binner.n_bins_)
            3
            >>> print(len(binner.bin_edges_))
            4
            >>>
            >>> # Fit with repeated values (may reduce number of bins)
            >>> X_repeated = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
            >>> binner2 = QuantileBinner(n_bins=5)
            >>> _ = binner2.fit(X_repeated)
            >>> # n_bins_ may be less than 5 due to duplicates
            >>> assert binner2.n_bins_ <= 5
        """
        # Note: Original implementation expects X, but sklearn TransformerMixin passes y=None.
        # Adjusted signature to (self, X: np.ndarray, y: object = None)

        if X.size == 0:
            raise ValueError("Input data `X` cannot be empty.")
        if len(X) > self.subsample:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.integers(0, len(X), self.subsample)]

        bin_edges = np.percentile(
            a=X, q=np.linspace(0, 100, self.n_bins + 1), method=self.method
        )

        # Remove duplicate edges (can happen when data has many repeated values)
        # to ensure bins are always numbered 0 to n_bins_-1
        self.bin_edges_ = np.unique(bin_edges)

        # Ensure at least 1 bin when all values are identical
        if len(self.bin_edges_) == 1:
            # Create artificial edges around the single value
            self.bin_edges_ = np.array([self.bin_edges_.item(), self.bin_edges_.item()])

        self.n_bins_ = len(self.bin_edges_) - 1

        if self.n_bins_ != self.n_bins:
            warnings.warn(
                f"The number of bins has been reduced from {self.n_bins} to "
                f"{self.n_bins_} due to duplicated edges caused by repeated predicted "
                f"values.",
                IgnoredArgumentWarning,
            )

        # Internal edges for optimized transform with searchsorted
        self.internal_edges_ = self.bin_edges_[1:-1]
        self.intervals_ = {
            int(i): (float(self.bin_edges_[i]), float(self.bin_edges_[i + 1]))
            for i in range(self.n_bins_)
        }

        return self

    def transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Assign new data to learned bins.

        Uses numpy.searchsorted for efficient bin assignment. Values are assigned
        to bins following the convention: bins[i-1] <= x < bins[i]. Values outside
        the fitted range are clipped to the first or last bin.

        Args:
            X: Data to assign to bins (1D numpy array).
            y: Ignored.

        Returns:
            Bin indices as numpy array with dtype specified in __init__.

        Raises:
            NotFittedError: If fit() has not been called yet.

        Examples:
            >>> import numpy as np
            >>> from spotforecast2_safe.preprocessing import QuantileBinner
            >>>
            >>> # Fit and transform
            >>> X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> binner = QuantileBinner(n_bins=3)
            >>> _ = binner.fit(X_train)
            >>>
            >>> X_test = np.array([1.5, 5.5, 9.5])
            >>> result = binner.transform(X_test)
            >>> print(result)
            [0. 1. 2.]
            >>>
            >>> # Values outside range are clipped
            >>> X_extreme = np.array([0, 100])
            >>> result_extreme = binner.transform(X_extreme)
            >>> print(result_extreme)  # Both clipped to valid bin indices
            [0. 2.]
        """

        if self.bin_edges_ is None:
            raise NotFittedError(
                "The model has not been fitted yet. Call 'fit' with training data first."
            )

        bin_indices = np.searchsorted(self.internal_edges_, X, side="right").astype(
            self.dtype
        )

        return bin_indices

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # fit_transform is usually provided by TransformerMixin but we can implement it
        # or rely on inheritance. The original implementation had it explicitly.

        self.fit(X, y)
        return self.transform(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters of the quantile binner.

        Returns:
            Dictionary containing n_bins, method, subsample, dtype, and
            random_state parameters.

        Examples:
            >>> import numpy as np
            >>> from spotforecast2_safe.preprocessing import QuantileBinner
            >>>
            >>> binner = QuantileBinner(n_bins=5, method='median_unbiased', subsample=1000)
            >>> params = binner.get_params()
            >>> print(params['n_bins'])
            5
            >>> print(params['method'])
            median_unbiased
            >>> print(params['subsample'])
            1000
        """

        return {
            "n_bins": self.n_bins,
            "method": self.method,
            "subsample": self.subsample,
            "dtype": self.dtype,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "QuantileBinner":
        """
        Set parameters of the QuantileBinner.

        Args:
            **params: Parameter names and values to set as keyword arguments.

        Returns:
            self: Returns the updated QuantileBinner instance.

        Examples:
            >>> import numpy as np
            >>> from spotforecast2_safe.preprocessing import QuantileBinner
            >>>
            >>> binner = QuantileBinner(n_bins=3)
            >>> print(binner.n_bins)
            3
            >>> binner.set_params(n_bins=5, method='weibull')
            >>> print(binner.n_bins)
            5
            >>> print(binner.method)
            weibull
        """

        for param, value in params.items():
            setattr(self, param, value)
        return self
