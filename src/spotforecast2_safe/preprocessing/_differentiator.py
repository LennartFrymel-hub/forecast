# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import _check_X_numpy_ndarray_1d


class TimeSeriesDifferentiator(BaseEstimator, TransformerMixin):
    """
    Transforms a time series into a differenced time series.

    Args:
        order (int, optional): Order of differentiation. Defaults to 1.
        window_size (int, optional): The window size used by the forecaster. Defaults to None.
        initial_values (list, numpy ndarray, optional): Values to be used for the inverse transformation (reverting differentiation).
            If None, the first `order` values of the training data `X` are stored during `fit`.

    Attributes:
        initial_values_ (list): Values stored for inverse transformation.
        last_values_ (list): Last values of the differenced time series.
        pre_train_values_ (list): First training values for inverse transformation of training data.
    """

    def __init__(
        self,
        order: int = 1,
        window_size: int | None = None,
        initial_values: list | np.ndarray | None = None,
    ):
        self.order = order
        self.window_size = window_size
        self.initial_values = initial_values

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit(self, X: np.ndarray, y: object = None) -> object:
        """
        Store initial values if not provided.
        """
        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")

        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise TypeError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {type(self.window_size)}."
                )
            if self.window_size < 1:
                raise ValueError(
                    f"Parameter `window_size` must be an integer greater than 0. "
                    f"Found {self.window_size}."
                )

        self.initial_values_ = []
        self.pre_train_values_ = []
        self.last_values_ = []

        if self.initial_values is None:
            if len(X) < self.order:
                raise ValueError(
                    f"The time series must have at least {self.order} values "
                    f"to compute the differentiation of order {self.order}."
                )

            # Logic similar to Skforecast:
            # We iterate to capture initial values, pre_train_values, and last_values
            # Skforecast does this iteratively. The original spotforecast implementation
            # was simplified. Integrating window_size support requires the iterative approach
            # or careful indexing.

            # Re-implementing skforecast logic for robustness with window_size

            current_X = X.copy()
            for i in range(self.order):
                self.initial_values_.append(current_X[0])
                if self.window_size is not None:
                    # Skforecast logic: self.pre_train_values.append(X[self.window_size - self.order])
                    # But wait, skforecast loop updates X_diff. A recursive implementation.

                    # If we follow skforecast exactly:
                    # if i == 0: X_diff = diff(X); initial = X[0]; pre = X[ws-order]; last = X[-1]
                    # else: X_diff = diff(previous_diff); initial = prev_diff[0]; pre = prev_diff[ws-order]; last = prev_diff[-1]

                    # Current X is the series being differentiated in this step
                    if len(current_X) > (self.window_size - self.order):
                        self.pre_train_values_.append(
                            current_X[self.window_size - self.order]
                        )
                    else:
                        # Fallback if X is smaller than window_size (shouldn't happen during training if window_size is correct)
                        self.pre_train_values_.append(np.nan)

                self.last_values_.append(current_X[-1])
                current_X = np.diff(current_X, n=1)

        else:
            if len(self.initial_values) != self.order:
                raise ValueError(
                    f"The length of `initial_values` must be equal to the order "
                    f"of differentiation ({self.order})."
                )
            self.initial_values_ = list(self.initial_values)
            # If initial_values provided, we can still compute last_values_ from X if we assume X is the training data
            # But usually initial_values are provided when loading/restoring.
            # For now, let's keep the user provided initial values.
            # But we still need last_values_ for next window.
            # Assuming X is the training data passed to fit()

            self.last_values_ = list(X[-self.order :])

        return self

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Fit and transform.
        """
        return self.fit(X).transform(X)

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Compute the differences.
        """
        if not hasattr(self, "initial_values_") and self.initial_values is not None:
            self.fit(X)
        elif not hasattr(self, "initial_values_"):
            check_is_fitted(self, ["initial_values_"])

        X_diff = np.diff(X, n=self.order)
        # Pad with NaNs to keep same length
        X_diff = np.concatenate([np.full(self.order, np.nan), X_diff])

        # Update last values seen (for next window inverse)
        self.last_values_ = X[-self.order :]

        return X_diff

    def inverse_transform_next_window(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform for the next window of predictions.
        """
        check_is_fitted(self, ["initial_values_", "last_values_"])

        if self.order == 1:
            result = np.cumsum(X) + self.last_values_[-1]
        else:
            # Recursive or iterative approach for higher orders
            # Simplified: Assuming order 1 is sufficient for now or throwing error
            raise NotImplementedError(
                "inverse_transform_next_window not implemented for order > 1"
            )

        return result

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def inverse_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Revert the differences.
        """
        check_is_fitted(self, ["initial_values_"])

        # X contains the differenced series (with NaNs at the beginning potentially)
        # remove NaNs at the start corresponding to order
        X_clean = X[self.order :]

        if len(X_clean) == 0:
            # Just return initial values if only NaNs were passed
            return np.array(self.initial_values_)

        result = list(self.initial_values_)

        if self.order == 1:
            current_value = result[-1]
            restored = []
            for diff_val in X_clean:
                current_value += diff_val
                restored.append(current_value)
            result.extend(restored)
        else:
            # Recursive reconstruction for higher orders logic check
            # For order > 1, np.diff does repeated diffs.
            # To invert, we need to do repeated cumsum.
            # But we need appropriate initial values for each level of integration.
            # This is a simplified version.

            raise NotImplementedError(
                "Inverse transform for order > 1 is currently not fully implemented in this port."
            )

        return np.array(result)

    @_check_X_numpy_ndarray_1d(ensure_1d=True)
    def inverse_transform_training(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """
        Reverts the differentiation for training data.
        """
        if not hasattr(self, "pre_train_values_") or not self.pre_train_values_:
            raise ValueError(
                "The `window_size` parameter must be set before fitting the "
                "transformer to revert the differentiation of the training "
                "time series."
            )

        # Implementation of inverse_transform_training logic
        # For order 1
        if self.order == 1:
            X_clean = X[self.order :]
            # Reconstruct
            # pre_train_values_ contains the value right BEFORE the training window starts?
            # skforecast: X_undiff = np.insert(X, 0, self.pre_train_values[-1])
            #             X_undiff = np.cumsum(X_undiff)
            #             X_undiff = X_undiff[self.order:]

            X_undiff = np.insert(X_clean, 0, self.pre_train_values_[-1])
            X_undiff = np.cumsum(X_undiff)
            # The skforecast logic seems to insert, cumsum, then slice.
            # If X_clean is the differentiated training data (which generates y_train),
            # we need the value just before it to start cumsum.

            # Simplified for order=1 for now to match safety/robustness needs.
            return X_undiff

        else:
            raise NotImplementedError(
                "inverse_transform_training not implemented for order > 1"
            )
