# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Base class for time series cross-validation splitting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from spotforecast2_safe.exceptions import IgnoredArgumentWarning


class BaseFold:
    """
    Base class for all Fold classes in spotforecast. All fold classes should specify
    all the parameters that can be set at the class level in their ``__init__``.

    Args:
        steps (int, optional): Number of observations used to be predicted in each fold.
            This is also commonly referred to as the forecast horizon or test size.
            Defaults to None.
        initial_train_size (int | str | pd.Timestamp, optional): Number of observations
            used for initial training.

            - If an integer, the number of observations used for initial training.
            - If a date string or pandas Timestamp, it is the last date included in
              the initial training set.
            Defaults to None.
        fold_stride (int, optional): Number of observations that the start of the test
            set advances between consecutive folds.

            - If `None`, it defaults to the same value as `steps`, meaning that folds
              are placed back-to-back without overlap.
            - If `fold_stride < steps`, test sets overlap and multiple forecasts will
              be generated for the same observations.
            - If `fold_stride > steps`, gaps are left between consecutive test sets.
            Defaults to None.
        window_size (int, optional): Number of observations needed to generate the
            autoregressive predictors. Defaults to None.
        differentiation (int, optional): Number of observations to use for differentiation.
            This is used to extend the `last_window` as many observations as the
            differentiation order. Defaults to None.
        refit (bool | int, optional): Whether to refit the forecaster in each fold.

            - If `True`, the forecaster is refitted in each fold.
            - If `False`, the forecaster is trained only in the first fold.
            - If an integer, the forecaster is trained in the first fold and then refitted
              every `refit` folds.
            Defaults to False.
        fixed_train_size (bool, optional): Whether the training size is fixed or increases
            in each fold. Defaults to True.
        gap (int, optional): Number of observations between the end of the training set
            and the start of the test set. Defaults to 0.
        skip_folds (int | list, optional): Number of folds to skip.

            - If an integer, every 'skip_folds'-th is returned.
            - If a list, the indexes of the folds to skip.

            For example, if `skip_folds=3` and there are 10 folds, the returned folds are
            0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
            8, and 9. Defaults to None.
        allow_incomplete_fold (bool, optional): Whether to allow the last fold to include
            fewer observations than `steps`. If `False`, the last fold is excluded if it
            is incomplete. Defaults to True.
        return_all_indexes (bool, optional): Whether to return all indexes or only the
            start and end indexes of each fold. Defaults to False.
        verbose (bool, optional): Whether to print information about generated folds.
            Defaults to True.

    Attributes:
        initial_train_size (int): Number of observations used for initial training.
        window_size (int): Number of observations needed to generate the
            autoregressive predictors.
        differentiation (int): Number of observations to use for differentiation.
            This is used to extend the `last_window` as many observations as the
            differentiation order.
        return_all_indexes (bool): Whether to return all indexes or only the start
            and end indexes of each fold.
        verbose (bool): Whether to print information about generated folds.

    Methods:
        _validate_params: Validates the input parameters to ensure correctness.
        _extract_index: Extracts and returns the index from the input data X.
        set_params: Set the parameters of the Fold object after validating them.
    """

    def __init__(
        self,
        steps: int | None = None,
        initial_train_size: int | str | pd.Timestamp | None = None,
        fold_stride: int | None = None,
        window_size: int | None = None,
        differentiation: int | None = None,
        refit: bool | int = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: int | list[int] | None = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True,
    ) -> None:

        self._validate_params(
            cv_name=type(self).__name__,
            steps=steps,
            initial_train_size=initial_train_size,
            fold_stride=fold_stride,
            window_size=window_size,
            differentiation=differentiation,
            refit=refit,
            fixed_train_size=fixed_train_size,
            gap=gap,
            skip_folds=skip_folds,
            allow_incomplete_fold=allow_incomplete_fold,
            return_all_indexes=return_all_indexes,
            verbose=verbose,
        )

        self.initial_train_size = initial_train_size
        self.window_size = window_size
        self.differentiation = differentiation
        self.return_all_indexes = return_all_indexes
        self.verbose = verbose

    def _validate_params(
        self,
        cv_name: str,
        steps: int | None = None,
        initial_train_size: int | str | pd.Timestamp | None = None,
        fold_stride: int | None = None,
        window_size: int | None = None,
        differentiation: int | None = None,
        refit: bool | int = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: int | list[int] | None = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Validate all input parameters to ensure correctness.

        Args:
            cv_name (str):
            steps (int | None):
            initial_train_size (int | str | pd.Timestamp | None):
            fold_stride (int | None):
            window_size (int | None):
            differentiation (int | None):
            refit (bool | int):
            fixed_train_size (bool):
            gap (int):
            skip_folds (int | list[int] | None):
            allow_incomplete_fold (bool):
            return_all_indexes (bool):
            verbose (bool):
            **kwargs:

        Returns:
            None

        Raises:
            ValueError: If any of the input parameters are invalid.

        Examples:
            >>> from spotforecast2_safe.model_selection import TimeSeriesFold
            >>> cv = TimeSeriesFold(steps=1)
            >>> cv._validate_params(
            ...     cv_name="TimeSeriesFold",
            ...     steps=1,
            ...     initial_train_size=1,
            ...     fold_stride=1,
            ...     window_size=1,
            ...     differentiation=1,
            ...     refit=False,
            ...     fixed_train_size=True,
            ...     gap=0,
            ...     skip_folds=None,
            ...     allow_incomplete_fold=True,
            ...     return_all_indexes=False,
            ...     verbose=True,
            ... )
        """

        if cv_name == "TimeSeriesFold":
            if not isinstance(steps, (int, np.integer)) or steps < 1:
                raise ValueError(
                    f"`steps` must be an integer greater than 0. Got {steps}."
                )
            if not isinstance(
                initial_train_size, (int, np.integer, str, pd.Timestamp, type(None))
            ):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0, a date "
                    f"string, a pandas Timestamp, or None. Got {initial_train_size}."
                )
            if (
                isinstance(initial_train_size, (int, np.integer))
                and initial_train_size < 1
            ):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0, "
                    f"a date string, a pandas Timestamp, or None. Got {initial_train_size}."
                )
            if fold_stride is not None:
                if not isinstance(fold_stride, (int, np.integer)) or fold_stride < 1:
                    raise ValueError(
                        f"`fold_stride` must be an integer greater than 0. Got {fold_stride}."
                    )
            if not isinstance(refit, (bool, int, np.integer)):
                raise TypeError(
                    f"`refit` must be a boolean or an integer equal or greater than 0. "
                    f"Got {refit}."
                )
            if (
                isinstance(refit, (int, np.integer))
                and not isinstance(refit, bool)
                and refit < 0
            ):
                raise TypeError(
                    f"`refit` must be a boolean or an integer equal or greater than 0. "
                    f"Got {refit}."
                )
            if not isinstance(fixed_train_size, bool):
                raise TypeError(
                    f"`fixed_train_size` must be a boolean: `True`, `False`. "
                    f"Got {fixed_train_size}."
                )
            if not isinstance(gap, (int, np.integer)) or gap < 0:
                raise ValueError(
                    f"`gap` must be an integer greater than or equal to 0. Got {gap}."
                )
            if skip_folds is not None:
                if not isinstance(skip_folds, (int, np.integer, list, type(None))):
                    raise TypeError(
                        f"`skip_folds` must be an integer greater than 0, a list of "
                        f"integers or `None`. Got {skip_folds}."
                    )
                if isinstance(skip_folds, (int, np.integer)) and skip_folds < 1:
                    raise ValueError(
                        f"`skip_folds` must be an integer greater than 0, a list of "
                        f"integers or `None`. Got {skip_folds}."
                    )
                if isinstance(skip_folds, list) and any([x < 1 for x in skip_folds]):
                    raise ValueError(
                        f"`skip_folds` list must contain integers greater than or "
                        f"equal to 1. The first fold is always needed to train the "
                        f"forecaster. Got {skip_folds}."
                    )
            if not isinstance(allow_incomplete_fold, bool):
                raise TypeError(
                    f"`allow_incomplete_fold` must be a boolean: `True`, `False`. "
                    f"Got {allow_incomplete_fold}."
                )

        if cv_name == "OneStepAheadFold":
            if not isinstance(initial_train_size, (int, np.integer, str, pd.Timestamp)):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0, a date "
                    f"string, or a pandas Timestamp. Got {initial_train_size}."
                )
            if (
                isinstance(initial_train_size, (int, np.integer))
                and initial_train_size < 1
            ):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0, "
                    f"a date string, or a pandas Timestamp. Got {initial_train_size}."
                )

        if (
            not isinstance(window_size, (int, np.integer, pd.DateOffset, type(None)))
            or isinstance(window_size, (int, np.integer))
            and window_size < 1
        ):
            raise ValueError(
                f"`window_size` must be an integer greater than 0. Got {window_size}."
            )

        if differentiation is not None:
            if (
                not isinstance(differentiation, (int, np.integer))
                or differentiation < 0
            ):
                raise ValueError(
                    f"`differentiation` must be None or an integer greater than or "
                    f"equal to 0. Got {differentiation}."
                )

        if not isinstance(return_all_indexes, bool):
            raise TypeError(
                f"`return_all_indexes` must be a boolean: `True`, `False`. "
                f"Got {return_all_indexes}."
            )

        if not isinstance(verbose, bool):
            raise TypeError(
                f"`verbose` must be a boolean: `True`, `False`. " f"Got {verbose}."
            )

    def _extract_index(
        self,
        X: pd.Series | pd.DataFrame | pd.Index | dict[str, pd.Series | pd.DataFrame],
    ) -> pd.Index:
        """
        Extracts and returns the index from the input data X.

        Args:
            X (pd.Series | pd.DataFrame | pd.Index | dict): Time series data or
                index to split.

        Returns:
            pd.Index: Index extracted from the input data.

        Raises:
            TypeError: If the input data is not of the expected type or if the series in
                a dictionary do not have valid indexes.
            ValueError: If the series in a dictionary have different frequencies or if any
                series have no frequency.

        Examples:
            >>> from spotforecast2_safe.model_selection import TimeSeriesFold
            >>> cv = TimeSeriesFold(steps=1)
            >>> idx = cv._extract_index(pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3)))
            >>> print(idx)
            DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]', freq='D')
        """

        if isinstance(X, (pd.Series, pd.DataFrame)):
            idx = X.index
        elif isinstance(X, dict):
            indexes_freq = set()
            not_valid_index = []
            min_index = []
            max_index = []
            for k, v in X.items():
                if v is None:
                    continue

                idx = v.index
                if isinstance(idx, pd.DatetimeIndex):
                    indexes_freq.add(idx.freq)
                elif isinstance(idx, pd.RangeIndex):
                    indexes_freq.add(idx.step)
                else:
                    not_valid_index.append(k)

                min_index.append(idx[0])
                max_index.append(idx[-1])

            if not_valid_index:
                raise TypeError(
                    f"If `X` is a dictionary, all series must have a Pandas "
                    f"RangeIndex or DatetimeIndex with the same step/frequency. "
                    f"Review series: {not_valid_index}"
                )

            if None in indexes_freq:
                raise ValueError(
                    "If `X` is a dictionary, all series must have a Pandas "
                    "RangeIndex or DatetimeIndex with the same step/frequency. "
                    "Found series with no frequency or step."
                )
            if not len(indexes_freq) == 1:
                raise ValueError(
                    f"If `X` is a dictionary, all series must have a Pandas "
                    f"RangeIndex or DatetimeIndex with the same step/frequency. "
                    f"Found frequencies: {sorted(indexes_freq)}"
                )

            if isinstance(idx, pd.DatetimeIndex):
                idx = pd.date_range(
                    start=min(min_index), end=max(max_index), freq=indexes_freq.pop()
                )
            else:
                idx = pd.RangeIndex(
                    start=min(min_index),
                    stop=max(max_index) + 1,
                    step=indexes_freq.pop(),
                )
        else:
            idx = X

        return idx

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the Fold object. Before overwriting the current
        parameters, the input parameters are validated to ensure correctness.

        Args:
            params (dict): Dictionary with the parameters to set.

        Examples:
            >>> from spotforecast2_safe.model_selection import TimeSeriesFold
            >>> cv = TimeSeriesFold(steps=1)
            >>> cv.set_params({
            ...     "steps": 2,
            ...     "initial_train_size": 10,
            ...     "fold_stride": 2,
            ...     "window_size": 5,
            ...     "differentiation": 1,
            ...     "refit": True,
            ...     "fixed_train_size": False,
            ...     "gap": 1,
            ...     "skip_folds": 2,
            ...     "allow_incomplete_fold": False,
            ...     "return_all_indexes": True,
            ...     "verbose": False,
            ... })
        """

        if not isinstance(params, dict):
            raise TypeError(f"`params` must be a dictionary. Got {type(params)}.")

        current_params = dict(vars(self))
        unknown_params = set(params.keys()) - set(current_params.keys())
        if unknown_params:
            warnings.warn(
                f"Unknown parameters: {unknown_params}. They have been ignored.",
                IgnoredArgumentWarning,
            )

        filtered_params = {k: v for k, v in params.items() if k in current_params}
        updated_params = {
            "cv_name": type(self).__name__,
            **current_params,
            **filtered_params,
        }

        self._validate_params(**updated_params)
        for key, value in updated_params.items():
            setattr(self, key, value)
