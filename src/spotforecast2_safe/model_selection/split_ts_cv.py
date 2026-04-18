# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Time series cross-validation splitting.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from spotforecast2_safe.exceptions import IgnoredArgumentWarning
from spotforecast2_safe.forecaster.utils import (
    date_to_index_position,
    get_style_repr_html,
)

from .split_base import BaseFold


class TimeSeriesFold(BaseFold):
    """Class to split time series data into train and test folds.

    When used within a backtesting or hyperparameter search, the arguments
    'initial_train_size', 'window_size' and 'differentiation' are not required
    as they are automatically set by the backtesting or hyperparameter search
    functions.

    Args:
        steps: Number of observations used to be predicted in each fold.
            This is also commonly referred to as the forecast horizon or test size.
        initial_train_size: Number of observations used for initial training.

            - If `None` or 0, the initial forecaster is not trained in the first fold.
            - If an integer, the number of observations used for initial training.
            - If a date string or pandas Timestamp, it is the last date included in
              the initial training set.

            Defaults to None.
        fold_stride: Number of observations that the start of the test set
            advances between consecutive folds.

            - If `None`, it defaults to the same value as `steps`, meaning that folds
              are placed back-to-back without overlap.
            - If `fold_stride < steps`, test sets overlap and multiple forecasts will
              be generated for the same observations.
            - If `fold_stride > steps`, gaps are left between consecutive test sets.

            Defaults to None.
        window_size: Number of observations needed to generate the
            autoregressive predictors. Defaults to None.
        differentiation: Number of observations to use for differentiation.
            This is used to extend the `last_window` as many observations as the
            differentiation order. Defaults to None.
        refit: Whether to refit the forecaster in each fold.

            - If `True`, the forecaster is refitted in each fold.
            - If `False`, the forecaster is trained only in the first fold.
            - If an integer, the forecaster is trained in the first fold and then refitted
              every `refit` folds.

            Defaults to False.
        fixed_train_size: Whether the training size is fixed or increases
            in each fold. Defaults to True.
        gap: Number of observations between the end of the training set
            and the start of the test set. Defaults to 0.
        skip_folds: Number of folds to skip.

            - If an integer, every 'skip_folds'-th is returned.
            - If a list, the indexes of the folds to skip.

            For example, if `skip_folds=3` and there are 10 folds, the returned folds are
            0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
            8, and 9. Defaults to None.
        allow_incomplete_fold: Whether to allow the last fold to include
            fewer observations than `steps`. If `False`, the last fold is excluded if it
            is incomplete. Defaults to True.
        return_all_indexes: Whether to return all indexes or only the
            start and end indexes of each fold. Defaults to False.
        verbose: Whether to print information about generated folds.
            Defaults to True.

    Attributes:
        steps: Number of observations used to be predicted in each fold.
        initial_train_size: Number of observations used for initial training.
            If `None` or 0, the initial forecaster is not trained in the first fold.
        fold_stride: Number of observations that the start of the test set
            advances between consecutive folds.
        overlapping_folds: Whether the folds overlap.
        window_size: Number of observations needed to generate the
            autoregressive predictors.
        differentiation: Number of observations to use for differentiation.
            This is used to extend the `last_window` as many observations as the
            differentiation order.
        refit: Whether to refit the forecaster in each fold.
        fixed_train_size: Whether the training size is fixed or increases in each fold.
        gap: Number of observations between the end of the training set and the
            start of the test set.
        skip_folds: Number of folds to skip.
        allow_incomplete_fold: Whether to allow the last fold to include fewer
            observations than `steps`.
        return_all_indexes: Whether to return all indexes or only the start
            and end indexes of each fold.
        verbose: Whether to print information about generated folds.

    Examples:
        Basic usage with fixed train size:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.model_selection import TimeSeriesFold
        >>> # Create sample time series data
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> y = pd.Series(np.arange(100), index=dates)
        >>> # Create fold splitter
        >>> cv = TimeSeriesFold(
        ...     steps=10,
        ...     initial_train_size=50,
        ...     refit=True,
        ...     fixed_train_size=True
        ... )
        >>> # Get folds
        >>> folds = cv.split(y)
        >>> print(f"Number of folds: {len(folds)}")
        Number of folds: 4

        Overlapping folds with custom stride:
        >>> cv = TimeSeriesFold(
        ...     steps=30,
        ...     initial_train_size=50,
        ...     fold_stride=7,
        ...     fixed_train_size=False
        ... )
        >>> folds = cv.split(y)
        >>> # First test fold covers [50, 80), second [57, 87), etc.

        Return as pandas DataFrame:
        >>> cv = TimeSeriesFold(steps=10, initial_train_size=50)
        >>> folds_df = cv.split(y, as_pandas=True)
        >>> print(folds_df.columns.tolist())
        ['fold', 'train_start', 'train_end', 'last_window_start', 'last_window_end', 'test_start', 'test_end', 'test_start_with_gap', 'test_end_with_gap', 'fit_forecaster']

        Skip folds for faster evaluation:
        >>> cv = TimeSeriesFold(
        ...     steps=5,
        ...     initial_train_size=50,
        ...     skip_folds=2
        ... )
        >>> folds = cv.split(y)
        >>> # Returns folds 0, 2, 4, 6, ...

    Note:
        Returned values are the positions of the observations and not the actual values of
        the index, so they can be used to slice the data directly using iloc. For example,
        if the input series is `X = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`, the
        `initial_train_size = 3`, `window_size = 2`, `steps = 4`, and `gap = 1`,
        the output of the first fold will: [0, [0, 3], [1, 3], [3, 8], [4, 8], True].

        The first element is the fold number, the first list `[0, 3]` indicates that
        the training set goes from the first to the third observation. The second
        list `[1, 3]` indicates that the last window seen by the forecaster during
        training goes from the second to the third observation. The third list `[3, 8]`
        indicates that the test set goes from the fourth to the eighth observation.
        The fourth list `[4, 8]` indicates that the test set including the gap goes
        from the fifth to the eighth observation. The boolean `False` indicates that
        the forecaster should not be trained in this fold.

        Following the python convention, the start index is inclusive and the end index is
        exclusive. This means that the last index is not included in the slice.

        As an example, with `initial_train_size=50`, `steps=30`, and `fold_stride=7`,
        the first test fold will cover observations [50, 80), the second fold [57, 87),
        and the third fold [64, 94). This configuration produces multiple forecasts
        for the same observations, which is often desirable in rolling-origin
        evaluation.
    """

    def __init__(
        self,
        steps: int,
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

        super().__init__(
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

        self.steps = steps
        self.fold_stride = fold_stride if fold_stride is not None else steps
        self.overlapping_folds = self.fold_stride < self.steps
        self.refit = refit
        self.fixed_train_size = fixed_train_size
        self.gap = gap
        self.skip_folds = skip_folds
        self.allow_incomplete_fold = allow_incomplete_fold

    def __repr__(self) -> str:
        """Information displayed when printed.

        Args:
            None

        Returns:
            String representation of the TimeSeriesFold object.

        Examples:
            >>> from spotforecast2_safe.model_selection import TimeSeriesFold
            >>> cv = TimeSeriesFold(steps=1)
            >>> print(cv)
        """

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Initial train size    = {self.initial_train_size},\n"
            f"Steps                 = {self.steps},\n"
            f"Fold stride           = {self.fold_stride},\n"
            f"Overlapping folds     = {self.overlapping_folds},\n"
            f"Window size           = {self.window_size},\n"
            f"Differentiation       = {self.differentiation},\n"
            f"Refit                 = {self.refit},\n"
            f"Fixed train size      = {self.fixed_train_size},\n"
            f"Gap                   = {self.gap},\n"
            f"Skip folds            = {self.skip_folds},\n"
            f"Allow incomplete fold = {self.allow_incomplete_fold},\n"
            f"Return all indexes    = {self.return_all_indexes},\n"
            f"Verbose               = {self.verbose}\n"
        )

        return info

    def _repr_html_(self) -> str:
        """HTML representation of the object.

        The "General Information" section is expanded by default.

        Returns:
            HTML string representation for Jupyter notebooks.
        """

        style, unique_id = get_style_repr_html()
        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Initial train size:</strong> {self.initial_train_size}</li>
                    <li><strong>Steps:</strong> {self.steps}</li>
                    <li><strong>Fold stride:</strong> {self.fold_stride}</li>
                    <li><strong>Overlapping folds:</strong> {self.overlapping_folds}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Differentiation:</strong> {self.differentiation}</li>
                    <li><strong>Refit:</strong> {self.refit}</li>
                    <li><strong>Fixed train size:</strong> {self.fixed_train_size}</li>
                    <li><strong>Gap:</strong> {self.gap}</li>
                    <li><strong>Skip folds:</strong> {self.skip_folds}</li>
                    <li><strong>Allow incomplete fold:</strong> {self.allow_incomplete_fold}</li>
                    <li><strong>Return all indexes:</strong> {self.return_all_indexes}</li>
                </ul>
            </details>
        </div>
        """

        return style + content

    def split(
        self,
        X: pd.Series | pd.DataFrame | pd.Index | dict[str, pd.Series | pd.DataFrame],
        as_pandas: bool = False,
    ) -> list | pd.DataFrame:
        """Split the time series data into train and test folds.

        Args:
            X: Time series data or index to split. Can be a pandas Series, DataFrame,
                Index, or a dictionary of Series/DataFrames.
            as_pandas: If True, the folds are returned as a DataFrame. This is useful
                to visualize the folds in a more interpretable way. Defaults to False.

        Returns:
            A list of lists containing the indices (position) for each fold, or a
            DataFrame if `as_pandas=True`. Each list contains 4 lists and a boolean
            with the following information:

            - **fold**: fold number.
            - **[train_start, train_end]**: list with the start and end positions of
                    the training set.
            - **[last_window_start, last_window_end]**: list with the start and end
                    positions of the last window seen by the forecaster during training.
                    The last window is used to generate the lags use as predictors. If
                    `differentiation` is included, the interval is extended as many
                    observations as the differentiation order. If the argument `window_size`
                    is `None`, this list is empty.
            - **[test_start, test_end]**: list with the start and end positions of
                    the test set. These are the observations used to evaluate the forecaster.
            - **[test_start_with_gap, test_end_with_gap]**: list with the start and
                    end positions of the test set including the gap. The gap is the number
                    of observations between the end of the training set and the start of
                    the test set.
            - **fit_forecaster**: boolean indicating whether the forecaster should be
                    fitted in this fold.

        Note:
            The returned values are the positions of the observations and not the
            actual values of the index, so they can be used to slice the data directly
            using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'last_window_start',
            'last_window_end', 'test_start', 'test_end', 'test_start_with_gap',
            'test_end_with_gap', 'fit_forecaster'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.
        """

        if not isinstance(X, (pd.Series, pd.DataFrame, pd.Index, dict)):
            raise TypeError(
                f"X must be a pandas Series, DataFrame, Index or a dictionary. "
                f"Got {type(X)}."
            )

        window_size_as_date_offset = isinstance(
            self.window_size, pd.tseries.offsets.DateOffset
        )
        if window_size_as_date_offset:
            # Calculate the window_size in steps. This is not a exact calculation
            # because the offset follows the calendar rules and the distance between
            # two dates may not be constant.
            first_valid_index = X.index[-1] - self.window_size
            try:
                window_size_idx_start = X.index.get_loc(first_valid_index)
                window_size_idx_end = X.index.get_loc(X.index[-1])
                self.window_size = window_size_idx_end - window_size_idx_start
            except KeyError:
                raise ValueError(
                    f"The length of `y` ({len(X)}), must be greater than or equal "
                    f"to the window size ({self.window_size}). This is because  "
                    f"the offset (forecaster.offset) is larger than the available "
                    f"data. Try to decrease the size of the offset (forecaster.offset), "
                    f"the number of `n_offsets` (forecaster.n_offsets) or increase the "
                    f"size of `y`."
                )

        if self.initial_train_size is None:
            if self.window_size is None:
                raise ValueError(
                    "To use split method when `initial_train_size` is None, "
                    "`window_size` must be an integer greater than 0. "
                    "Although no initial training is done and all data is used to "
                    "evaluate the model, the first `window_size` observations are "
                    "needed to create the initial predictors. Got `window_size` = None."
                )
            if self.refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`. "
                    "Set `refit` to `False` if you want to use `initial_train_size = None`."
                )
            externally_fitted = True
            self.initial_train_size = self.window_size  # Reset to None later
        else:
            if self.window_size is None:
                warnings.warn(
                    "Last window cannot be calculated because `window_size` is None.",
                    IgnoredArgumentWarning,
                )
            externally_fitted = False

        index = self._extract_index(X)
        idx = range(len(index))
        folds = []
        i = 0

        self.initial_train_size = date_to_index_position(
            index=index,
            date_input=self.initial_train_size,
            method="validation",
            date_literal="initial_train_size",
        )

        if window_size_as_date_offset:
            if self.initial_train_size is not None:
                if self.initial_train_size < self.window_size:
                    raise ValueError(
                        f"If `initial_train_size` is an integer, it must be greater than "
                        f"the `window_size` of the forecaster ({self.window_size}) "
                        f"and smaller than the length of the series ({len(X)}). If "
                        f"it is a date, it must be within this range of the index."
                    )

        if self.allow_incomplete_fold:
            # At least one observation after the gap to allow incomplete fold
            if len(index) <= self.initial_train_size + self.gap:
                raise ValueError(
                    f"The time series must have more than `initial_train_size + gap` "
                    f"observations to create at least one fold.\n"
                    f"    Time series length: {len(index)}\n"
                    f"    Required > {self.initial_train_size + self.gap}\n"
                    f"    initial_train_size: {self.initial_train_size}\n"
                    f"    gap: {self.gap}\n"
                )
        else:
            # At least one complete fold
            if len(index) < self.initial_train_size + self.gap + self.steps:
                raise ValueError(
                    f"The time series must have at least `initial_train_size + gap + steps` "
                    f"observations to create a minimum of one complete fold "
                    f"(allow_incomplete_fold=False).\n"
                    f"    Time series length: {len(index)}\n"
                    f"    Required >= {self.initial_train_size + self.gap + self.steps}\n"
                    f"    initial_train_size: {self.initial_train_size}\n"
                    f"    gap: {self.gap}\n"
                    f"    steps: {self.steps}\n"
                )

        while self.initial_train_size + (i * self.fold_stride) + self.gap < len(index):

            if self.refit:
                # NOTE: If `fixed_train_size` the train size doesn't increase but
                # moves by `fold_stride` positions in each iteration. If `False`,
                # the train size increases by `fold_stride` in each iteration.
                train_iloc_start = (
                    i * (self.fold_stride) if self.fixed_train_size else 0
                )
                train_iloc_end = self.initial_train_size + i * (self.fold_stride)
                test_iloc_start = train_iloc_end
            else:
                # NOTE: The train size doesn't increase and doesn't move.
                train_iloc_start = 0
                train_iloc_end = self.initial_train_size
                test_iloc_start = self.initial_train_size + i * (self.fold_stride)

            if self.window_size is not None:
                last_window_iloc_start = test_iloc_start - self.window_size

            test_iloc_end = test_iloc_start + self.gap + self.steps

            partitions = [
                idx[train_iloc_start:train_iloc_end],
                (
                    idx[last_window_iloc_start:test_iloc_start]
                    if self.window_size is not None
                    else []
                ),
                idx[test_iloc_start:test_iloc_end],
                idx[test_iloc_start + self.gap : test_iloc_end],
            ]
            folds.append(partitions)
            i += 1

        # NOTE: Delete all incomplete folds at the end if not allowed
        n_removed_folds = 0
        if not self.allow_incomplete_fold:
            # NOTE: While folds and the last "test_index_with_gap" is incomplete,
            # calculating len of range objects
            while folds and len(folds[-1][3]) < self.steps:
                folds.pop()
                n_removed_folds += 1

        # Replace partitions inside folds with length 0 with `None`
        folds = [
            [partition if len(partition) > 0 else None for partition in fold]
            for fold in folds
        ]

        # Create a flag to know whether to train the forecaster
        if self.refit == 0:
            self.refit = False

        if isinstance(self.refit, bool):
            fit_forecaster = [self.refit] * len(folds)
            fit_forecaster[0] = True
        else:
            fit_forecaster = [False] * len(folds)
            for i in range(0, len(fit_forecaster), self.refit):
                fit_forecaster[i] = True

        for i in range(len(folds)):
            folds[i].insert(0, i)
            folds[i].append(fit_forecaster[i])
            if fit_forecaster[i] is False:
                folds[i][1] = folds[i - 1][1]

        index_to_skip = []
        if self.skip_folds is not None:
            if isinstance(self.skip_folds, (int, np.integer)) and self.skip_folds > 0:
                index_to_keep = np.arange(0, len(folds), self.skip_folds)
                index_to_skip = np.setdiff1d(
                    np.arange(0, len(folds)), index_to_keep, assume_unique=True
                )
                index_to_skip = [
                    int(x) for x in index_to_skip
                ]  # Required since numpy 2.0
            if isinstance(self.skip_folds, list):
                index_to_skip = [i for i in self.skip_folds if i < len(folds)]

        if self.verbose:
            self._print_info(
                index=index,
                folds=folds,
                externally_fitted=externally_fitted,
                n_removed_folds=n_removed_folds,
                index_to_skip=index_to_skip,
            )

        folds = [fold for i, fold in enumerate(folds) if i not in index_to_skip]
        if not self.return_all_indexes:
            # NOTE: +1 to prevent iloc pandas from deleting the last observation
            folds = [
                [
                    fold[0],
                    [fold[1][0], fold[1][-1] + 1],
                    (
                        [fold[2][0], fold[2][-1] + 1]
                        if self.window_size is not None
                        else []
                    ),
                    [fold[3][0], fold[3][-1] + 1],
                    [fold[4][0], fold[4][-1] + 1],
                    fold[5],
                ]
                for fold in folds
            ]

        if externally_fitted:
            self.initial_train_size = None
            folds[0][5] = False

        if as_pandas:
            if self.window_size is None:
                for fold in folds:
                    fold[2] = [None, None]

            if not self.return_all_indexes:
                folds = pd.DataFrame(
                    data=[
                        [fold[0]] + list(itertools.chain(*fold[1:-1])) + [fold[-1]]
                        for fold in folds
                    ],
                    columns=[
                        "fold",
                        "train_start",
                        "train_end",
                        "last_window_start",
                        "last_window_end",
                        "test_start",
                        "test_end",
                        "test_start_with_gap",
                        "test_end_with_gap",
                        "fit_forecaster",
                    ],
                )
            else:
                folds = pd.DataFrame(
                    data=folds,
                    columns=[
                        "fold",
                        "train_index",
                        "last_window_index",
                        "test_index",
                        "test_index_with_gap",
                        "fit_forecaster",
                    ],
                )

        return folds

    def _print_info(
        self,
        index: pd.Index,
        folds: list[list[int]],
        externally_fitted: bool,
        n_removed_folds: int,
        index_to_skip: list[int],
    ) -> None:
        """Print information about folds.

        Args:
            index: Index of the time series data.
            folds: A list of lists containing the indices (position) for each fold.
            externally_fitted: Whether an already trained forecaster is to be used.
            n_removed_folds: Number of folds removed.
            index_to_skip: Number of folds skipped.
        """

        print("Information of folds")
        print("--------------------")
        if externally_fitted:
            print(
                f"An already trained forecaster is to be used. Window size: "
                f"{self.window_size}"
            )
        else:
            if self.differentiation is None:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size}"
                )
            else:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size - self.differentiation}"
                )
                print(
                    f"    First {self.differentiation} observation/s in training sets "
                    f"are used for differentiation"
                )
        print(
            f"Number of observations used for backtesting: "
            f"{len(index) - self.initial_train_size}"
        )
        print(f"    Number of folds: {len(folds)}")
        print(
            f"    Number skipped folds: "
            f"{len(index_to_skip)} {index_to_skip if index_to_skip else ''}"
        )
        print(f"    Number of steps per fold: {self.steps}")
        if self.steps != self.fold_stride:
            print(
                f"    Number of steps to the next fold (fold stride): {self.fold_stride}"
            )
        print(
            f"    Number of steps to exclude between last observed data "
            f"(last window) and predictions (gap): {self.gap}"
        )
        if n_removed_folds > 0:
            print(
                f"    The last {n_removed_folds} fold(s) have been excluded "
                f"because they were incomplete."
            )

        if len(folds[-1][4]) < self.steps:
            print(f"    Last fold only includes {len(folds[-1][4])} observations.")

        print("")

        if self.differentiation is None:
            differentiation = 0
        else:
            differentiation = self.differentiation

        for i, fold in enumerate(folds):
            is_fold_skipped = i in index_to_skip
            has_training = fold[-1] if i != 0 else True
            training_start = (
                index[fold[1][0] + differentiation] if fold[1] is not None else None
            )
            training_end = index[fold[1][-1]] if fold[1] is not None else None
            training_length = (
                len(fold[1]) - differentiation if fold[1] is not None else 0
            )
            validation_start = index[fold[4][0]]
            validation_end = index[fold[4][-1]]
            validation_length = len(fold[4])

            print(f"Fold: {i}")
            if is_fold_skipped:
                print("    Fold skipped")
            elif not externally_fitted and has_training:
                print(
                    f"    Training:   {training_start} -- {training_end}  "
                    f"(n={training_length})"
                )
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )
            else:
                print("    Training:   No training in this fold")
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )

        print("")
