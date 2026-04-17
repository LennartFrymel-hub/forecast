# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
One step ahead cross-validation splitting.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from spotforecast2_safe.forecaster.utils import (
    date_to_index_position,
    get_style_repr_html,
)

from .split_base import BaseFold


class OneStepAheadFold(BaseFold):
    """
    Class to split time series data into train and test folds for one-step-ahead
    forecasting.

    Args:
        initial_train_size (int | str | pd.Timestamp): Number of observations used
            for initial training.

            - If an integer, the number of observations used for initial training.
            - If a date string or pandas Timestamp, it is the last date included in
              the initial training set.
        window_size (int, optional): Number of observations needed to generate the
            autoregressive predictors. Defaults to None.
        differentiation (int, optional): Number of observations to use for differentiation.
            This is used to extend the `last_window` as many observations as the
            differentiation order. Defaults to None.
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
    """

    # Class-level type annotations for static analysis tools
    initial_train_size: int | str | pd.Timestamp
    window_size: int | None
    differentiation: int | None
    return_all_indexes: bool
    verbose: bool

    def __init__(
        self,
        initial_train_size: int | str | pd.Timestamp,
        window_size: int | None = None,
        differentiation: int | None = None,
        return_all_indexes: bool = False,
        verbose: bool = True,
    ) -> None:

        super().__init__(
            initial_train_size=initial_train_size,
            window_size=window_size,
            differentiation=differentiation,
            return_all_indexes=return_all_indexes,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        """
        Information displayed when printed.
        """

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Initial train size = {self.initial_train_size},\n"
            f"Window size        = {self.window_size},\n"
            f"Differentiation    = {self.differentiation},\n"
            f"Return all indexes = {self.return_all_indexes},\n"
            f"Verbose            = {self.verbose}\n"
        )

        return info

    def _repr_html_(self) -> str:
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        style, unique_id = get_style_repr_html()
        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Initial train size:</strong> {self.initial_train_size}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Differentiation:</strong> {self.differentiation}</li>
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
        externally_fitted: Any = None,
    ) -> list | pd.DataFrame:
        """
        Split the time series data into train and test folds.

        Args:
            X (pd.Series | pd.DataFrame | pd.Index | dict): Time series data or index to split.
            as_pandas (bool, optional): If True, the folds are returned as a DataFrame.
                This is useful to visualize the folds in a more interpretable way.
                Defaults to False.
            externally_fitted (Any, optional): This argument is not used in this class.
                It is included for API consistency. Defaults to None.

        Returns:
            list | pd.DataFrame: A list of lists containing the indices (position) of
            the fold. The list contains 2 lists with the following information:

            - fold: fold number.
            - [train_start, train_end]: list with the start and end positions of the
                training set.
            - [test_start, test_end]: list with the start and end positions of the test
                set. These are the observations used to evaluate the forecaster.
            - fit_forecaster: boolean indicating whether the forecaster should be fitted
                in this fold.

            It is important to note that the returned values are the positions of the
            observations and not the actual values of the index, so they can be used to
            slice the data directly using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'test_start',
            'test_end', 'fit_forecaster'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.
        """

        if not isinstance(X, (pd.Series, pd.DataFrame, pd.Index, dict)):
            raise TypeError(
                f"X must be a pandas Series, DataFrame, Index or a dictionary. "
                f"Got {type(X)}."
            )

        index = self._extract_index(X)

        self.initial_train_size = date_to_index_position(
            index=index,
            date_input=self.initial_train_size,
            method="validation",
            date_literal="initial_train_size",
        )

        fold = [
            0,
            [0, self.initial_train_size - 1],
            [self.initial_train_size, len(X)],
            True,
        ]

        if self.verbose:
            self._print_info(index=index, fold=fold)

        # NOTE: +1 to prevent iloc pandas from deleting the last observation
        if self.return_all_indexes:
            fold = [
                fold[0],
                [range(fold[1][0], fold[1][1] + 1)],
                [range(fold[2][0], fold[2][1])],
                fold[3],
            ]
        else:
            fold = [
                fold[0],
                [fold[1][0], fold[1][1] + 1],
                [fold[2][0], fold[2][1]],
                fold[3],
            ]

        if as_pandas:
            if not self.return_all_indexes:
                fold = pd.DataFrame(
                    data=[[fold[0]] + list(itertools.chain(*fold[1:-1])) + [fold[-1]]],
                    columns=[
                        "fold",
                        "train_start",
                        "train_end",
                        "test_start",
                        "test_end",
                        "fit_forecaster",
                    ],
                )
            else:
                fold = pd.DataFrame(
                    data=[fold],
                    columns=["fold", "train_index", "test_index", "fit_forecaster"],
                )

        return fold

    def _print_info(self, index: pd.Index, fold: list[list[int]]) -> None:
        """
        Print information about folds.

        Args:
            index (pd.Index): Index of the time series data.
            fold (list): A list of lists containing the indices (position) of the fold.
        """

        if self.differentiation is None:
            differentiation = 0
        else:
            differentiation = self.differentiation

        initial_train_size = self.initial_train_size - differentiation
        test_length = len(index) - (initial_train_size + differentiation)

        print("Information of folds")
        print("--------------------")
        print(f"Number of observations in train: {initial_train_size}")
        if self.differentiation is not None:
            print(
                f"    First {differentiation} observation/s in training set "
                f"are used for differentiation"
            )
        print(f"Number of observations in test: {test_length}")

        training_start = index[fold[1][0] + differentiation]
        training_end = index[fold[1][-1]]
        test_start = index[fold[2][0]]
        test_end = index[fold[2][-1] - 1]

        print(f"Training : {training_start} -- {training_end} (n={initial_train_size})")
        print(f"Test     : {test_start} -- {test_end} (n={test_length})")
        print("")
