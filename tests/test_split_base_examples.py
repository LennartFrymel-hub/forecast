import pandas as pd

from spotforecast2_safe.model_selection import TimeSeriesFold


def test_set_params():
    cv = TimeSeriesFold(steps=1)
    cv.set_params(
        {
            "steps": 2,
            "initial_train_size": 10,
            "fold_stride": 2,
            "window_size": 5,
            "differentiation": 1,
            "refit": True,
            "fixed_train_size": False,
            "gap": 1,
            "skip_folds": 2,
            "allow_incomplete_fold": False,
            "return_all_indexes": True,
            "verbose": False,
        }
    )
    # Check that parameters are set correctly
    assert cv.steps == 2
    assert cv.initial_train_size == 10
    assert cv.fold_stride == 2
    assert cv.window_size == 5
    assert cv.differentiation == 1
    assert cv.refit is True
    assert cv.fixed_train_size is False
    assert cv.gap == 1
    assert cv.skip_folds == 2
    assert cv.allow_incomplete_fold is False
    assert cv.return_all_indexes is True
    assert cv.verbose is False


def test_extract_index_with_datetimeindex():
    cv = TimeSeriesFold(steps=1)
    idx = cv._extract_index(
        pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
    )
    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx) == list(pd.date_range("2020-01-01", periods=3))


def test_validate_params():
    cv = TimeSeriesFold(steps=1)
    # Should not raise
    cv._validate_params(
        cv_name="TimeSeriesFold",
        steps=1,
        initial_train_size=1,
        fold_stride=1,
        window_size=1,
        differentiation=1,
        refit=False,
        fixed_train_size=True,
        gap=0,
        skip_folds=None,
        allow_incomplete_fold=True,
        return_all_indexes=False,
        verbose=True,
    )


def test_validate_params_example():
    """
    Test the example from BaseFold._validate_params docstring.
    Note: The original example cv = TimeSeriesFold() might fail because steps is required.
    """
    # The corrected example code:
    cv = TimeSeriesFold(steps=1)

    cv._validate_params(
        cv_name="TimeSeriesFold",
        steps=1,
        initial_train_size=1,
        fold_stride=1,
        window_size=1,
        differentiation=1,
        refit=False,
        fixed_train_size=True,
        gap=0,
        skip_folds=None,
        allow_incomplete_fold=True,
        return_all_indexes=False,
        verbose=True,
    )
