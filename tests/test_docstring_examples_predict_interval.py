import numpy as np
import pandas as pd


def test_docstring_example_recursive_predict_bootstrapping():
    # Example from _recursive_predict_bootstrapping docstring
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    y = pd.Series(np.arange(10))
    _ = forecaster.fit(y=y, store_in_sample_residuals=True)

    last_window = np.array([8, 9])
    residuals = np.random.normal(size=(3, 5))  # 3 steps, 5 boots

    preds = forecaster._recursive_predict_bootstrapping(
        steps=3,
        last_window_values=last_window,
        sampled_residuals=residuals,
        use_binned_residuals=False,
        n_boot=5,
    )
    assert preds.shape == (3, 5)


def test_docstring_example_predict_bootstrapping():
    # Example from predict_bootstrapping docstring
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    _ = forecaster.fit(y=y, store_in_sample_residuals=True)

    boot_preds = forecaster.predict_bootstrapping(steps=3, n_boot=5)
    assert boot_preds.shape == (3, 5)


def test_docstring_example_predict_interval_conformal():
    # Example from _predict_interval_conformal docstring
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    _ = forecaster.fit(y=y, store_in_sample_residuals=True)

    preds = forecaster._predict_interval_conformal(steps=3, nominal_coverage=0.9)
    assert preds.columns.tolist() == ["pred", "lower_bound", "upper_bound"]


def test_docstring_example_predict_interval():
    # Example from predict_interval docstring
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), name="y")
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    _ = forecaster.fit(y=y, store_in_sample_residuals=True)

    # Bootstrapping method
    intervals_boot = forecaster.predict_interval(
        steps=3, method="bootstrapping", interval=[5, 95]
    )
    assert intervals_boot.columns.tolist() == ["pred", "lower_bound", "upper_bound"]

    # Conformal method
    intervals_conf = forecaster.predict_interval(
        steps=3, method="conformal", interval=0.95
    )
    assert intervals_conf.columns.tolist() == ["pred", "lower_bound", "upper_bound"]
