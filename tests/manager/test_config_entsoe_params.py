import pandas as pd
import pytest

from spotforecast2_safe.data import Period
from spotforecast2_safe.manager.configurator.config_entsoe import ConfigEntsoe


def test_config_entsoe_get_params():
    """Test get_params returns all expected parameters."""
    config = ConfigEntsoe(api_country_code="ES", predict_size=12)
    params = config.get_params()

    assert params["api_country_code"] == "ES"
    assert params["predict_size"] == 12
    assert "random_state" in params
    assert "periods" in params
    assert "lags_consider" in params
    assert "train_size" in params
    assert "end_train_default" in params
    assert "delta_val" in params
    assert "refit_size" in params
    assert "n_hyperparameters_trials" in params
    assert "data_filename" in params


def test_config_entsoe_set_params_kwargs():
    """Test set_params with keyword arguments."""
    config = ConfigEntsoe()
    config.set_params(api_country_code="FR", predict_size=48)

    assert config.API_COUNTRY_CODE == "FR"
    assert config.predict_size == 48


def test_config_entsoe_set_params_dict():
    """Test set_params with a dictionary."""
    config = ConfigEntsoe()
    config.set_params({"api_country_code": "IT", "random_state": 123})

    assert config.API_COUNTRY_CODE == "IT"
    assert config.random_state == 123


def test_config_entsoe_set_params_chaining():
    """Test set_params supports method chaining."""
    config = (
        ConfigEntsoe().set_params(api_country_code="GB").set_params(predict_size=72)
    )

    assert config.API_COUNTRY_CODE == "GB"
    assert config.predict_size == 72


def test_config_entsoe_set_params_invalid():
    """Test set_params raises ValueError for invalid parameters."""
    config = ConfigEntsoe()
    with pytest.raises(ValueError, match="Invalid parameter invalid_param"):
        config.set_params(invalid_param=True)


def test_config_entsoe_set_params_complex_types():
    """Test set_params with complex types like Period and Timedelta."""
    config = ConfigEntsoe()
    new_periods = [
        Period(name="custom", n_periods=5, column="test", input_range=(0, 4))
    ]
    new_train_size = pd.Timedelta(days=1)

    config.set_params(periods=new_periods, train_size=new_train_size)

    assert config.periods == new_periods
    assert config.train_size == new_train_size


def test_config_entsoe_get_params_deep():
    """Test get_params with deep=True returns nested period parameters."""
    config = ConfigEntsoe()
    params = config.get_params(deep=True)

    # daily period params
    assert "periods__daily__n_periods" in params
    assert params["periods__daily__n_periods"] == 12
    assert params["periods__daily__column"] == "hour"
    assert params["periods__daily__input_range"] == (1, 24)

    # weekly params
    assert "periods__weekly__column" in params


def test_config_entsoe_set_params_deep():
    """Test set_params handles deep parameter updates correctly for frozen Period dataclasses."""
    config = ConfigEntsoe()

    # Check baseline
    daily_period = next(p for p in config.periods if p.name == "daily")
    assert daily_period.n_periods == 12

    # Update deep parameters using Scikit-Learn convention
    config.set_params(periods__daily__n_periods=24, periods__weekly__column="iso_week")

    # Verify update (replace creates a new object so we check the new list)
    daily_period_updated = next(p for p in config.periods if p.name == "daily")
    weekly_period_updated = next(p for p in config.periods if p.name == "weekly")

    assert daily_period_updated.n_periods == 24
    assert weekly_period_updated.column == "iso_week"


def test_config_entsoe_set_params_deep_invalid():
    """Test deep parameter setting raises exception for non-existent entities."""
    config = ConfigEntsoe()
    with pytest.raises(
        ValueError, match="Period with name 'nonexistent' not found in configuration."
    ):
        config.set_params(periods__nonexistent__n_periods=10)
