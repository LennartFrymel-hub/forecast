import pandas as pd

from spotforecast2_safe.data import Period
from spotforecast2_safe.manager.configurator.config_entsoe import ConfigEntsoe


def test_config_entsoe_defaults():
    """Test that ConfigEntsoe initializes with correct defaults."""
    config = ConfigEntsoe()
    assert config.API_COUNTRY_CODE == "DE"
    assert config.predict_size == 24
    assert config.refit_size == 7
    assert config.random_state == 314159
    assert len(config.periods) == 5
    assert isinstance(config.train_size, pd.Timedelta)
    assert config.train_size == pd.Timedelta(days=3 * 365)


def test_config_entsoe_custom():
    """Test custom overrides in ConfigEntsoe."""
    custom_periods = [
        Period(name="test", n_periods=1, column="col", input_range=(0, 1))
    ]
    config = ConfigEntsoe(
        api_country_code="FR", predict_size=48, random_state=42, periods=custom_periods
    )
    assert config.API_COUNTRY_CODE == "FR"
    assert config.predict_size == 48
    assert config.random_state == 42
    assert config.periods == custom_periods


def test_config_entsoe_timedeltas():
    """Test timedelta parameters."""
    train_size = pd.Timedelta(days=10)
    delta_val = pd.Timedelta(hours=5)
    config = ConfigEntsoe(train_size=train_size, delta_val=delta_val)
    assert config.train_size == train_size
    assert config.delta_val == delta_val


def test_config_entsoe_root_import():
    """Test that Config can be imported from the root package."""
    from spotforecast2_safe import Config
    from spotforecast2_safe import ConfigEntsoe as CE

    assert Config is CE

    config = Config()
    assert isinstance(config, CE)
