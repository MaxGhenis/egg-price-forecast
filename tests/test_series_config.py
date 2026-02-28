"""Tests for series_config module."""

from egg_price_forecast.series_config import SERIES_CONFIGS, SeriesConfig
from egg_price_forecast.data import COMMODITY_SERIES


def test_all_commodities_have_configs():
    """Every commodity in data.COMMODITY_SERIES has a matching config."""
    for name in COMMODITY_SERIES:
        assert name in SERIES_CONFIGS, f"Missing config for commodity: {name}"


def test_fred_ids_match_data_module():
    """FRED IDs in configs match those in data.COMMODITY_SERIES."""
    for name, config in SERIES_CONFIGS.items():
        assert config.fred_id == COMMODITY_SERIES[name], (
            f"{name}: config fred_id {config.fred_id} != "
            f"data module {COMMODITY_SERIES[name]}"
        )


def test_all_configs_are_series_config():
    """Every value in SERIES_CONFIGS is a SeriesConfig instance."""
    for name, config in SERIES_CONFIGS.items():
        assert isinstance(config, SeriesConfig), f"{name} is not a SeriesConfig"


def test_required_fields_present():
    """All configs have non-empty required fields."""
    for name, config in SERIES_CONFIGS.items():
        assert config.name, f"{name}: empty name"
        assert config.fred_id, f"{name}: empty fred_id"
        assert config.display_name, f"{name}: empty display_name"
        assert config.unit, f"{name}: empty unit"


def test_config_name_matches_key():
    """The config.name matches its dict key."""
    for key, config in SERIES_CONFIGS.items():
        assert config.name == key, f"Key {key} != config.name {config.name}"


def test_covariates_are_dicts_with_fred_ids():
    """All covariates have string keys and FRED-format IDs."""
    for name, config in SERIES_CONFIGS.items():
        assert isinstance(config.covariates, dict), f"{name}: covariates not dict"
        for cov_name, cov_id in config.covariates.items():
            assert isinstance(cov_name, str), f"{name}/{cov_name}: key not str"
            assert isinstance(cov_id, str), f"{name}/{cov_name}: ID not str"
            assert len(cov_id) > 0, f"{name}/{cov_name}: empty FRED ID"


def test_defaults():
    """Default values are sensible."""
    for name, config in SERIES_CONFIGS.items():
        assert config.covariate_max_lag == 12, f"{name}: unexpected max_lag"
        assert config.seasonal_harmonics == 2, f"{name}: unexpected harmonics"
        assert config.forecast_horizon == 1, f"{name}: unexpected horizon"


def test_eggs_has_ppi_covariate():
    """Eggs specifically should have PPI eggs as a covariate."""
    eggs = SERIES_CONFIGS["eggs"]
    assert "ppi_eggs" in eggs.covariates
    assert eggs.covariates["ppi_eggs"] == "WPU0171"


def test_all_configs_have_cpi_expectations():
    """Every commodity should have CPI expectations as a covariate."""
    for name, config in SERIES_CONFIGS.items():
        assert "cpi_expectations" in config.covariates, (
            f"{name}: missing cpi_expectations covariate"
        )
        assert config.covariates["cpi_expectations"] == "EXPINF1YR"


def test_frozen_dataclass():
    """SeriesConfig should be immutable."""
    config = SERIES_CONFIGS["eggs"]
    try:
        config.name = "not_eggs"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected — frozen dataclass
