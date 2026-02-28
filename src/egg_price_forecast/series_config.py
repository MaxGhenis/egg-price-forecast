"""Declarative series registry for BSTS price forecasting.

Each commodity gets a SeriesConfig specifying its FRED ID, display metadata,
and which covariates (with FRED IDs) to use as exogenous regressors in the
structural time series model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterventionConfig:
    """Configuration for an intervention regressor (e.g. HPAI supply shock).

    Attributes:
        onset_date: ISO date string for the start of the intervention.
        decay: Geometric decay rate per month (0 < decay < 1).
        label: Short label used as column name prefix.
    """

    onset_date: str
    decay: float
    label: str


@dataclass(frozen=True)
class SeriesConfig:
    """Configuration for a single commodity price series.

    Attributes:
        name: Internal commodity key (matches data.COMMODITY_SERIES keys).
        fred_id: FRED series ID for the retail CPI price.
        display_name: Human-readable name for display.
        unit: Price unit description (e.g. "$/dozen").
        covariates: Mapping of covariate name to FRED series ID.
            These become Almon PDL exogenous regressors.
        covariate_max_lag: Maximum lag months for Almon PDL (default 12).
        seasonal_harmonics: Number of trigonometric seasonal harmonics (default 2).
        forecast_horizon: Months ahead to forecast (default 1).
        interventions: Tuple of InterventionConfig for structural breaks.
        use_deterministic_seasonal: If True, use month-of-year dummies instead
            of stochastic trig seasonal (prevents one-off events from being
            learned as recurring seasonal patterns).
    """

    name: str
    fred_id: str
    display_name: str
    unit: str
    covariates: dict[str, str] = field(default_factory=dict)
    covariate_max_lag: int = 12
    seasonal_harmonics: int = 2
    forecast_horizon: int = 1
    interventions: tuple[InterventionConfig, ...] = ()
    use_deterministic_seasonal: bool = False


# PPI / wholesale series IDs used as covariates
_PPI_EGGS = "WPU0171"  # PPI Chicken Eggs
_PPI_MEATS = "WPU0132"  # PPI Slaughter Cattle (proxy for beef wholesale)
_PPI_POULTRY = "WPU0175"  # PPI Processed Poultry
_PPI_DAIRY = "WPU022"  # PPI Dairy Products
_PPI_CEREAL = "WPU021"  # PPI Cereal and Bakery Products
_PPI_FRUITS_VEG = "WPU013"  # PPI Fresh Fruits and Vegetables
_PPI_SUGAR = "WPU0621"  # PPI Refined Sugar
_PPI_FATS_OILS = "WPU0171"  # PPI Fats and Oils (same bucket as eggs)
_PPI_BEVERAGE_MATERIALS = "WPU0282"  # PPI Roasted Coffee

# Common covariate: 1-year inflation expectations
_CPI_EXPECTATIONS = "EXPINF1YR"


SERIES_CONFIGS: dict[str, SeriesConfig] = {
    "eggs": SeriesConfig(
        name="eggs",
        fred_id="APU0000708111",
        display_name="Eggs (Grade A, large, dozen)",
        unit="$/dozen",
        covariates={
            "ppi_eggs": _PPI_EGGS,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
        interventions=(
            InterventionConfig(
                onset_date="2014-11-01",
                decay=0.85,
                label="hpai_2015",
            ),
            InterventionConfig(
                onset_date="2024-01-01",
                decay=0.90,
                label="hpai_2024",
            ),
        ),
        use_deterministic_seasonal=True,
    ),
    "ground_beef": SeriesConfig(
        name="ground_beef",
        fred_id="APU0000703112",
        display_name="Ground beef (100% beef, per lb)",
        unit="$/lb",
        covariates={
            "ppi_cattle": _PPI_MEATS,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "chicken_breast": SeriesConfig(
        name="chicken_breast",
        fred_id="APU0000706111",
        display_name="Chicken breast (boneless, per lb)",
        unit="$/lb",
        covariates={
            "ppi_poultry": _PPI_POULTRY,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "whole_milk": SeriesConfig(
        name="whole_milk",
        fred_id="APU0000709112",
        display_name="Whole milk (per gallon)",
        unit="$/gallon",
        covariates={
            "ppi_dairy": _PPI_DAIRY,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "white_bread": SeriesConfig(
        name="white_bread",
        fred_id="APU0000702111",
        display_name="White bread (per lb)",
        unit="$/lb",
        covariates={
            "ppi_cereal": _PPI_CEREAL,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "flour": SeriesConfig(
        name="flour",
        fred_id="APU0000701111",
        display_name="Flour (white, all purpose, per lb)",
        unit="$/lb",
        covariates={
            "ppi_cereal": _PPI_CEREAL,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "rice": SeriesConfig(
        name="rice",
        fred_id="APU0000701312",
        display_name="Rice (white, long grain, per lb)",
        unit="$/lb",
        covariates={
            "ppi_cereal": _PPI_CEREAL,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "bananas": SeriesConfig(
        name="bananas",
        fred_id="APU0000711211",
        display_name="Bananas (per lb)",
        unit="$/lb",
        covariates={
            "ppi_fruits_veg": _PPI_FRUITS_VEG,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "tomatoes": SeriesConfig(
        name="tomatoes",
        fred_id="APU0000712311",
        display_name="Tomatoes (field grown, per lb)",
        unit="$/lb",
        covariates={
            "ppi_fruits_veg": _PPI_FRUITS_VEG,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "potatoes": SeriesConfig(
        name="potatoes",
        fred_id="APU0000712112",
        display_name="Potatoes (white, per lb)",
        unit="$/lb",
        covariates={
            "ppi_fruits_veg": _PPI_FRUITS_VEG,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "sugar": SeriesConfig(
        name="sugar",
        fred_id="APU0000715211",
        display_name="Sugar (white, per lb)",
        unit="$/lb",
        covariates={
            "ppi_sugar": _PPI_SUGAR,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "butter": SeriesConfig(
        name="butter",
        fred_id="APU0000FS1101",
        display_name="Butter (salted, Grade AA, per lb)",
        unit="$/lb",
        covariates={
            "ppi_dairy": _PPI_DAIRY,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "cheddar_cheese": SeriesConfig(
        name="cheddar_cheese",
        fred_id="APU0000710212",
        display_name="Cheddar cheese (natural, per lb)",
        unit="$/lb",
        covariates={
            "ppi_dairy": _PPI_DAIRY,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "orange_juice": SeriesConfig(
        name="orange_juice",
        fred_id="APU0000713111",
        display_name="Orange juice (frozen concentrate, 12 oz)",
        unit="$/12oz",
        covariates={
            "ppi_fruits_veg": _PPI_FRUITS_VEG,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
    "coffee": SeriesConfig(
        name="coffee",
        fred_id="APU0000717311",
        display_name="Coffee (ground roast, per lb)",
        unit="$/lb",
        covariates={
            "ppi_coffee": _PPI_BEVERAGE_MATERIALS,
            "cpi_expectations": _CPI_EXPECTATIONS,
        },
    ),
}
