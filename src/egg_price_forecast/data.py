"""Data fetching for egg price forecasting.

Fetches BLS egg price data from FRED and USDA wholesale prices.
"""

import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

_FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_FRED_URL = _FRED_CSV_BASE + "APU0000708111"
_USDA_URL = (
    "https://usda.library.cornell.edu/apidatasets/MARKETNEWS/data"
    "?reportId=2848&frequencyId=1"
)
_CACHE_DIR = Path.home() / ".cache" / "egg-price-forecast"
_CACHE_TTL_SECONDS = 3600  # 1 hour

# BLS Average Price series available from FRED
COMMODITY_SERIES: dict[str, str] = {
    "eggs": "APU0000708111",
    "ground_beef": "APU0000703112",
    "chicken_breast": "APU0000706111",
    "whole_milk": "APU0000709112",
    "white_bread": "APU0000702111",
    "flour": "APU0000701111",
    "rice": "APU0000701312",
    "bananas": "APU0000711211",
    "tomatoes": "APU0000712311",
    "potatoes": "APU0000712112",
    "sugar": "APU0000715211",
    "butter": "APU0000FS1101",
    "cheddar_cheese": "APU0000710212",
    "orange_juice": "APU0000713111",
    "coffee": "APU0000717311",
}

_CPI_EXPECTATIONS_SERIES = "EXPINF1YR"
PPI_EGGS_SERIES = "WPU0171"


def _cache_path() -> Path:
    """Return the path for the cached BLS egg price CSV."""
    return _CACHE_DIR / "bls_eggs.csv"


def fetch_bls_series(series_id: str) -> pd.Series:
    """Fetch a BLS Average Price series from FRED by series ID.

    Downloads a CSV from FRED's public endpoint (no API key needed).
    Results are cached locally with a 1-hour TTL.

    Args:
        series_id: FRED series ID (e.g. "APU0000708111").

    Returns:
        pd.Series with DatetimeIndex and float values.
    """
    cache_file = _CACHE_DIR / f"{series_id}.csv"

    # Check cache
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_TTL_SECONDS:
            df = pd.read_csv(
                cache_file,
                parse_dates=["observation_date"],
                index_col="observation_date",
            )
            series = pd.to_numeric(df[series_id], errors="coerce").dropna()
            series.index = pd.DatetimeIndex(series.index)
            return series

    # Fetch from FRED
    url = _FRED_CSV_BASE + series_id
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)

    # Parse
    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    series = pd.to_numeric(df[series_id], errors="coerce").dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series


def fetch_all_commodities() -> dict[str, pd.Series]:
    """Fetch all BLS commodity price series.

    Returns:
        Dict mapping commodity name to price Series.
    """
    result = {}
    for name, series_id in COMMODITY_SERIES.items():
        try:
            result[name] = fetch_bls_series(series_id)
        except Exception:
            continue
    return result


def fetch_cpi_expectations() -> pd.Series:
    """Fetch 1-year expected inflation rate from Cleveland Fed Nowcast.

    Uses FRED series EXPINF1YR.

    Returns:
        pd.Series with DatetimeIndex and float values (percentage points).
    """
    return fetch_bls_series(_CPI_EXPECTATIONS_SERIES)


def fetch_ppi_eggs() -> pd.Series:
    """Fetch PPI Chicken Eggs index (WPU0171) from FRED.

    Monthly wholesale egg price index (1982=100), available from 1971.

    Returns:
        pd.Series with DatetimeIndex and float values.
    """
    return fetch_bls_series(PPI_EGGS_SERIES)


def fetch_bls_eggs() -> pd.Series:
    """Fetch BLS average egg price series APU0000708111 from FRED.

    Downloads a CSV from FRED's public endpoint (no API key needed).
    Results are cached locally with a 1-hour TTL.

    Returns:
        pd.Series with DatetimeIndex and float values (price per dozen).
    """
    cache_file = _cache_path()

    # Check cache
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_TTL_SECONDS:
            df = pd.read_csv(cache_file, parse_dates=["observation_date"], index_col="observation_date")
            series = pd.to_numeric(df["APU0000708111"], errors="coerce").dropna()
            series.index = pd.DatetimeIndex(series.index)
            return series

    # Fetch from FRED
    resp = requests.get(_FRED_URL, timeout=30)
    resp.raise_for_status()

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)

    # Parse
    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"], index_col="observation_date")
    series = pd.to_numeric(df["APU0000708111"], errors="coerce").dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series


def get_wholesale_prices() -> float | None:
    """Fetch latest USDA AMS shell egg wholesale price.

    Queries the USDA Market News API for shell egg prices and returns
    the midpoint of the 'mostly' price range in $/dozen.

    Returns:
        Float price in $/dozen, or None if fetching/parsing fails.
    """
    try:
        resp = requests.get(_USDA_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return None

        # Use the first record's mostly_low and mostly_high prices
        record = data[0]
        mostly_low = float(record["mostly_low_price"])
        mostly_high = float(record["mostly_high_price"])

        return (mostly_low + mostly_high) / 2.0

    except Exception:
        return None
