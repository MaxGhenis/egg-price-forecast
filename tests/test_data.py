"""Tests for egg_price_forecast.data module."""

import os
import time
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


SAMPLE_CSV = """observation_date,APU0000708111
1980-01-01,0.879
1980-02-01,0.822
1980-03-01,0.812
2024-12-01,3.650
2025-01-01,4.950
"""

SAMPLE_USDA_JSON = [
    {
        "report_date": "02/25/2026",
        "market": "New York",
        "slug_line": "Eggs, shell, large, white",
        "low_price": "8.00",
        "high_price": "8.50",
        "mostly_low_price": "8.10",
        "mostly_high_price": "8.40",
    }
]


class TestFetchBlsEggs:
    """Tests for fetch_bls_eggs."""

    def test_returns_series_with_datetime_index(self, mocker, tmp_path):
        """fetch_bls_eggs should return a pd.Series with DatetimeIndex."""
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        # Use tmp_path for cache to avoid polluting real cache
        cache_dir = tmp_path / "cache"
        mocker.patch(
            "egg_price_forecast.data._cache_path",
            return_value=cache_dir / "bls_eggs.csv",
        )

        from egg_price_forecast.data import fetch_bls_eggs

        result = fetch_bls_eggs()

        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5
        assert result.dtype == float

    def test_values_are_correct(self, mocker, tmp_path):
        """Returned values should match the CSV data."""
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        cache_dir = tmp_path / "cache"
        mocker.patch(
            "egg_price_forecast.data._cache_path",
            return_value=cache_dir / "bls_eggs.csv",
        )

        from egg_price_forecast.data import fetch_bls_eggs

        result = fetch_bls_eggs()
        assert result.iloc[0] == pytest.approx(0.879)
        assert result.iloc[-1] == pytest.approx(4.950)

    def test_cache_is_used_on_second_call(self, mocker, tmp_path):
        """Second call within TTL should use cache, not make a new request."""
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mock_get = mocker.patch("requests.get", return_value=mock_resp)

        cache_file = tmp_path / "cache" / "bls_eggs.csv"
        mocker.patch(
            "egg_price_forecast.data._cache_path",
            return_value=cache_file,
        )

        from egg_price_forecast.data import fetch_bls_eggs

        # First call should make a request
        result1 = fetch_bls_eggs()
        assert mock_get.call_count == 1

        # Second call should use cache
        result2 = fetch_bls_eggs()
        assert mock_get.call_count == 1  # No additional request

        pd.testing.assert_series_equal(result1, result2)

    def test_cache_expires_after_ttl(self, mocker, tmp_path):
        """Cache should be refreshed after TTL expires."""
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mock_get = mocker.patch("requests.get", return_value=mock_resp)

        cache_file = tmp_path / "cache" / "bls_eggs.csv"
        mocker.patch(
            "egg_price_forecast.data._cache_path",
            return_value=cache_file,
        )

        from egg_price_forecast.data import fetch_bls_eggs

        # First call
        fetch_bls_eggs()
        assert mock_get.call_count == 1

        # Backdate the cache file modification time by 2 hours
        os.utime(cache_file, (time.time() - 7200, time.time() - 7200))

        # Next call should re-fetch
        fetch_bls_eggs()
        assert mock_get.call_count == 2


class TestFetchBlsSeries:
    """Tests for fetch_bls_series (generic FRED fetcher)."""

    def test_returns_series(self, mocker, tmp_path):
        """fetch_bls_series returns a pd.Series."""
        mock_resp = MagicMock()
        # The generic fetcher uses the series_id as the column name
        mock_resp.text = "observation_date,APU0000703112\n2024-01-01,5.123\n2024-02-01,5.456\n"
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        # Patch cache dir
        mocker.patch(
            "egg_price_forecast.data._CACHE_DIR",
            tmp_path / "cache",
        )

        from egg_price_forecast.data import fetch_bls_series

        result = fetch_bls_series("APU0000703112")
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(5.123)

    def test_uses_correct_url(self, mocker, tmp_path):
        """fetch_bls_series requests the correct FRED URL."""
        mock_resp = MagicMock()
        mock_resp.text = "observation_date,TEST123\n2024-01-01,1.0\n"
        mock_resp.raise_for_status = MagicMock()
        mock_get = mocker.patch("requests.get", return_value=mock_resp)

        mocker.patch(
            "egg_price_forecast.data._CACHE_DIR",
            tmp_path / "cache",
        )

        from egg_price_forecast.data import fetch_bls_series

        fetch_bls_series("TEST123")
        call_url = mock_get.call_args[0][0]
        assert "TEST123" in call_url
        assert "fredgraph.csv" in call_url


class TestFetchAllCommodities:
    """Tests for fetch_all_commodities."""

    def test_returns_dict_of_series(self, mocker, tmp_path):
        """fetch_all_commodities returns a dict of pd.Series."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def mock_fetch(series_id):
            dates = pd.date_range("2024-01-01", periods=3, freq="MS")
            return pd.Series([1.0, 2.0, 3.0], index=dates)

        mocker.patch(
            "egg_price_forecast.data.fetch_bls_series",
            side_effect=mock_fetch,
        )

        from egg_price_forecast.data import fetch_all_commodities

        result = fetch_all_commodities()
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, series in result.items():
            assert isinstance(series, pd.Series)

    def test_handles_fetch_failures(self, mocker):
        """fetch_all_commodities skips series that fail to fetch."""
        call_count = 0

        def mock_fetch(series_id):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception("Network error")
            dates = pd.date_range("2024-01-01", periods=3, freq="MS")
            return pd.Series([1.0, 2.0, 3.0], index=dates)

        mocker.patch(
            "egg_price_forecast.data.fetch_bls_series",
            side_effect=mock_fetch,
        )

        from egg_price_forecast.data import fetch_all_commodities

        result = fetch_all_commodities()
        # Some should succeed, some should fail
        assert len(result) > 0


class TestFetchCpiExpectations:
    """Tests for fetch_cpi_expectations."""

    def test_returns_series(self, mocker, tmp_path):
        """fetch_cpi_expectations returns a pd.Series."""
        mock_resp = MagicMock()
        mock_resp.text = "observation_date,EXPINF1YR\n2024-01-01,2.5\n2024-02-01,2.6\n"
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)
        mocker.patch(
            "egg_price_forecast.data._CACHE_DIR",
            tmp_path / "cache",
        )

        from egg_price_forecast.data import fetch_cpi_expectations

        result = fetch_cpi_expectations()
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestGetWholesalePrices:
    """Tests for get_wholesale_prices."""

    def test_returns_float_on_success(self, mocker):
        """Should return a float price in $/dozen."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_USDA_JSON
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        from egg_price_forecast.data import get_wholesale_prices

        result = get_wholesale_prices()
        assert isinstance(result, float)
        assert result > 0

    def test_uses_midpoint_of_mostly_prices(self, mocker):
        """Should average mostly_low and mostly_high prices."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_USDA_JSON
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        from egg_price_forecast.data import get_wholesale_prices

        result = get_wholesale_prices()
        # (8.10 + 8.40) / 2 = 8.25 dollars/dozen
        # The data is already in dollars from the JSON
        assert result == pytest.approx(8.25, abs=0.01)

    def test_returns_none_on_network_error(self, mocker):
        """Should return None if the request fails."""
        mocker.patch("requests.get", side_effect=Exception("Network error"))

        from egg_price_forecast.data import get_wholesale_prices

        result = get_wholesale_prices()
        assert result is None

    def test_returns_none_on_empty_response(self, mocker):
        """Should return None if the API returns empty data."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        from egg_price_forecast.data import get_wholesale_prices

        result = get_wholesale_prices()
        assert result is None

    def test_returns_none_on_malformed_json(self, mocker):
        """Should return None if JSON structure is unexpected."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"unexpected": "data"}]
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("requests.get", return_value=mock_resp)

        from egg_price_forecast.data import get_wholesale_prices

        result = get_wholesale_prices()
        assert result is None
