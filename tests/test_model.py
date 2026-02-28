"""Tests for egg_price_forecast.model (Bayesian version)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    """Create ~5 years of monthly egg prices with realistic patterns."""
    dates = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    np.random.seed(42)
    base = 1.50
    seasonal = 0.10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    trend = np.linspace(0, 0.30, len(dates))
    noise = np.random.normal(0, 0.05, len(dates))
    values = base + seasonal + trend + noise
    return pd.Series(values, index=dates, name="egg_price")


def _make_commodity_data(n_years=10, start="2015-01-01"):
    """Create synthetic multi-commodity data."""
    np.random.seed(42)
    commodities = {}
    for name in ["eggs", "milk", "bread", "beef"]:
        n = n_years * 12
        base = 2.0 + np.random.random() * 3
        trend = np.linspace(0, 0.5, n)
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.05, n)
        values = base + trend + seasonal + noise
        dates = pd.date_range(start=start, periods=n, freq="MS")
        commodities[name] = pd.Series(values, index=dates, dtype=float)
    return commodities


class TestEstimateProbability:
    """Tests for estimate_probability function."""

    def test_returns_correct_structure(self, synthetic_prices, mocker):
        """estimate_probability returns dict with required keys."""
        commodities = _make_commodity_data()
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert isinstance(result, dict)
        assert "probability" in result
        assert "components" in result
        # No more ci_low/ci_high
        assert "ci_low" not in result
        assert "ci_high" not in result

    def test_probability_between_005_and_095(self, synthetic_prices, mocker):
        """Probability is clamped to [0.05, 0.95]."""
        commodities = _make_commodity_data()
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert 0.05 <= result["probability"] <= 0.95

    def test_components_has_new_keys(self, synthetic_prices, mocker):
        """Components dict contains the new Bayesian component names."""
        commodities = _make_commodity_data()
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)
        components = result["components"]

        assert "intercept" in components
        assert "seasonal" in components
        assert "streak" in components
        assert "momentum" in components
        assert "egg_specific" in components

    def test_model_info_present(self, synthetic_prices, mocker):
        """Result includes model_info with training stats."""
        commodities = _make_commodity_data()
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert "model_info" in result
        assert "n_obs" in result["model_info"]
        assert "n_commodities" in result["model_info"]
        assert "coefficients" in result["model_info"]

    def test_wholesale_param_accepted(self, synthetic_prices, mocker):
        """wholesale parameter is accepted for backward compatibility."""
        commodities = _make_commodity_data()
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import estimate_probability

        # Should not raise
        result = estimate_probability(synthetic_prices, wholesale=5.0)
        assert "probability" in result


class TestBacktest:
    """Tests for backtest function."""

    def test_returns_correct_structure(self, mocker):
        """Backtest returns dict with predictions, accuracy, brier_score."""
        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        commodities = _make_commodity_data(n_years=16, start="2009-01-01")
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)

        assert isinstance(result, dict)
        assert "predictions" in result
        assert "accuracy" in result
        assert "brier_score" in result

    def test_predictions_list_structure(self, mocker):
        """Each prediction has year, predicted_prob, actual_up, correct."""
        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        commodities = _make_commodity_data(n_years=16, start="2009-01-01")
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)

        assert len(result["predictions"]) > 0
        for pred in result["predictions"]:
            assert "year" in pred
            assert "predicted_prob" in pred
            assert "actual_up" in pred
            assert "correct" in pred
            assert isinstance(pred["actual_up"], bool)
            assert isinstance(pred["correct"], bool)
            assert 0.0 <= pred["predicted_prob"] <= 1.0

    def test_accuracy_range(self, mocker):
        """Accuracy is between 0 and 1."""
        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        commodities = _make_commodity_data(n_years=16, start="2009-01-01")
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_brier_score_range(self, mocker):
        """Brier score is between 0 and 1."""
        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        commodities = _make_commodity_data(n_years=16, start="2009-01-01")
        mocker.patch(
            "egg_price_forecast.model.fetch_all_commodities",
            return_value=commodities,
        )
        mocker.patch(
            "egg_price_forecast.model.fetch_cpi_expectations",
            side_effect=Exception("no CPI"),
        )

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)
        assert 0.0 <= result["brier_score"] <= 1.0
