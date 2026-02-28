"""Tests for egg_price_forecast.model (quantile forecast version)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


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


def _make_mock_forecast_result():
    """Create a mock ForecastResult for testing."""
    mock = MagicMock()
    mock.p_up = 0.65
    mock.point_forecast = 5.23
    mock.intervals = {90: (4.20, 6.30), 50: (4.80, 5.65)}
    mock.quantiles = {
        0.05: 4.20,
        0.10: 4.45,
        0.25: 4.80,
        0.50: 5.23,
        0.75: 5.65,
        0.90: 6.10,
        0.95: 6.30,
    }
    mock.samples = np.random.normal(5.23, 0.55, 1000)
    mock.current_price = 4.95
    mock.feature_importances = {
        "lag_1": 0.25,
        "lag_12": 0.15,
        "pct_change_1m": 0.12,
        "month_sin": 0.10,
    }
    return mock


class TestEstimateProbability:
    """Tests for estimate_probability function."""

    def test_returns_correct_structure(self, synthetic_prices, monkeypatch):
        """estimate_probability returns dict with required keys."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert isinstance(result, dict)
        assert "probability" in result
        assert "point_forecast" in result
        assert "intervals" in result
        assert "components" in result
        assert "model_info" in result
        assert "quantiles" in result
        assert "samples_summary" in result

    def test_probability_bounds(self, synthetic_prices, monkeypatch):
        """Probability is between 0 and 1, point_forecast is positive."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert 0.0 <= result["probability"] <= 1.0
        assert result["point_forecast"] > 0

    def test_intervals_ordered(self, synthetic_prices, monkeypatch):
        """Intervals have low < high for both 50 and 90."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        for key, interval in result["intervals"].items():
            assert interval["low"] < interval["high"], (
                f"Interval {key}: low ({interval['low']}) >= high ({interval['high']})"
            )

    def test_components_are_feature_importances(self, synthetic_prices, monkeypatch):
        """Components dict contains feature importance entries."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)
        components = result["components"]

        # Should have at most 8 entries (top features)
        assert len(components) <= 8
        # All values should be numeric
        for v in components.values():
            assert isinstance(v, float)

    def test_model_info_present(self, synthetic_prices, monkeypatch):
        """Result includes model_info with type and feature_importances."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert "model_info" in result
        assert result["model_info"]["type"] == "lightgbm_quantile"
        assert "n_quantile_models" in result["model_info"]
        assert "n_samples" in result["model_info"]
        assert "feature_importances" in result["model_info"]

    def test_quantiles_present(self, synthetic_prices, monkeypatch):
        """Result includes quantiles dict with string keys."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        assert "quantiles" in result
        assert len(result["quantiles"]) == 7
        for k, v in result["quantiles"].items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_samples_summary_present(self, synthetic_prices, monkeypatch):
        """Result includes samples_summary with mean, std, n."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)

        summary = result["samples_summary"]
        assert "mean" in summary
        assert "std" in summary
        assert "n" in summary
        assert summary["n"] == 1000

    def test_returns_float_values(self, synthetic_prices, monkeypatch):
        """All numeric values in result are floats."""
        mock_result = _make_mock_forecast_result()

        import egg_price_forecast.forecaster as forecaster_mod
        monkeypatch.setattr(forecaster_mod, "run_forecast", lambda **kwargs: mock_result)

        from egg_price_forecast.model import estimate_probability

        result = estimate_probability(synthetic_prices)
        assert isinstance(result["probability"], float)
        assert isinstance(result["point_forecast"], float)


class TestBacktest:
    """Tests for backtest function."""

    def _make_backtest_mocks(self, monkeypatch):
        """Set up mocks for backtest's forecaster imports."""
        import egg_price_forecast.forecaster as forecaster_mod
        import egg_price_forecast.data as data_mod

        # Mock data fetchers
        np.random.seed(42)
        commodities = {}
        for name in ["eggs", "milk", "bread", "beef"]:
            n = 16 * 12
            dates = pd.date_range("2009-01-01", periods=n, freq="MS")
            values = 2.0 + np.random.normal(0, 0.1, n)
            commodities[name] = pd.Series(values, index=dates, dtype=float)

        cpi = pd.Series(
            np.random.normal(2.5, 0.3, 16 * 12),
            index=pd.date_range("2009-01-01", periods=16 * 12, freq="MS"),
        )

        monkeypatch.setattr(data_mod, "fetch_all_commodities", lambda: commodities)
        monkeypatch.setattr(data_mod, "fetch_cpi_expectations", lambda: cpi)

        # Build a synthetic feature DataFrame
        n_rows = 150
        dates = pd.date_range("2009-01-01", periods=n_rows, freq="MS")
        feature_df = pd.DataFrame(
            {
                "date": dates,
                "commodity": ["eggs"] * n_rows,
                "target": np.random.normal(2.0, 0.2, n_rows),
                "price_level": np.random.normal(2.0, 0.1, n_rows),
                "lag_1": np.random.normal(0, 0.05, n_rows),
                "lag_12": np.random.normal(0, 0.05, n_rows),
                "pct_change_1m": np.random.normal(0, 0.02, n_rows),
            }
        )

        monkeypatch.setattr(
            forecaster_mod, "build_forecast_features", lambda *a, **kw: feature_df
        )

        # Mock train_quantile_models to return a dummy
        monkeypatch.setattr(
            forecaster_mod, "train_quantile_models", lambda *a, **kw: {"models": True}
        )

        # Mock predict_quantiles to return a dict of quantile -> value
        def mock_predict(models, features_row):
            return {
                0.05: 1.70,
                0.10: 1.80,
                0.25: 1.90,
                0.50: 2.00,
                0.75: 2.10,
                0.90: 2.20,
                0.95: 2.30,
            }

        monkeypatch.setattr(forecaster_mod, "predict_quantiles", mock_predict)

        # Mock sample_from_quantiles to return array of samples
        monkeypatch.setattr(
            forecaster_mod,
            "sample_from_quantiles",
            lambda q_preds: np.random.normal(2.0, 0.15, 500),
        )

    def test_returns_correct_structure(self, monkeypatch):
        """Backtest returns dict with direction_accuracy, mape, coverage, brier_score."""
        self._make_backtest_mocks(monkeypatch)

        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)

        assert isinstance(result, dict)
        assert "predictions" in result
        assert "direction_accuracy" in result
        assert "mape" in result
        assert "coverage_90" in result
        assert "coverage_50" in result
        assert "brier_score" in result
        # Backward compat
        assert "accuracy" in result

    def test_predictions_list_structure(self, monkeypatch):
        """Each prediction has date, year, month, predicted_prob, actual_up, correct."""
        self._make_backtest_mocks(monkeypatch)

        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)

        if len(result["predictions"]) > 0:
            for pred in result["predictions"]:
                assert "date" in pred
                assert "year" in pred
                assert "predicted_prob" in pred
                assert "actual_up" in pred
                assert "correct" in pred
                assert isinstance(pred["actual_up"], bool)
                assert isinstance(pred["correct"], bool)
                assert 0.0 <= pred["predicted_prob"] <= 1.0
                assert "error" in pred
                assert "in_90_interval" in pred
                assert "in_50_interval" in pred

    def test_accuracy_range(self, monkeypatch):
        """Direction accuracy is between 0 and 1."""
        self._make_backtest_mocks(monkeypatch)

        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)
        assert 0.0 <= result["direction_accuracy"] <= 1.0
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_brier_score_range(self, monkeypatch):
        """Brier score is between 0 and 1."""
        self._make_backtest_mocks(monkeypatch)

        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_coverage_metrics_range(self, monkeypatch):
        """Coverage metrics are between 0 and 1."""
        self._make_backtest_mocks(monkeypatch)

        dates = pd.date_range("2009-01-01", "2024-12-01", freq="MS")
        np.random.seed(42)
        values = 1.50 + np.random.normal(0, 0.10, len(dates))
        prices = pd.Series(values, index=dates, name="egg_price")

        from egg_price_forecast.model import backtest

        result = backtest(prices, start_year=2020)
        assert 0.0 <= result["coverage_90"] <= 1.0
        assert 0.0 <= result["coverage_50"] <= 1.0


