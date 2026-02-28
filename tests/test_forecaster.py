"""Tests for egg_price_forecast.forecaster (LightGBM quantile regression)."""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_series(
    n_months: int = 50,
    base: float = 3.0,
    seed: int = 42,
    start: str = "2019-01-01",
) -> pd.Series:
    """Generate a synthetic monthly price series (random walk)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    changes = rng.normal(0, 0.05, n_months)
    values = base + np.cumsum(changes)
    values = np.maximum(values, 0.5)  # keep positive
    return pd.Series(values, index=dates, dtype=float)


def _make_synthetic_commodities(
    n_commodities: int = 3,
    n_months: int = 50,
    start: str = "2019-01-01",
) -> dict[str, pd.Series]:
    """Generate synthetic multi-commodity data including eggs."""
    names = ["eggs"] + [f"commodity_{i}" for i in range(1, n_commodities)]
    commodities = {}
    for i, name in enumerate(names):
        commodities[name] = _make_synthetic_series(
            n_months=n_months,
            base=2.0 + i * 0.5,
            seed=42 + i,
            start=start,
        )
    return commodities


def _make_synthetic_cpi(
    n_months: int = 50,
    start: str = "2019-01-01",
) -> pd.Series:
    """Generate synthetic CPI expectation series."""
    rng = np.random.RandomState(99)
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    values = 2.5 + rng.normal(0, 0.3, n_months)
    return pd.Series(values, index=dates, dtype=float)


class TestBuildForecastFeatures:
    """Tests for build_forecast_features."""

    def test_returns_dataframe(self):
        """build_forecast_features returns a pandas DataFrame."""
        from egg_price_forecast.forecaster import build_forecast_features

        commodities = _make_synthetic_commodities()
        cpi = _make_synthetic_cpi()
        result = build_forecast_features(commodities, cpi)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_correct_columns_present(self):
        """All expected feature columns are present."""
        from egg_price_forecast.forecaster import build_forecast_features

        commodities = _make_synthetic_commodities()
        cpi = _make_synthetic_cpi()
        result = build_forecast_features(commodities, cpi)

        expected_cols = [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "lag_6",
            "lag_12",
            "pct_change_1m",
            "pct_change_3m",
            "pct_change_12m",
            "month_sin",
            "month_cos",
            "cpi_expectation",
            "cross_commodity_momentum",
            "is_eggs",
            "price_level",
            "target",
            "commodity",
            "date",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_target_is_next_month_price(self):
        """Target column equals the next month's price."""
        from egg_price_forecast.forecaster import build_forecast_features

        # Use a single commodity for clarity
        series = _make_synthetic_series(n_months=30, base=3.0, start="2019-01-01")
        commodities = {"test_item": series}
        cpi = _make_synthetic_cpi(n_months=30)
        result = build_forecast_features(commodities, cpi)

        # For each row, verify target == price at the next month
        for _, row in result.iterrows():
            date = row["date"]
            date_idx = series.index.get_loc(date)
            expected_target = series.iloc[date_idx + 1]
            assert abs(row["target"] - expected_target) < 1e-10, (
                f"Target mismatch at {date}: got {row['target']}, "
                f"expected {expected_target}"
            )


class TestTrainQuantileModels:
    """Tests for train_quantile_models."""

    def test_returns_dict_with_all_quantiles(self):
        """train_quantile_models returns a dict with all quantile keys."""
        from egg_price_forecast.forecaster import (
            build_forecast_features,
            train_quantile_models,
        )

        commodities = _make_synthetic_commodities(n_months=60)
        cpi = _make_synthetic_cpi(n_months=60)
        df = build_forecast_features(commodities, cpi)

        quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
        models = train_quantile_models(df, quantiles=quantiles)

        assert isinstance(models, dict)
        for q in quantiles:
            assert q in models, f"Missing quantile model: {q}"


class TestPredictQuantiles:
    """Tests for predict_quantiles."""

    def test_quantiles_ordered(self):
        """Predicted quantile values are monotonically non-decreasing."""
        from egg_price_forecast.forecaster import (
            build_forecast_features,
            predict_quantiles,
            train_quantile_models,
        )

        commodities = _make_synthetic_commodities(n_months=60)
        cpi = _make_synthetic_cpi(n_months=60)
        df = build_forecast_features(commodities, cpi)

        quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
        models = train_quantile_models(df, quantiles=quantiles)

        # Use the last row as prediction input
        last_row = df.iloc[[-1]]
        preds = predict_quantiles(models, last_row)

        sorted_qs = sorted(preds.keys())
        for i in range(1, len(sorted_qs)):
            assert preds[sorted_qs[i]] >= preds[sorted_qs[i - 1]], (
                f"Quantile crossing: q{sorted_qs[i]} ({preds[sorted_qs[i]]}) "
                f"< q{sorted_qs[i-1]} ({preds[sorted_qs[i-1]]})"
            )


class TestSampleFromQuantiles:
    """Tests for sample_from_quantiles."""

    def test_sample_shape(self):
        """sample_from_quantiles returns array of requested size."""
        from egg_price_forecast.forecaster import sample_from_quantiles

        quantile_preds = {
            0.05: 2.0,
            0.25: 2.5,
            0.50: 3.0,
            0.75: 3.5,
            0.95: 4.0,
        }
        samples = sample_from_quantiles(quantile_preds, n=500)

        assert isinstance(samples, np.ndarray)
        assert len(samples) == 500

    def test_samples_non_negative(self):
        """All samples are non-negative (prices cannot be negative)."""
        from egg_price_forecast.forecaster import sample_from_quantiles

        quantile_preds = {
            0.05: 0.5,
            0.25: 1.0,
            0.50: 1.5,
            0.75: 2.0,
            0.95: 2.5,
        }
        samples = sample_from_quantiles(quantile_preds, n=2000)

        assert np.all(samples >= 0)

    def test_distribution_roughly_matches_quantiles(self):
        """Empirical quantiles of samples roughly match input quantiles."""
        from egg_price_forecast.forecaster import sample_from_quantiles

        quantile_preds = {
            0.05: 2.0,
            0.25: 2.5,
            0.50: 3.0,
            0.75: 3.5,
            0.95: 4.0,
        }

        np.random.seed(42)
        samples = sample_from_quantiles(quantile_preds, n=10000)

        for q, expected_val in quantile_preds.items():
            empirical = float(np.quantile(samples, q))
            # Allow 20% tolerance
            tolerance = 0.20 * abs(expected_val) + 0.1
            assert abs(empirical - expected_val) < tolerance, (
                f"Quantile {q}: empirical={empirical:.3f}, "
                f"expected={expected_val:.3f}, tolerance={tolerance:.3f}"
            )


class TestForecastResult:
    """Tests for the ForecastResult dataclass."""

    def test_dataclass_fields(self):
        """ForecastResult can be constructed and fields accessed."""
        from egg_price_forecast.forecaster import ForecastResult

        samples = np.array([2.5, 3.0, 3.5, 4.0])
        result = ForecastResult(
            point_forecast=3.0,
            intervals={90: (2.0, 4.0), 50: (2.5, 3.5)},
            quantiles={0.05: 2.0, 0.50: 3.0, 0.95: 4.0},
            samples=samples,
            p_up=0.65,
            current_price=2.8,
            feature_importances={"lag_1": 100, "lag_2": 80},
        )

        assert result.point_forecast == 3.0
        assert result.p_up == 0.65
        assert result.current_price == 2.8
        assert len(result.samples) == 4
        assert 90 in result.intervals
        assert 50 in result.intervals
        assert 0.50 in result.quantiles
        assert "lag_1" in result.feature_importances
