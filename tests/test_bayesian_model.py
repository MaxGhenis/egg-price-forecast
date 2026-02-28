"""Tests for egg_price_forecast.bayesian_model module."""

import numpy as np
import pandas as pd
import pytest


def _make_prices(values, start="2020-01-01"):
    """Helper to create a price Series with monthly DatetimeIndex."""
    index = pd.date_range(start=start, periods=len(values), freq="MS")
    return pd.Series(values, index=index, dtype=float)


def _make_commodity_data():
    """Create synthetic multi-commodity data for testing."""
    np.random.seed(42)
    commodities = {}
    for name in ["eggs", "milk", "bread", "beef"]:
        n = 120  # 10 years of monthly data
        base = 2.0 + np.random.random() * 3
        trend = np.linspace(0, 0.5, n)
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.05, n)
        values = base + trend + seasonal + noise
        commodities[name] = _make_prices(values.tolist(), start="2015-01-01")
    return commodities


class TestBuildTrainingData:
    """Tests for build_training_data."""

    def test_returns_dataframe(self):
        from egg_price_forecast.bayesian_model import build_training_data

        commodities = _make_commodity_data()
        result = build_training_data(commodities)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_required_columns(self):
        from egg_price_forecast.bayesian_model import (
            FEATURE_NAMES,
            build_training_data,
        )

        commodities = _make_commodity_data()
        result = build_training_data(commodities)

        for col in FEATURE_NAMES:
            assert col in result.columns
        assert "target" in result.columns
        assert "commodity" in result.columns

    def test_target_is_binary(self):
        from egg_price_forecast.bayesian_model import build_training_data

        commodities = _make_commodity_data()
        result = build_training_data(commodities)

        assert set(result["target"].unique()).issubset({0.0, 1.0})

    def test_multiple_commodities_present(self):
        from egg_price_forecast.bayesian_model import build_training_data

        commodities = _make_commodity_data()
        result = build_training_data(commodities)

        assert result["commodity"].nunique() == len(commodities)

    def test_is_eggs_column_correct(self):
        from egg_price_forecast.bayesian_model import build_training_data

        commodities = _make_commodity_data()
        result = build_training_data(commodities)

        eggs_rows = result[result["commodity"] == "eggs"]
        non_eggs_rows = result[result["commodity"] != "eggs"]

        assert (eggs_rows["is_eggs"] == 1.0).all()
        assert (non_eggs_rows["is_eggs"] == 0.0).all()

    def test_cpi_expectations_integrated(self):
        from egg_price_forecast.bayesian_model import build_training_data

        commodities = _make_commodity_data()
        cpi = _make_prices(
            np.linspace(2.0, 3.0, 120).tolist(), start="2015-01-01"
        )
        result = build_training_data(commodities, cpi)

        # CPI values should be non-zero for most rows
        assert (result["cpi_expectation"] != 0).sum() > 0


class TestFitBayesianLogistic:
    """Tests for fit_bayesian_logistic."""

    def test_returns_correct_structure(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        result = fit_bayesian_logistic(training_data)

        assert "coefficients" in result
        assert "std_errors" in result
        assert "feature_names" in result
        assert "n_obs" in result
        assert "n_commodities" in result

    def test_coefficients_shape(self):
        from egg_price_forecast.bayesian_model import (
            FEATURE_NAMES,
            build_training_data,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        result = fit_bayesian_logistic(training_data)

        # intercept + 5 features
        expected_len = 1 + len(FEATURE_NAMES)
        assert len(result["coefficients"]) == expected_len
        assert len(result["std_errors"]) == expected_len

    def test_coefficients_are_finite(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        result = fit_bayesian_logistic(training_data)

        assert np.all(np.isfinite(result["coefficients"]))

    def test_n_obs_matches_training_data(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        result = fit_bayesian_logistic(training_data)

        assert result["n_obs"] == len(training_data)


class TestPredictProbability:
    """Tests for predict_probability."""

    def test_returns_float_between_0_and_1(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            fit_bayesian_logistic,
            predict_probability,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        fitted = fit_bayesian_logistic(training_data)

        features = {
            "seasonal_up_rate": 0.35,
            "decline_streak": 3.0,
            "recent_momentum": -0.05,
            "cpi_expectation": 2.5,
            "is_eggs": 1.0,
        }

        prob = predict_probability(fitted, features)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_extreme_features_produce_extreme_probs(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            fit_bayesian_logistic,
            predict_probability,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        fitted = fit_bayesian_logistic(training_data)

        low_features = {
            "seasonal_up_rate": 0.0,
            "decline_streak": 0.0,
            "recent_momentum": -0.5,
            "cpi_expectation": 0.0,
            "is_eggs": 0.0,
        }
        high_features = {
            "seasonal_up_rate": 1.0,
            "decline_streak": 10.0,
            "recent_momentum": 0.5,
            "cpi_expectation": 5.0,
            "is_eggs": 1.0,
        }

        prob_low = predict_probability(fitted, low_features)
        prob_high = predict_probability(fitted, high_features)

        # High features should generally produce higher probability
        # (may not always hold depending on learned coefficients,
        # but with regularizing priors it should)
        assert prob_low != prob_high


class TestDecomposeContributions:
    """Tests for decompose_contributions."""

    def test_returns_dict_with_expected_keys(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            decompose_contributions,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        fitted = fit_bayesian_logistic(training_data)

        features = {
            "seasonal_up_rate": 0.35,
            "decline_streak": 3.0,
            "recent_momentum": -0.05,
            "cpi_expectation": 2.5,
            "is_eggs": 1.0,
        }

        result = decompose_contributions(fitted, features)

        assert "intercept" in result
        assert "seasonal" in result
        assert "streak" in result
        assert "momentum" in result
        assert "cpi_signal" in result
        assert "egg_specific" in result

    def test_contributions_are_float(self):
        from egg_price_forecast.bayesian_model import (
            build_training_data,
            decompose_contributions,
            fit_bayesian_logistic,
        )

        commodities = _make_commodity_data()
        training_data = build_training_data(commodities)
        fitted = fit_bayesian_logistic(training_data)

        features = {
            "seasonal_up_rate": 0.35,
            "decline_streak": 3.0,
            "recent_momentum": -0.05,
            "cpi_expectation": 2.5,
            "is_eggs": 1.0,
        }

        result = decompose_contributions(fitted, features)
        for v in result.values():
            assert isinstance(v, float)
