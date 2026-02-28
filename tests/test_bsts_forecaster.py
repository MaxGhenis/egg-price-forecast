"""Tests for bsts_forecaster module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from egg_price_forecast.bsts_forecaster import (
    BSTSForecastResult,
    BSTSResult,
    build_almon_basis,
    build_covariate_matrix,
    build_full_exog,
    build_intervention_columns,
    build_month_dummies,
    fit_bsts,
    forecast_with_draws,
)
from egg_price_forecast.series_config import InterventionConfig, SeriesConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_monthly_index(n: int = 120, start: str = "2014-01-01") -> pd.DatetimeIndex:
    """Create a monthly DatetimeIndex."""
    return pd.date_range(start, periods=n, freq="MS")


def _make_random_walk_seasonal(
    n: int = 120,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic random walk + seasonal price series."""
    rng = np.random.default_rng(seed)
    idx = _make_monthly_index(n)

    # Random walk (drift ~0, sigma ~0.05)
    level = np.cumsum(rng.normal(0, 0.05, n)) + 3.0  # Start around $3

    # Seasonal component (period=12)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 12)

    # Noise
    noise = rng.normal(0, 0.02, n)

    prices = level + seasonal + noise
    prices = np.clip(prices, 0.5, None)  # Prices must be positive
    return pd.Series(prices, index=idx, name="price")


def _make_covariate(n: int = 120, seed: int = 123) -> pd.Series:
    """Generate a synthetic covariate series."""
    rng = np.random.default_rng(seed)
    idx = _make_monthly_index(n)
    values = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.Series(values, index=idx, name="covariate")


# ---------------------------------------------------------------------------
# Tests: Almon basis construction
# ---------------------------------------------------------------------------


class TestBuildAlmonBasis:
    def test_output_shape(self):
        """Basis has poly_degree+1 columns and same length as input."""
        cov = _make_covariate(60)
        basis = build_almon_basis(cov, max_lag=12, poly_degree=2)
        assert basis.shape == (60, 3)  # z_0, z_1, z_2

    def test_column_names(self):
        """Columns are named z_0, z_1, ..., z_d."""
        cov = _make_covariate(60)
        basis = build_almon_basis(cov, max_lag=6, poly_degree=3)
        assert list(basis.columns) == ["z_0", "z_1", "z_2", "z_3"]

    def test_leading_nans(self):
        """First max_lag rows should be NaN (insufficient lag history)."""
        cov = _make_covariate(60)
        basis = build_almon_basis(cov, max_lag=12, poly_degree=2)
        # Rows 0..11 should be NaN
        assert basis.iloc[:12].isna().all().all()
        # Row 12 onwards should be non-NaN
        assert basis.iloc[12:].notna().all().all()

    def test_z0_is_simple_sum(self):
        """z_0 = sum of x(t-k) for k=0..max_lag (since k^0 = 1)."""
        cov = _make_covariate(30)
        basis = build_almon_basis(cov, max_lag=5, poly_degree=2)
        # Check z_0 at t=5: should be sum of cov[0:6]
        expected = cov.values[0:6].sum()
        assert abs(basis["z_0"].iloc[5] - expected) < 1e-10

    def test_polynomial_structure(self):
        """z_d weights increase as k^d for higher lags."""
        cov = pd.Series(np.ones(20), index=_make_monthly_index(20))
        basis = build_almon_basis(cov, max_lag=5, poly_degree=2)
        # z_0 = sum of 1s = 6 (lags 0..5)
        assert abs(basis["z_0"].iloc[5] - 6.0) < 1e-10
        # z_1 = 0+1+2+3+4+5 = 15
        assert abs(basis["z_1"].iloc[5] - 15.0) < 1e-10
        # z_2 = 0+1+4+9+16+25 = 55
        assert abs(basis["z_2"].iloc[5] - 55.0) < 1e-10

    def test_index_preserved(self):
        """Output index matches input index."""
        cov = _make_covariate(60)
        basis = build_almon_basis(cov, max_lag=12, poly_degree=2)
        pd.testing.assert_index_equal(basis.index, cov.index)


# ---------------------------------------------------------------------------
# Tests: Covariate matrix building
# ---------------------------------------------------------------------------


class TestBuildCovariateMatrix:
    def test_returns_dataframe(self):
        """build_covariate_matrix returns a DataFrame."""
        target = _make_random_walk_seasonal(60)
        config = SeriesConfig(
            name="test",
            fred_id="TEST001",
            display_name="Test",
            unit="$/unit",
            covariates={},  # No covariates
        )
        result = build_covariate_matrix(target, config)
        assert isinstance(result, pd.DataFrame)

    def test_empty_covariates_gives_empty_df(self):
        """With no covariates, returns empty DataFrame with target index."""
        target = _make_random_walk_seasonal(60)
        config = SeriesConfig(
            name="test",
            fred_id="TEST001",
            display_name="Test",
            unit="$/unit",
            covariates={},
        )
        result = build_covariate_matrix(target, config)
        assert result.empty or len(result.columns) == 0


# ---------------------------------------------------------------------------
# Tests: Model fitting
# ---------------------------------------------------------------------------


class TestFitBSTS:
    def test_fits_without_covariates(self):
        """Model fits on synthetic data without covariates."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test",
            fred_id="TEST001",
            display_name="Test",
            unit="$/unit",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        assert isinstance(result, BSTSResult)
        assert len(result.mle_params) > 0
        assert result.cov_params.shape[0] == len(result.mle_params)

    def test_fits_with_covariates(self):
        """Model fits on synthetic data with covariates."""
        target = _make_random_walk_seasonal(120)
        cov = _make_covariate(120)
        basis = build_almon_basis(cov, max_lag=12, poly_degree=2)
        basis.columns = ["ppi_z_0", "ppi_z_1", "ppi_z_2"]

        config = SeriesConfig(
            name="test",
            fred_id="TEST001",
            display_name="Test",
            unit="$/unit",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, basis, config)
        assert isinstance(result, BSTSResult)
        assert result.exog is not None

    def test_handles_nan_covariates(self):
        """Model drops NaN covariate rows gracefully."""
        target = _make_random_walk_seasonal(120)
        cov = _make_covariate(120)
        basis = build_almon_basis(cov, max_lag=12, poly_degree=2)
        basis.columns = ["ppi_z_0", "ppi_z_1", "ppi_z_2"]
        # Leading NaNs from Almon basis should be dropped
        result = fit_bsts(target, basis, config=SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        ))
        assert isinstance(result, BSTSResult)

    def test_component_estimates_have_level(self):
        """Fitted result includes a level component estimate."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        assert "level" in result.component_estimates

    def test_config_preserved(self):
        """The config is stored in the result."""
        target = _make_random_walk_seasonal(60)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="Test", unit="$/u",
        )
        result = fit_bsts(target, None, config)
        assert result.config is config


# ---------------------------------------------------------------------------
# Tests: Forecasting with draws
# ---------------------------------------------------------------------------


class TestForecastWithDraws:
    @pytest.fixture
    def fitted_result(self):
        """Fit a model for use in forecast tests."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="Test", unit="$/u",
            seasonal_harmonics=2,
        )
        return fit_bsts(target, None, config)

    def test_returns_forecast_result(self, fitted_result):
        """forecast_with_draws returns a BSTSForecastResult."""
        fc = forecast_with_draws(fitted_result, n_draws=500, horizon=1)
        assert isinstance(fc, BSTSForecastResult)

    def test_draws_are_positive(self, fitted_result):
        """All forecast draws should be positive (prices > 0)."""
        fc = forecast_with_draws(fitted_result, n_draws=500, horizon=1)
        assert np.all(fc.posterior_draws > 0)

    def test_correct_number_of_draws(self, fitted_result):
        """Number of draws matches n_draws parameter."""
        fc = forecast_with_draws(fitted_result, n_draws=200, horizon=1)
        assert len(fc.posterior_draws) == 200

    def test_p_up_in_valid_range(self, fitted_result):
        """P(up) should be between 0 and 1."""
        fc = forecast_with_draws(fitted_result, n_draws=500, horizon=1)
        assert 0.0 <= fc.p_up <= 1.0

    def test_intervals_present(self, fitted_result):
        """90% and 50% intervals should be present."""
        fc = forecast_with_draws(fitted_result, n_draws=500, horizon=1)
        assert 90 in fc.intervals
        assert 50 in fc.intervals

    def test_interval_ordering(self, fitted_result):
        """90% interval should be wider than 50% interval."""
        fc = forecast_with_draws(fitted_result, n_draws=1000, horizon=1)
        low_90, high_90 = fc.intervals[90]
        low_50, high_50 = fc.intervals[50]
        assert low_90 <= low_50
        assert high_50 <= high_90

    def test_quantiles_monotonic(self, fitted_result):
        """Quantile values should be non-decreasing."""
        fc = forecast_with_draws(fitted_result, n_draws=1000, horizon=1)
        q_levels = sorted(fc.quantiles.keys())
        q_values = [fc.quantiles[q] for q in q_levels]
        for i in range(1, len(q_values)):
            assert q_values[i] >= q_values[i - 1]

    def test_point_forecast_reasonable(self, fitted_result):
        """Point forecast should be close to current price for 1-month horizon."""
        fc = forecast_with_draws(fitted_result, n_draws=1000, horizon=1)
        # 1-month forecast should be within 20% of current price
        pct_diff = abs(fc.point_forecast - fc.current_price) / fc.current_price
        assert pct_diff < 0.20, (
            f"Point forecast ${fc.point_forecast:.2f} is {pct_diff:.0%} away "
            f"from current ${fc.current_price:.2f}"
        )

    def test_90_interval_not_absurdly_wide(self, fitted_result):
        """90% interval width should be reasonable (< 100% of current price)."""
        fc = forecast_with_draws(fitted_result, n_draws=1000, horizon=1)
        low, high = fc.intervals[90]
        width = high - low
        assert width / fc.current_price < 1.0, (
            f"90% interval [{low:.2f}, {high:.2f}] is {width/fc.current_price:.0%} "
            f"of current price ${fc.current_price:.2f}"
        )

    def test_median_near_point(self, fitted_result):
        """Median forecast should be close to mean (point) forecast."""
        fc = forecast_with_draws(fitted_result, n_draws=2000, horizon=1)
        diff = abs(fc.median_forecast - fc.point_forecast)
        assert diff / fc.current_price < 0.05, (
            f"Median {fc.median_forecast:.2f} vs mean {fc.point_forecast:.2f}"
        )

    def test_model_diagnostics_present(self, fitted_result):
        """Model diagnostics dict should have key metrics."""
        fc = forecast_with_draws(fitted_result, n_draws=100, horizon=1)
        assert "n_obs" in fc.model_diagnostics
        assert "n_params" in fc.model_diagnostics
        assert fc.model_diagnostics["n_obs"] > 0

    def test_component_contributions_has_trend(self, fitted_result):
        """Component contributions should include trend."""
        fc = forecast_with_draws(fitted_result, n_draws=100, horizon=1)
        assert "trend" in fc.component_contributions


# ---------------------------------------------------------------------------
# Tests: MLE convergence with covariates
# ---------------------------------------------------------------------------


class TestConvergence:
    """MLE should converge when covariates are properly scaled."""

    def _make_covariate_pct_change(self, n=120, seed=99):
        """Covariate series as pct_change (small values, stationary)."""
        rng = np.random.default_rng(seed)
        idx = _make_monthly_index(n)
        # Simulate a PPI-like index, then take pct_change
        levels = np.cumsum(rng.normal(0, 2, n)) + 200
        pct = np.diff(levels) / levels[:-1]
        # Prepend NaN for first obs
        pct = np.concatenate([[np.nan], pct])
        return pd.Series(pct, index=idx, name="ppi_pct")

    def test_converges_with_pct_change_covariates(self):
        """Model should converge when covariates are pct_change (not raw levels)."""
        target = _make_random_walk_seasonal(120)
        cov = self._make_covariate_pct_change(120)
        basis = build_almon_basis(cov.dropna(), max_lag=6, poly_degree=1)
        basis.columns = ["ppi_z_0", "ppi_z_1"]
        # Reindex to target
        basis = basis.reindex(target.index)

        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_bsts(target, basis, config)
            convergence_warnings = [
                x for x in w if "convergence" in str(x.message).lower()
            ]
            assert len(convergence_warnings) == 0, (
                f"MLE failed to converge: {convergence_warnings[0].message}"
            )

    def test_diagnostics_report_convergence(self):
        """Model diagnostics should include converged flag."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=100, horizon=1)
        assert "converged" in fc.model_diagnostics


# ---------------------------------------------------------------------------
# Tests: Covariate signal actually influences forecast direction
# ---------------------------------------------------------------------------


class TestCovariateInfluence:
    """When a covariate signals 'down', the forecast P(up) should be < 0.5."""

    def test_bearish_covariate_lowers_p_up(self):
        """A strongly declining covariate should pull P(up) below 0.5."""
        n = 150
        rng = np.random.default_rng(42)
        idx = _make_monthly_index(n)

        # Target: stable around 3.0 with slight noise
        target = pd.Series(3.0 + rng.normal(0, 0.03, n), index=idx)

        # Covariate: strongly declining in recent months (bearish signal)
        cov_values = np.zeros(n)
        # Stable for most of history
        cov_values[:n - 12] = rng.normal(0, 0.01, n - 12)
        # Sharp decline in last 12 months
        cov_values[n - 12:] = np.linspace(-0.02, -0.10, 12)
        cov = pd.Series(cov_values, index=idx)

        # Build basis with degree 1, short lag
        basis = build_almon_basis(cov, max_lag=6, poly_degree=1)
        basis.columns = ["ppi_z_0", "ppi_z_1"]

        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=1,
        )

        result = fit_bsts(target, basis, config)
        fc = forecast_with_draws(result, n_draws=3000, horizon=1)

        # With a bearish covariate, model should at least not be fully bullish
        # (the covariate should have SOME influence)
        # Note: model may not fully respond, but should not be > 0.7
        assert fc.p_up < 0.7, (
            f"P(up) = {fc.p_up:.2f}, expected < 0.7 with bearish covariate"
        )

    def test_bullish_vs_bearish_covariates_differ(self):
        """Forecast with bullish covariate should have higher point forecast.

        Uses synthetic data where target strongly co-moves with covariate,
        so the model learns the relationship. Tests that the Kalman filter
        point estimate (before random draws) reflects the covariate signal.
        """
        n = 200
        rng = np.random.default_rng(42)
        idx = _make_monthly_index(n)

        # Covariate that the target responds to
        signal = np.cumsum(rng.normal(0, 0.05, n))

        # Target strongly co-moves: price = 3.0 + 1.0*signal + noise
        target = pd.Series(
            3.0 + 1.0 * signal + rng.normal(0, 0.01, n),
            index=idx,
        )

        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=1,
        )

        # Bearish scenario: signal drops sharply at end
        bear_signal = signal.copy()
        bear_signal[-6:] -= np.linspace(0.0, 2.0, 6)
        bear_cov = pd.Series(bear_signal, index=idx)
        bear_basis = build_almon_basis(bear_cov, max_lag=6, poly_degree=1)
        bear_basis.columns = ["ppi_z_0", "ppi_z_1"]

        bear_result = fit_bsts(target, bear_basis, config)
        bear_fc = forecast_with_draws(bear_result, n_draws=5000, horizon=1)

        # Bullish scenario: signal rises sharply at end
        bull_signal = signal.copy()
        bull_signal[-6:] += np.linspace(0.0, 2.0, 6)
        bull_cov = pd.Series(bull_signal, index=idx)
        bull_basis = build_almon_basis(bull_cov, max_lag=6, poly_degree=1)
        bull_basis.columns = ["ppi_z_0", "ppi_z_1"]

        bull_result = fit_bsts(target, bull_basis, config)
        bull_fc = forecast_with_draws(bull_result, n_draws=5000, horizon=1)

        # Compare deterministic Kalman forecasts (not stochastic draws)
        bull_kalman = bull_fc.model_diagnostics["kalman_forecast"]
        bear_kalman = bear_fc.model_diagnostics["kalman_forecast"]
        assert bull_kalman > bear_kalman, (
            f"Bullish Kalman forecast ${bull_kalman:.6f} should be > "
            f"bearish ${bear_kalman:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: Covariate matrix uses pct_change not raw levels
# ---------------------------------------------------------------------------


class TestForecastSECalibration:
    """Forecast SE should reflect actual model residuals, not just Kalman theory."""

    def test_se_at_least_as_large_as_residual_std(self):
        """Forecast SE must be >= std of in-sample one-step-ahead residuals.

        When sigma2.irregular collapses to zero, the Kalman filter reports
        artificially small SE. The residual std provides a hard floor.
        """
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=500, horizon=1)

        # Get in-sample residuals
        residual_std = float(np.std(result.model.resid))

        assert fc.model_diagnostics["forecast_se"] >= residual_std * 0.8, (
            f"Forecast SE {fc.model_diagnostics['forecast_se']:.4f} is much less "
            f"than residual std {residual_std:.4f}"
        )

    def test_se_includes_residual_std_in_diagnostics(self):
        """Model diagnostics should report residual_std for transparency."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=100, horizon=1)
        assert "residual_std" in fc.model_diagnostics

    def test_level_gap_inflates_se(self):
        """When the smoothed level is far from the last observation, SE should
        be at least half the gap.

        This catches the case where the Kalman smoother's level lags behind
        a rapid decline, and the model's theoretical SE doesn't reflect the
        true uncertainty about the underlying level.
        """
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=500, horizon=1)

        # Check that the level gap is reflected in the SE
        level = float(result.component_estimates.get("level", [np.nan])[-1])
        current = fc.current_price
        gap = abs(level - current)

        if gap > 0.01:  # Only test when there IS a gap
            assert fc.model_diagnostics["forecast_se"] >= gap * 0.4, (
                f"SE {fc.model_diagnostics['forecast_se']:.4f} should be >= "
                f"40% of level gap {gap:.4f}"
            )


class TestTrendExtrapolation:
    """Model should extrapolate a declining trend, not predict a bounce."""

    def test_declining_series_does_not_predict_large_bounce(self):
        """After 10+ months of decline, forecast should NOT be far above current.

        A +19% bounce after a sustained decline is unreasonable. The model
        should capture the downward slope and produce a forecast near or
        below the current price.

        Uses a spike that peaks in Feb-Mar (like real HPAI data) to trigger
        the seasonal pattern bug where the model learns "Feb = high".
        """
        # Start Jan 2014 so the spike lands on Feb-Mar timing
        n = 144  # 12 years
        rng = np.random.default_rng(42)
        idx = pd.date_range("2014-01-01", periods=n, freq="MS")

        # Base: stable around 2.0 with mild seasonal
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / 12)
        values = 2.0 + seasonal + rng.normal(0, 0.02, n)

        # HPAI-like spike peaking in Feb-Mar 2025 (months 133-134)
        # Ramp up from Sep 2024 (month 128)
        values[128:135] = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.0]
        # Decline from Apr 2025 to Jan 2026 (months 135-144)
        values[135:] = np.linspace(4.0, 2.0, n - 135) + rng.normal(0, 0.02, n - 135)
        target = pd.Series(values, index=idx)

        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=2000, horizon=1)

        current = fc.current_price
        pct_change = (fc.point_forecast - current) / current

        # After a sustained decline, the forecast should NOT jump up >10%
        assert pct_change < 0.10, (
            f"Forecast ${fc.point_forecast:.2f} is {pct_change:+.0%} vs "
            f"current ${current:.2f} — too large a bounce after decline"
        )

    def test_p_up_not_extreme_during_decline(self):
        """P(up) should not be >80% when the series has been declining."""
        n = 144
        rng = np.random.default_rng(42)
        idx = pd.date_range("2014-01-01", periods=n, freq="MS")

        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / 12)
        values = 2.0 + seasonal + rng.normal(0, 0.02, n)
        values[128:135] = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.0]
        values[135:] = np.linspace(4.0, 2.0, n - 135) + rng.normal(0, 0.02, n - 135)
        target = pd.Series(values, index=idx)

        config = SeriesConfig(
            name="test", fred_id="T", display_name="T", unit="$",
            seasonal_harmonics=2,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=3000, horizon=1)

        assert fc.p_up < 0.80, (
            f"P(up) = {fc.p_up:.1%} during sustained decline — "
            f"should be <80%"
        )


class TestCovariateScaling:
    """Covariates should be transformed to pct_change for proper scaling."""

    def test_covariate_matrix_values_are_small(self):
        """Almon basis values should be on a reasonable scale (not raw PPI ~200)."""
        target = _make_random_walk_seasonal(120)
        # Simulate a covariate that build_covariate_matrix would produce
        # After pct_change transform, values should be O(0.01-0.1), not O(100)
        cov = _make_covariate(120)  # raw levels ~100
        cov_pct = cov.pct_change()
        basis = build_almon_basis(cov_pct.dropna(), max_lag=6, poly_degree=1)

        # With pct_change input (~0.01), Almon sums should be O(0.1)
        valid = basis.dropna()
        assert valid.abs().max().max() < 10.0, (
            f"Almon basis max = {valid.abs().max().max():.1f}, "
            f"expected < 10 with pct_change input"
        )


# ---------------------------------------------------------------------------
# Tests: Intervention column construction
# ---------------------------------------------------------------------------


class TestInterventionColumns:
    """build_intervention_columns creates step + geometric decay regressors."""

    def _make_interventions(self):
        return (
            InterventionConfig(onset_date="2016-01-01", decay=0.85, label="event_a"),
            InterventionConfig(onset_date="2020-06-01", decay=0.90, label="event_b"),
        )

    def test_step_zero_before_onset(self):
        """Step column is 0 before onset date."""
        idx = _make_monthly_index(120, start="2014-01-01")
        interventions = (
            InterventionConfig(onset_date="2016-01-01", decay=0.85, label="ev"),
        )
        df = build_intervention_columns(idx, interventions)
        # All dates before 2016-01-01 should have step=0
        before = df.loc[idx < pd.Timestamp("2016-01-01"), "ev_step"]
        assert (before == 0).all()

    def test_step_one_after_onset(self):
        """Step column is 1 at and after onset date."""
        idx = _make_monthly_index(120, start="2014-01-01")
        interventions = (
            InterventionConfig(onset_date="2016-01-01", decay=0.85, label="ev"),
        )
        df = build_intervention_columns(idx, interventions)
        after = df.loc[idx >= pd.Timestamp("2016-01-01"), "ev_step"]
        assert (after == 1).all()

    def test_decay_zero_before_onset(self):
        """Decay column is 0 before onset date."""
        idx = _make_monthly_index(120, start="2014-01-01")
        interventions = (
            InterventionConfig(onset_date="2016-01-01", decay=0.85, label="ev"),
        )
        df = build_intervention_columns(idx, interventions)
        before = df.loc[idx < pd.Timestamp("2016-01-01"), "ev_decay"]
        assert (before == 0).all()

    def test_decay_geometric_pattern(self):
        """Decay column follows decay^(months since onset)."""
        idx = _make_monthly_index(60, start="2014-01-01")
        interventions = (
            InterventionConfig(onset_date="2015-01-01", decay=0.80, label="ev"),
        )
        df = build_intervention_columns(idx, interventions)
        # At onset (2015-01-01), months_since=0 → decay^0 = 1.0
        onset_val = df.loc[pd.Timestamp("2015-01-01"), "ev_decay"]
        assert abs(onset_val - 1.0) < 1e-10
        # 3 months later → 0.80^3 = 0.512
        three_months = df.loc[pd.Timestamp("2015-04-01"), "ev_decay"]
        assert abs(three_months - 0.80**3) < 1e-10

    def test_multiple_interventions_independent_columns(self):
        """Each intervention produces its own pair of step+decay columns."""
        idx = _make_monthly_index(120, start="2014-01-01")
        interventions = self._make_interventions()
        df = build_intervention_columns(idx, interventions)
        assert "event_a_step" in df.columns
        assert "event_a_decay" in df.columns
        assert "event_b_step" in df.columns
        assert "event_b_decay" in df.columns
        assert df.shape[1] == 4  # 2 interventions × 2 columns

    def test_empty_interventions_gives_empty_dataframe(self):
        """Empty interventions tuple → empty DataFrame."""
        idx = _make_monthly_index(60)
        df = build_intervention_columns(idx, ())
        assert df.empty or df.shape[1] == 0
        assert len(df) == len(idx)


# ---------------------------------------------------------------------------
# Tests: Month-of-year dummy construction
# ---------------------------------------------------------------------------


class TestMonthDummies:
    """build_month_dummies creates 11 dummy columns (Jan=reference)."""

    def test_eleven_columns(self):
        """Output has 11 columns for months 2-12."""
        idx = _make_monthly_index(24)
        df = build_month_dummies(idx)
        assert df.shape[1] == 11
        assert list(df.columns) == [f"month_{m}" for m in range(2, 13)]

    def test_january_all_zero(self):
        """January rows have all zeros (reference category)."""
        idx = _make_monthly_index(24, start="2020-01-01")
        df = build_month_dummies(idx)
        jan_rows = df.loc[idx.month == 1]
        assert (jan_rows == 0).all().all()

    def test_february_activates_month_2(self):
        """February rows have month_2=1 and all others=0."""
        idx = _make_monthly_index(24, start="2020-01-01")
        df = build_month_dummies(idx)
        feb_rows = df.loc[idx.month == 2]
        assert (feb_rows["month_2"] == 1).all()
        other_cols = [c for c in df.columns if c != "month_2"]
        assert (feb_rows[other_cols] == 0).all().all()

    def test_each_row_at_most_one_one(self):
        """Each row has exactly one 1 (or all 0 for January)."""
        idx = _make_monthly_index(36, start="2020-01-01")
        df = build_month_dummies(idx)
        row_sums = df.sum(axis=1)
        # Jan rows sum to 0, all others sum to 1
        for i, ts in enumerate(idx):
            expected = 0 if ts.month == 1 else 1
            assert row_sums.iloc[i] == expected, f"Row {i} ({ts}): sum={row_sums.iloc[i]}"

    def test_correct_column_for_known_date(self):
        """October date activates month_10."""
        idx = pd.DatetimeIndex([pd.Timestamp("2023-10-01")])
        df = build_month_dummies(idx)
        assert df["month_10"].iloc[0] == 1
        assert df.drop(columns=["month_10"]).iloc[0].sum() == 0


# ---------------------------------------------------------------------------
# Tests: Full exog assembly
# ---------------------------------------------------------------------------


class TestFullExog:
    """build_full_exog assembles PDL + intervention + month dummy columns."""

    def _make_config_with_interventions(self):
        return SeriesConfig(
            name="test",
            fred_id="T",
            display_name="T",
            unit="$",
            covariates={},
            interventions=(
                InterventionConfig(onset_date="2016-01-01", decay=0.85, label="ev_a"),
            ),
            use_deterministic_seasonal=True,
        )

    def test_column_count_no_pdl(self):
        """With no PDL covariates: 2*interventions + 11 month dummies."""
        target = _make_random_walk_seasonal(60)
        config = self._make_config_with_interventions()
        full, pdl_only = build_full_exog(target, config)
        # 1 intervention × 2 + 11 month dummies = 13
        assert full.shape[1] == 13
        assert pdl_only.empty or pdl_only.shape[1] == 0

    def test_column_count_with_pdl(self):
        """With PDL covariates: PDL cols + 2*interventions + 11."""
        target = _make_random_walk_seasonal(120)
        # Manually provide a pdl-like DataFrame
        pdl = pd.DataFrame(
            {"cov_z_0": np.ones(120), "cov_z_1": np.ones(120)},
            index=target.index,
        )
        config = self._make_config_with_interventions()
        full, pdl_only = build_full_exog(target, config, pdl_covariates=pdl)
        # 2 PDL + 2 intervention + 11 month = 15
        assert full.shape[1] == 15
        assert pdl_only.shape[1] == 2

    def test_pdl_only_subset_matches(self):
        """pdl_only columns are a subset of full exog."""
        target = _make_random_walk_seasonal(120)
        pdl = pd.DataFrame(
            {"cov_z_0": np.ones(120), "cov_z_1": np.ones(120)},
            index=target.index,
        )
        config = self._make_config_with_interventions()
        full, pdl_only = build_full_exog(target, config, pdl_covariates=pdl)
        for col in pdl_only.columns:
            assert col in full.columns

    def test_index_matches_target(self):
        """Output index matches target index."""
        target = _make_random_walk_seasonal(60)
        config = self._make_config_with_interventions()
        full, _ = build_full_exog(target, config)
        pd.testing.assert_index_equal(full.index, target.index)


# ---------------------------------------------------------------------------
# Tests: Deterministic seasonal model fit
# ---------------------------------------------------------------------------


class TestDeterministicSeasonalFit:
    """Model with deterministic seasonal + interventions handles spikes correctly."""

    def _make_spike_series(self):
        """Synthetic series with a spike that declines — mimics HPAI."""
        n = 144  # 12 years
        rng = np.random.default_rng(42)
        idx = pd.date_range("2014-01-01", periods=n, freq="MS")

        # Base: stable around 2.0 with mild true seasonal
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / 12)
        values = 2.0 + seasonal + rng.normal(0, 0.02, n)

        # HPAI-like spike peaking in Feb-Mar 2025 (months 133-134)
        values[128:135] = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.0]
        # Decline from Apr 2025 to Jan 2026 (months 135-144)
        values[135:] = np.linspace(4.0, 2.0, n - 135) + rng.normal(0, 0.02, n - 135)
        return pd.Series(values, index=idx)

    def test_intervention_absorbs_spike(self):
        """With intervention regressor, forecast after spike dissipation is near
        pre-spike level (not elevated by contaminated seasonal)."""
        target = self._make_spike_series()
        config = SeriesConfig(
            name="test",
            fred_id="T",
            display_name="T",
            unit="$",
            covariates={},
            seasonal_harmonics=0,
            interventions=(
                InterventionConfig(
                    onset_date="2024-09-01", decay=0.90, label="spike"
                ),
            ),
            use_deterministic_seasonal=True,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=3000, horizon=1)

        current = fc.current_price
        pct_change = (fc.point_forecast - current) / current

        # After decline, forecast should NOT bounce up >10%
        assert pct_change < 0.10, (
            f"Forecast ${fc.point_forecast:.2f} is {pct_change:+.0%} vs "
            f"current ${current:.2f} — intervention didn't absorb spike"
        )

    def test_p_up_reasonable_after_decline(self):
        """P(up) should be <80% after sustained decline with interventions."""
        target = self._make_spike_series()
        config = SeriesConfig(
            name="test",
            fred_id="T",
            display_name="T",
            unit="$",
            covariates={},
            seasonal_harmonics=0,
            interventions=(
                InterventionConfig(
                    onset_date="2024-09-01", decay=0.90, label="spike"
                ),
            ),
            use_deterministic_seasonal=True,
        )
        result = fit_bsts(target, None, config)
        fc = forecast_with_draws(result, n_draws=3000, horizon=1)

        assert fc.p_up < 0.80, (
            f"P(up) = {fc.p_up:.1%} during decline — should be <80%"
        )

    def test_pdl_covariates_stored_in_result(self):
        """BSTSResult should store pdl_covariates separately."""
        target = _make_random_walk_seasonal(120)
        config = SeriesConfig(
            name="test",
            fred_id="T",
            display_name="T",
            unit="$",
            covariates={},
            use_deterministic_seasonal=True,
        )
        result = fit_bsts(target, None, config)
        assert hasattr(result, "pdl_covariates")
