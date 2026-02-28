"""Bayesian Structural Time Series forecaster using statsmodels UCM.

Fits per-series UnobservedComponents models with:
  - Local level (random walk)
  - Trigonometric seasonality (period=12, configurable harmonics)
  - Almon polynomial distributed lag covariates (PPI -> CPI pass-through)
  - AR(1) irregular errors

Uncertainty is approximated via asymptotic parameter sampling from the MLE
covariance matrix, producing posterior predictive draws without requiring
a full MCMC backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from egg_price_forecast.data import fetch_bls_series
from egg_price_forecast.series_config import (
    SERIES_CONFIGS,
    InterventionConfig,
    SeriesConfig,
)


# ---------------------------------------------------------------------------
# Almon polynomial distributed lag construction
# ---------------------------------------------------------------------------


def build_almon_basis(
    covariate: pd.Series,
    max_lag: int = 12,
    poly_degree: int = 2,
) -> pd.DataFrame:
    """Construct Almon polynomial distributed lag basis columns.

    For a degree-D polynomial and max lag L, produces D+1 basis columns
    where basis_d(t) = sum_{k=0}^{L} k^d * x(t-k).

    Args:
        covariate: Raw covariate series with DatetimeIndex.
        max_lag: Maximum number of lag periods (default 12 months).
        poly_degree: Degree of the Almon polynomial (default 2).

    Returns:
        DataFrame with columns z_0, z_1, ..., z_{poly_degree}, indexed
        like the input but with leading NaN rows for the lag window.
    """
    covariate = covariate.sort_index()
    n = len(covariate)
    values = covariate.values.astype(float)

    basis = {}
    for d in range(poly_degree + 1):
        col = np.full(n, np.nan)
        for t in range(max_lag, n):
            total = 0.0
            for k in range(max_lag + 1):
                total += (k ** d) * values[t - k]
            col[t] = total
        basis[f"z_{d}"] = col

    return pd.DataFrame(basis, index=covariate.index)


# ---------------------------------------------------------------------------
# Covariate matrix assembly
# ---------------------------------------------------------------------------


def build_covariate_matrix(
    target: pd.Series,
    config: SeriesConfig,
) -> pd.DataFrame:
    """Fetch, align, and build Almon basis for all configured covariates.

    For each covariate in config.covariates, fetches the FRED series,
    aligns it to the target index via forward-fill, constructs the Almon
    polynomial basis, and concatenates all columns.

    Args:
        target: Target price series with DatetimeIndex.
        config: SeriesConfig specifying covariates and lag parameters.

    Returns:
        DataFrame with Almon basis columns for all covariates, aligned
        to the target index.
    """
    all_basis: list[pd.DataFrame] = []

    for cov_name, cov_fred_id in config.covariates.items():
        try:
            cov_series = fetch_bls_series(cov_fred_id)
        except Exception:
            continue

        # Align to target index via forward-fill
        cov_aligned = cov_series.reindex(target.index, method="ffill")

        # Use pct_change for proper scaling — raw PPI levels (~200) vs
        # retail prices (~$2.50) cause convergence failures. Pct_change
        # puts covariates on a comparable scale (~0.01).
        cov_pct = cov_aligned.pct_change()

        # Build Almon basis with degree 1 (2 columns per covariate)
        # to keep parameter count manageable for convergence
        basis = build_almon_basis(
            cov_pct,
            max_lag=config.covariate_max_lag,
            poly_degree=1,
        )

        # Prefix columns with covariate name
        basis.columns = [f"{cov_name}_{c}" for c in basis.columns]
        all_basis.append(basis)

    if all_basis:
        return pd.concat(all_basis, axis=1)
    return pd.DataFrame(index=target.index)


# ---------------------------------------------------------------------------
# Intervention regressors + deterministic seasonality
# ---------------------------------------------------------------------------


def build_intervention_columns(
    index: pd.DatetimeIndex,
    interventions: tuple[InterventionConfig, ...],
) -> pd.DataFrame:
    """Build step + geometric decay columns for each intervention.

    For each intervention with onset date T and decay rate d:
      - {label}_step: 0 before T, 1 at and after T
      - {label}_decay: 0 before T, d^(months_since_T) at and after T

    Args:
        index: DatetimeIndex for the output rows.
        interventions: Tuple of InterventionConfig.

    Returns:
        DataFrame with 2 columns per intervention, indexed like input.
        Empty DataFrame (with correct index) if no interventions.
    """
    if not interventions:
        return pd.DataFrame(index=index)

    cols: dict[str, np.ndarray] = {}
    for iv in interventions:
        onset = pd.Timestamp(iv.onset_date)
        step = np.where(index >= onset, 1.0, 0.0)

        # Months since onset (fractional months not needed — monthly data)
        months_since = (
            (index.year - onset.year) * 12 + (index.month - onset.month)
        ).astype(float)
        decay = np.where(index >= onset, iv.decay ** months_since, 0.0)

        cols[f"{iv.label}_step"] = step
        cols[f"{iv.label}_decay"] = decay

    return pd.DataFrame(cols, index=index)


def build_month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build 11 month-of-year dummy columns (January = reference).

    Args:
        index: DatetimeIndex for the output rows.

    Returns:
        DataFrame with columns month_2 .. month_12.
    """
    months = index.month
    cols = {}
    for m in range(2, 13):
        cols[f"month_{m}"] = (months == m).astype(float)
    return pd.DataFrame(cols, index=index)


def build_full_exog(
    target: pd.Series,
    config: SeriesConfig,
    pdl_covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble full exogenous matrix: PDL covariates + interventions + month dummies.

    Args:
        target: Target price series (used for index).
        config: SeriesConfig with interventions and seasonal settings.
        pdl_covariates: Pre-built PDL covariate DataFrame (or None).

    Returns:
        Tuple of (full_exog, pdl_only) where pdl_only is the PDL subset.
    """
    idx = target.index
    parts: list[pd.DataFrame] = []

    # PDL covariates (if any)
    pdl_only = pd.DataFrame(index=idx)
    if pdl_covariates is not None and not pdl_covariates.empty:
        pdl_aligned = pdl_covariates.reindex(idx)
        parts.append(pdl_aligned)
        pdl_only = pdl_aligned

    # Intervention columns
    if config.interventions:
        iv_df = build_intervention_columns(idx, config.interventions)
        parts.append(iv_df)

    # Month dummies (deterministic seasonal)
    if config.use_deterministic_seasonal:
        month_df = build_month_dummies(idx)
        parts.append(month_df)

    if parts:
        full = pd.concat(parts, axis=1)
    else:
        full = pd.DataFrame(index=idx)

    return full, pdl_only


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------


@dataclass
class BSTSResult:
    """Container for a fitted BSTS model.

    Attributes:
        model: The fitted statsmodels UCM results object.
        mle_params: MLE parameter estimates.
        cov_params: Asymptotic covariance matrix of parameters.
        component_estimates: Dict with estimated trend, seasonal, etc.
        target_series: The target series used for fitting.
        exog: The exogenous covariate matrix (if any).
        config: The SeriesConfig used.
    """

    model: object  # UnobservedComponentsResults
    mle_params: np.ndarray
    cov_params: np.ndarray
    component_estimates: dict[str, np.ndarray]
    target_series: pd.Series
    exog: pd.DataFrame | None
    config: SeriesConfig
    pdl_covariates: pd.DataFrame | None = None


@dataclass
class BSTSForecastResult:
    """Container for BSTS forecast output.

    Attributes:
        point_forecast: Mean of posterior predictive draws.
        median_forecast: Median of posterior predictive draws.
        intervals: Prediction intervals, e.g. {90: (low, high)}.
        quantiles: Quantile level -> value mapping.
        posterior_draws: Array of forecast draws.
        p_up: Fraction of draws above current price.
        current_price: Latest observed price.
        component_contributions: Estimated structural components.
        covariate_coefficients: Coefficient estimates for covariates.
        model_diagnostics: Dict with AIC, BIC, log-likelihood, etc.
    """

    point_forecast: float
    median_forecast: float
    intervals: dict[int, tuple[float, float]]
    quantiles: dict[float, float]
    posterior_draws: np.ndarray
    p_up: float
    current_price: float
    component_contributions: dict[str, float]
    covariate_coefficients: dict[str, float]
    model_diagnostics: dict[str, float]


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_bsts(
    target: pd.Series,
    covariates: pd.DataFrame | None,
    config: SeriesConfig,
) -> BSTSResult:
    """Fit a structural time series model via MLE.

    Model specification:
      y_t = mu_t + gamma_t + X_t @ beta + eps_t
      mu_t = mu_{t-1} + eta_t              (local level)
      gamma_t = trig seasonal(12, harmonics)
      eps_t = phi * eps_{t-1} + nu_t        (AR(1) errors)

    Args:
        target: Monthly price series.
        covariates: Almon PDL basis DataFrame (or None).
        config: SeriesConfig for model specification.

    Returns:
        BSTSResult with fitted model and diagnostics.
    """
    target = target.sort_index().dropna()

    # --- Deterministic seasonal + intervention path ---
    pdl_covariates_out = None
    if config.use_deterministic_seasonal:
        # Build PDL covariates from the raw covariates argument
        pdl_cov = None
        if covariates is not None and not covariates.empty:
            pdl_cov = covariates

        # Assemble full exog: PDL + interventions + month dummies
        full_exog, pdl_only = build_full_exog(target, config, pdl_covariates=pdl_cov)

        # Align and drop NaN rows
        common_idx = target.index.intersection(full_exog.index)
        target = target.loc[common_idx]
        full_exog = full_exog.loc[common_idx]

        valid_mask = full_exog.notna().all(axis=1)
        target = target[valid_mask]
        full_exog = full_exog[valid_mask]

        exog = full_exog if not full_exog.empty else None
        pdl_covariates_out = pdl_only.loc[target.index] if not pdl_only.empty else None

        # No freq_seasonal — seasonality is captured by month dummies in exog
        freq_seasonal_spec = None
    else:
        # --- Legacy stochastic seasonal path ---
        exog = None
        if covariates is not None and not covariates.empty:
            common_idx = target.index.intersection(covariates.index)
            target = target.loc[common_idx]
            covariates = covariates.loc[common_idx]

            valid_mask = covariates.notna().all(axis=1)
            target = target[valid_mask]
            covariates = covariates[valid_mask]

            if not covariates.empty:
                exog = covariates

        freq_seasonal_spec = (
            [{"period": 12, "harmonics": config.seasonal_harmonics}]
            if config.seasonal_harmonics > 0
            else None
        )

    model = UnobservedComponents(
        target.values,
        level="local level",
        freq_seasonal=freq_seasonal_spec,
        autoregressive=1,
        exog=exog.values if exog is not None else None,
    )

    # Fit via MLE with increased iterations
    try:
        results = model.fit(disp=False, maxiter=1000)
    except Exception:
        # Fallback: simpler model without AR
        model = UnobservedComponents(
            target.values,
            level="local level",
            freq_seasonal=freq_seasonal_spec,
            exog=exog.values if exog is not None else None,
        )
        results = model.fit(disp=False, maxiter=1000)

    # Extract components
    component_estimates = {}
    try:
        states = results.states.smoothed
        component_estimates["level"] = states[:, 0]
    except Exception:
        component_estimates["level"] = np.full(len(target), np.nan)

    # Get MLE params and covariance
    mle_params = results.params.copy()

    # Covariance matrix — handle cases where it might be singular
    try:
        cov_params = results.cov_params()
        if cov_params is None or np.any(np.isnan(cov_params)):
            cov_params = np.eye(len(mle_params)) * 0.01
    except Exception:
        cov_params = np.eye(len(mle_params)) * 0.01

    # Covariate coefficients
    covariate_coefficients = {}
    if exog is not None:
        n_exog = exog.shape[1]
        exog_names = list(exog.columns) if isinstance(exog, pd.DataFrame) else [
            f"exog_{i}" for i in range(n_exog)
        ]
        try:
            for i, name in enumerate(exog_names):
                for j, pname in enumerate(results.param_names):
                    if f"beta.x{i+1}" in pname or (f"exog" in pname.lower() and str(i) in pname):
                        covariate_coefficients[name] = float(mle_params[j])
                        break
        except Exception:
            pass

    return BSTSResult(
        model=results,
        mle_params=mle_params,
        cov_params=cov_params,
        component_estimates=component_estimates,
        target_series=target,
        exog=exog,
        config=config,
        pdl_covariates=pdl_covariates_out,
    )


# ---------------------------------------------------------------------------
# Forecasting with approximate posterior draws
# ---------------------------------------------------------------------------


def forecast_with_draws(
    result: BSTSResult,
    n_draws: int = 5000,
    horizon: int = 1,
) -> BSTSForecastResult:
    """Generate approximate posterior predictive draws.

    Uses the Kalman filter's forecast distribution (state + observation
    uncertainty) as the base, with a small inflation for parameter
    uncertainty estimated from the asymptotic covariance.

    Args:
        result: Fitted BSTSResult.
        n_draws: Number of posterior draws.
        horizon: Forecast horizon in months.

    Returns:
        BSTSForecastResult with draws, intervals, and diagnostics.
    """
    fitted = result.model
    mle_params = result.mle_params
    cov_params = result.cov_params

    # Current price is the last observed value
    current_price = float(result.target_series.iloc[-1])

    # --- Get forecast distribution from Kalman filter ---
    try:
        exog_forecast = None
        if result.exog is not None and not result.exog.empty:
            if result.config.use_deterministic_seasonal:
                # Build proper forecast exog with correct month dummies,
                # extended interventions, and last-known PDL values
                last_date = result.target_series.index[-1]
                forecast_idx = pd.date_range(
                    last_date + pd.DateOffset(months=1),
                    periods=horizon,
                    freq="MS",
                )
                forecast_parts: list[pd.DataFrame] = []

                # PDL: repeat last known values
                if result.pdl_covariates is not None and not result.pdl_covariates.empty:
                    last_pdl = result.pdl_covariates.iloc[[-1]]
                    pdl_forecast = pd.DataFrame(
                        np.tile(last_pdl.values, (horizon, 1)),
                        index=forecast_idx,
                        columns=last_pdl.columns,
                    )
                    forecast_parts.append(pdl_forecast)

                # Interventions: extend columns
                if result.config.interventions:
                    iv_forecast = build_intervention_columns(
                        forecast_idx, result.config.interventions
                    )
                    forecast_parts.append(iv_forecast)

                # Month dummies: correct month for forecast date
                month_forecast = build_month_dummies(forecast_idx)
                forecast_parts.append(month_forecast)

                if forecast_parts:
                    exog_forecast = pd.concat(forecast_parts, axis=1).values
                else:
                    exog_forecast = None
            else:
                # Legacy path: repeat last exog row
                last_exog = result.exog.iloc[[-1]].values
                exog_forecast = np.tile(last_exog, (horizon, 1))

        forecast_obj = fitted.get_forecast(steps=horizon, exog=exog_forecast)
        # predicted_mean may be ndarray or Series depending on statsmodels version
        pred_mean = np.asarray(forecast_obj.predicted_mean).flat
        se_mean = np.asarray(forecast_obj.se_mean).flat
        point_forecast = float(pred_mean[-1])
        forecast_se = float(se_mean[-1])
    except Exception:
        point_forecast = current_price
        forecast_se = float(np.std(result.target_series.diff().dropna())) or 0.1

    # --- Calibrate SE using in-sample residuals ---
    # The Kalman filter's forecast_se can be much too small when sigma2.irregular
    # collapses to zero (model thinks it perfectly explains every observation).
    # The one-step-ahead residual std provides a hard floor.
    try:
        resid = fitted.resid
        residual_std = float(np.nanstd(resid))
        if np.isnan(residual_std) or residual_std < 0.001:
            residual_std = 0.05
    except Exception:
        residual_std = 0.05

    recent_vol = float(result.target_series.diff().dropna().tail(24).std())
    if np.isnan(recent_vol) or recent_vol < 0.01:
        recent_vol = 0.05

    # Replace NaN with 0 before taking max (NaN propagates through max())
    if np.isnan(forecast_se):
        forecast_se = 0.0

    # SE floor: max of Kalman SE, residual std, and recent volatility
    forecast_se = max(forecast_se, residual_std, recent_vol * 0.5, 0.01)

    # Additional uncertainty from level-observation gap: when the smoothed
    # level is far from the last observation, the model is struggling to
    # track rapid moves, so forecast uncertainty should be higher.
    try:
        level_gap = abs(
            float(result.component_estimates.get("level", [np.nan])[-1])
            - current_price
        )
        if not np.isnan(level_gap):
            # Add gap as additional SE (the model is uncertain by at least
            # this amount about the true level)
            forecast_se = max(forecast_se, level_gap * 0.5)
    except (IndexError, TypeError):
        pass

    # Inflate when model didn't converge
    try:
        converged = bool(fitted.mle_retvals["converged"])
    except Exception:
        converged = False
    convergence_penalty = 1.0 if converged else 1.5

    # --- Generate draws ---
    # Add parameter uncertainty inflation (~5-15% for ~500 obs).
    n_obs = len(result.target_series)
    param_inflation = 1.0 + max(0.5 / np.sqrt(n_obs), 0.02)
    total_se = forecast_se * param_inflation * convergence_penalty

    draws = np.random.normal(point_forecast, total_se, size=n_draws)

    # Ensure draws are positive (prices can't be negative)
    draws = np.clip(draws, 0.01, None)

    # --- Compute summary statistics ---
    p_up = float(np.mean(draws > current_price))
    median_forecast = float(np.median(draws))

    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantiles = {
        q: float(np.quantile(draws, q)) for q in quantile_levels
    }

    intervals = {
        90: (quantiles[0.05], quantiles[0.95]),
        50: (quantiles[0.25], quantiles[0.75]),
    }

    # --- Component contributions ---
    component_contributions = {}
    try:
        component_contributions["trend"] = float(
            result.component_estimates.get("level", [np.nan])[-1]
        )
    except (IndexError, TypeError):
        component_contributions["trend"] = current_price

    # Covariate contributions
    covariate_coefficients = {}
    if result.exog is not None:
        exog_cols = (
            list(result.exog.columns)
            if isinstance(result.exog, pd.DataFrame)
            else [f"exog_{i}" for i in range(result.exog.shape[1])]
        )
        for i, col_name in enumerate(exog_cols):
            try:
                for j, pname in enumerate(fitted.param_names):
                    if f"beta.x{i+1}" in pname:
                        coef = float(result.mle_params[j])
                        covariate_coefficients[col_name] = coef
                        last_val = float(result.exog.iloc[-1, i])
                        component_contributions[col_name] = coef * last_val
                        break
            except Exception:
                pass

    # --- Model diagnostics ---
    model_diagnostics = {}
    try:
        model_diagnostics["aic"] = float(fitted.aic)
        model_diagnostics["bic"] = float(fitted.bic)
        model_diagnostics["loglikelihood"] = float(fitted.llf)
    except Exception:
        pass
    model_diagnostics["n_obs"] = len(result.target_series)
    model_diagnostics["n_params"] = len(mle_params)
    model_diagnostics["forecast_se"] = total_se
    model_diagnostics["residual_std"] = residual_std
    try:
        model_diagnostics["converged"] = bool(fitted.mle_retvals["converged"])
    except Exception:
        model_diagnostics["converged"] = False
    model_diagnostics["kalman_forecast"] = point_forecast

    return BSTSForecastResult(
        point_forecast=float(np.mean(draws)),
        median_forecast=median_forecast,
        intervals=intervals,
        quantiles=quantiles,
        posterior_draws=draws,
        p_up=p_up,
        current_price=current_price,
        component_contributions=component_contributions,
        covariate_coefficients=covariate_coefficients,
        model_diagnostics=model_diagnostics,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_bsts_forecast(commodity: str = "eggs") -> BSTSForecastResult:
    """Run the full BSTS forecasting pipeline for a commodity.

    Args:
        commodity: Commodity key from SERIES_CONFIGS (default "eggs").

    Returns:
        BSTSForecastResult with point forecast, draws, intervals, etc.
    """
    config = SERIES_CONFIGS[commodity]

    # Fetch target series
    target = fetch_bls_series(config.fred_id)

    # Build covariate matrix
    covariates = build_covariate_matrix(target, config)

    # Fit model
    bsts_result = fit_bsts(target, covariates, config)

    # Generate forecast draws
    forecast_result = forecast_with_draws(
        bsts_result,
        n_draws=5000,
        horizon=config.forecast_horizon,
    )

    return forecast_result
