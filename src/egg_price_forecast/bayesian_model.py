"""Bayesian logistic regression for food commodity price direction prediction.

Trains a pooled model on ~15 BLS food commodities to predict whether a
commodity's price will increase month-over-month. Uses MAP estimation with
a Laplace approximation for posterior uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # logistic sigmoid

from egg_price_forecast.features import (
    decline_streak,
    monthly_seasonal_rate,
    recent_momentum,
)

# Feature names in consistent order
FEATURE_NAMES = [
    "seasonal_up_rate",
    "decline_streak",
    "recent_momentum",
    "cpi_expectation",
    "is_eggs",
]

# Priors: N(mean, std) for each coefficient + intercept
# intercept gets a slight prior toward "prices don't always go up"
_PRIOR_INTERCEPT = (-0.3, 1.0)
_PRIOR_COEFF = (0.0, 2.0)


def build_training_data(
    commodities: dict[str, pd.Series],
    cpi_expectations: pd.Series | None = None,
) -> pd.DataFrame:
    """Build pooled training dataset from multiple commodity price series.

    For each commodity and each month where we have both the target month
    and the preceding month, compute features and binary outcome.

    Args:
        commodities: Dict mapping commodity name to price Series.
        cpi_expectations: Optional Series of 1-year inflation expectations.

    Returns:
        DataFrame with feature columns and 'target' (1 = price went up).
    """
    rows = []

    for name, prices in commodities.items():
        prices = prices.sort_index()
        if len(prices) < 13:
            continue

        # Compute MoM direction for each month
        for i in range(1, len(prices)):
            curr_date = prices.index[i]
            prev_price = prices.iloc[i - 1]
            curr_price = prices.iloc[i]

            target_month_str = curr_date.strftime("%Y-%m")
            month = curr_date.month

            # Use only data up to the month before the target
            historical = prices[prices.index < curr_date]
            if len(historical) < 12:
                continue

            # Features
            seasonal = monthly_seasonal_rate(historical, month)
            streak = decline_streak(historical, target_month_str)
            momentum = recent_momentum(historical, target_month_str, window=3)

            # CPI expectation: find the closest available date
            cpi_val = 0.0
            if cpi_expectations is not None and len(cpi_expectations) > 0:
                cpi_before = cpi_expectations[
                    cpi_expectations.index < curr_date
                ]
                if len(cpi_before) > 0:
                    cpi_val = float(cpi_before.iloc[-1])

            is_eggs = 1.0 if name == "eggs" else 0.0
            went_up = 1.0 if curr_price > prev_price else 0.0

            rows.append(
                {
                    "commodity": name,
                    "date": curr_date,
                    "seasonal_up_rate": seasonal,
                    "decline_streak": float(streak),
                    "recent_momentum": momentum,
                    "cpi_expectation": cpi_val,
                    "is_eggs": is_eggs,
                    "target": went_up,
                }
            )

    return pd.DataFrame(rows)


def _log_prior(coeffs: np.ndarray) -> float:
    """Log prior probability of coefficients under independent normal priors."""
    intercept = coeffs[0]
    betas = coeffs[1:]

    # Intercept prior
    lp = -0.5 * ((intercept - _PRIOR_INTERCEPT[0]) / _PRIOR_INTERCEPT[1]) ** 2

    # Coefficient priors
    for b in betas:
        lp -= 0.5 * ((b - _PRIOR_COEFF[0]) / _PRIOR_COEFF[1]) ** 2

    return lp


def _neg_log_posterior(
    coeffs: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Negative log posterior for MAP estimation."""
    logits = X @ coeffs
    # Log likelihood (numerically stable)
    ll = np.sum(y * logits - np.logaddexp(0, logits))
    lp = _log_prior(coeffs)
    return -(ll + lp)


def _neg_log_posterior_grad(
    coeffs: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Gradient of negative log posterior."""
    logits = X @ coeffs
    probs = expit(logits)
    residuals = y - probs

    # Likelihood gradient
    grad_ll = X.T @ residuals

    # Prior gradient
    grad_prior = np.zeros_like(coeffs)
    grad_prior[0] = -(coeffs[0] - _PRIOR_INTERCEPT[0]) / _PRIOR_INTERCEPT[1] ** 2
    for i in range(1, len(coeffs)):
        grad_prior[i] = -(coeffs[i] - _PRIOR_COEFF[0]) / _PRIOR_COEFF[1] ** 2

    return -(grad_ll + grad_prior)


def fit_bayesian_logistic(training_data: pd.DataFrame) -> dict:
    """Fit Bayesian logistic regression via MAP estimation.

    Uses L-BFGS-B optimization with analytic gradients. Computes
    Hessian-based standard errors via Laplace approximation.

    Args:
        training_data: DataFrame from build_training_data.

    Returns:
        Dict with 'coefficients' (array), 'std_errors' (array),
        'feature_names' (list), 'n_obs' (int), 'n_commodities' (int).
    """
    X_features = training_data[FEATURE_NAMES].values
    # Add intercept column
    X = np.column_stack([np.ones(len(X_features)), X_features])
    y = training_data["target"].values

    n_params = X.shape[1]
    x0 = np.zeros(n_params)

    result = minimize(
        _neg_log_posterior,
        x0,
        args=(X, y),
        jac=_neg_log_posterior_grad,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    coeffs = result.x

    # Laplace approximation: Hessian of neg log posterior at MAP
    # Compute numerically via finite differences on gradient
    eps = 1e-5
    n = len(coeffs)
    hessian = np.zeros((n, n))
    grad_at_map = _neg_log_posterior_grad(coeffs, X, y)
    for i in range(n):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        grad_plus = _neg_log_posterior_grad(coeffs_plus, X, y)
        hessian[i] = (grad_plus - grad_at_map) / eps

    # Make symmetric
    hessian = (hessian + hessian.T) / 2

    # Standard errors from inverse Hessian
    try:
        cov = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        std_errors = np.full(n, np.nan)

    n_commodities = training_data["commodity"].nunique()

    return {
        "coefficients": coeffs,
        "std_errors": std_errors,
        "feature_names": ["intercept"] + FEATURE_NAMES,
        "n_obs": len(y),
        "n_commodities": n_commodities,
    }


def predict_probability(fitted_model: dict, features: dict) -> float:
    """Predict probability of price increase given features.

    Args:
        fitted_model: Output of fit_bayesian_logistic.
        features: Dict with keys matching FEATURE_NAMES.

    Returns:
        Probability in [0, 1].
    """
    coeffs = fitted_model["coefficients"]
    x = np.array(
        [1.0] + [features[name] for name in FEATURE_NAMES]
    )
    logit = float(x @ coeffs)
    return float(expit(logit))


def decompose_contributions(fitted_model: dict, features: dict) -> dict:
    """Decompose probability into feature contributions in logit space.

    Each feature's contribution is its coefficient × feature value,
    converted to approximate probability-space contribution.

    Args:
        fitted_model: Output of fit_bayesian_logistic.
        features: Dict with keys matching FEATURE_NAMES.

    Returns:
        Dict mapping component names to their logit-space contributions.
    """
    coeffs = fitted_model["coefficients"]
    feature_values = np.array(
        [1.0] + [features[name] for name in FEATURE_NAMES]
    )
    contributions = coeffs * feature_values
    names = ["intercept"] + FEATURE_NAMES

    # Map to friendly component names
    name_map = {
        "intercept": "intercept",
        "seasonal_up_rate": "seasonal",
        "decline_streak": "streak",
        "recent_momentum": "momentum",
        "cpi_expectation": "cpi_signal",
        "is_eggs": "egg_specific",
    }

    return {
        name_map.get(n, n): float(contributions[i])
        for i, n in enumerate(names)
    }
