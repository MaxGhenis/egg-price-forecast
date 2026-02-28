"""Probability estimation model for BLS egg price direction.

Uses Bayesian logistic regression trained on multiple BLS food commodity
price series to predict whether egg prices will increase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from egg_price_forecast.bayesian_model import (
    FEATURE_NAMES,
    build_training_data,
    decompose_contributions,
    fit_bayesian_logistic,
    predict_probability,
)
from egg_price_forecast.data import (
    COMMODITY_SERIES,
    fetch_all_commodities,
    fetch_bls_series,
    fetch_cpi_expectations,
)
from egg_price_forecast.features import (
    decline_streak,
    monthly_seasonal_rate,
    recent_momentum,
)


def _next_february(prices: pd.Series) -> str:
    """Determine the next February target month from the price series.

    Returns format "YYYY-MM" for the February following the latest data point.
    """
    last_date = prices.index.max()
    if last_date.month >= 2:
        target_year = last_date.year + 1
    else:
        target_year = last_date.year
    return f"{target_year}-02"


def _compute_egg_features(
    prices: pd.Series,
    target_month: str,
    cpi_expectations: pd.Series | None = None,
) -> dict:
    """Compute feature dict for egg prediction at target_month."""
    month = int(target_month.split("-")[1])
    target_ts = pd.Timestamp(target_month + "-01")

    seasonal = monthly_seasonal_rate(prices, month)
    streak = decline_streak(prices, target_month)
    momentum = recent_momentum(prices, target_month, window=3)

    cpi_val = 0.0
    if cpi_expectations is not None and len(cpi_expectations) > 0:
        cpi_before = cpi_expectations[cpi_expectations.index < target_ts]
        if len(cpi_before) > 0:
            cpi_val = float(cpi_before.iloc[-1])

    return {
        "seasonal_up_rate": seasonal,
        "decline_streak": float(streak),
        "recent_momentum": momentum,
        "cpi_expectation": cpi_val,
        "is_eggs": 1.0,
    }


def estimate_probability(
    prices: pd.Series, wholesale: float | None = None
) -> dict:
    """Estimate probability that egg prices increase in February.

    Trains a Bayesian logistic regression on multiple BLS food commodity
    series and predicts the egg-specific probability.

    Args:
        prices: BLS egg prices with DatetimeIndex (monthly).
        wholesale: Optional current wholesale price per dozen (unused in
            Bayesian model, kept for API compatibility).

    Returns:
        Dict with 'probability' and 'components'.
    """
    target_month = _next_february(prices)

    # Fetch training data
    commodities = fetch_all_commodities()
    # Ensure eggs are in the training data using the passed-in prices
    commodities["eggs"] = prices

    try:
        cpi_expectations = fetch_cpi_expectations()
    except Exception:
        cpi_expectations = None

    # Build training data and fit model
    training_data = build_training_data(commodities, cpi_expectations)
    fitted = fit_bayesian_logistic(training_data)

    # Compute egg-specific features for the target month
    features = _compute_egg_features(prices, target_month, cpi_expectations)

    # Predict
    probability = predict_probability(fitted, features)

    # Clamp to [0.05, 0.95]
    probability = float(np.clip(probability, 0.05, 0.95))

    # Decompose contributions
    components = decompose_contributions(fitted, features)

    return {
        "probability": probability,
        "components": components,
        "model_info": {
            "n_obs": fitted["n_obs"],
            "n_commodities": fitted["n_commodities"],
            "coefficients": {
                name: float(c)
                for name, c in zip(
                    fitted["feature_names"], fitted["coefficients"]
                )
            },
        },
    }


def backtest(prices: pd.Series, start_year: int = 2010) -> dict:
    """Backtest the Bayesian model on historical price changes.

    For each month from start_year onward where we have both the target
    month and preceding month data, uses only prior data to train and
    checks if the model's prediction matched reality.

    Args:
        prices: BLS egg prices with DatetimeIndex (monthly).
        start_year: First year to include in backtest.

    Returns:
        Dict with 'predictions' list, 'accuracy', and 'brier_score'.
    """
    predictions = []

    # Find all years where we have February data
    years_with_feb = sorted(
        set(prices.index[prices.index.month == 2].year)
    )

    # Fetch other commodities once for efficiency
    all_commodities = fetch_all_commodities()

    try:
        cpi_expectations = fetch_cpi_expectations()
    except Exception:
        cpi_expectations = None

    for year in years_with_feb:
        if year < start_year:
            continue

        # Use data only up to January of that year
        cutoff = pd.Timestamp(year=year, month=1, day=31)
        historical_eggs = prices[prices.index <= cutoff]

        if len(historical_eggs) < 12:
            continue

        # Filter all commodities to pre-cutoff data
        historical_commodities = {}
        for name, series in all_commodities.items():
            hist = series[series.index <= cutoff]
            if len(hist) >= 12:
                historical_commodities[name] = hist
        historical_commodities["eggs"] = historical_eggs

        # Filter CPI expectations
        historical_cpi = None
        if cpi_expectations is not None:
            historical_cpi = cpi_expectations[
                cpi_expectations.index <= cutoff
            ]
            if len(historical_cpi) == 0:
                historical_cpi = None

        # Build training data and fit on pre-cutoff data only
        training_data = build_training_data(
            historical_commodities, historical_cpi
        )

        if len(training_data) < 50:
            continue

        fitted = fit_bayesian_logistic(training_data)

        # Compute features for February of this year
        target_month = f"{year}-02"
        features = _compute_egg_features(
            historical_eggs, target_month, historical_cpi
        )

        predicted_prob = predict_probability(fitted, features)
        predicted_prob = float(np.clip(predicted_prob, 0.05, 0.95))

        # Get actual: did price go up in February vs January?
        jan_mask = (prices.index.year == year) & (prices.index.month == 1)
        feb_mask = (prices.index.year == year) & (prices.index.month == 2)

        jan_prices = prices[jan_mask]
        feb_prices = prices[feb_mask]

        if len(jan_prices) == 0 or len(feb_prices) == 0:
            continue

        actual_up = bool(feb_prices.iloc[0] > jan_prices.iloc[0])
        predicted_up = predicted_prob >= 0.5
        correct = predicted_up == actual_up

        predictions.append(
            {
                "year": year,
                "predicted_prob": predicted_prob,
                "actual_up": actual_up,
                "correct": correct,
            }
        )

    if len(predictions) == 0:
        return {
            "predictions": [],
            "accuracy": 0.0,
            "brier_score": 1.0,
        }

    accuracy = sum(p["correct"] for p in predictions) / len(predictions)
    brier_score = np.mean(
        [
            (p["predicted_prob"] - float(p["actual_up"])) ** 2
            for p in predictions
        ]
    )

    return {
        "predictions": predictions,
        "accuracy": float(accuracy),
        "brier_score": float(brier_score),
    }
