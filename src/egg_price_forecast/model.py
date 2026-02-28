"""Probability estimation model for BLS egg price direction.

Uses LightGBM quantile regression trained on multiple BLS food commodity
price series to predict whether egg prices will increase.
"""

from __future__ import annotations

import pandas as pd


def estimate_probability(prices: pd.Series, brier_score: float = 0.325) -> dict:
    """Estimate probability of egg price increase using simulation-based forecasting.

    Args:
        prices: BLS egg prices with DatetimeIndex (monthly).

    Returns:
        Dict with probability, point_forecast, intervals, components,
        model_info, quantiles, and samples_summary.
    """
    from .forecaster import run_forecast

    # Get current price from latest observation
    current_price = float(prices.iloc[-1])

    # Run the quantile forecast
    result = run_forecast(current_price=current_price)

    # Calibrate probability
    from .features import february_seasonal_rate
    from .calibration import calibrate_probability

    base_rate = february_seasonal_rate(prices)
    calibration = calibrate_probability(
        raw_p_up=result.p_up,
        base_rate=base_rate,
        brier_score=brier_score,
    )

    return {
        "probability": calibration["calibrated_p_up"],
        "raw_probability": result.p_up,
        "calibration": calibration,
        "point_forecast": result.point_forecast,
        "intervals": {
            str(k): {"low": float(v[0]), "high": float(v[1])}
            for k, v in result.intervals.items()
        },
        "components": {
            # Top 8 feature importances, normalized to sum to contributions
            k: float(v)
            for k, v in sorted(
                result.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:8]
        },
        "model_info": {
            "type": "lightgbm_quantile",
            "n_quantile_models": 7,
            "n_samples": len(result.samples),
            "feature_importances": {
                k: float(v) for k, v in result.feature_importances.items()
            },
        },
        "quantiles": {str(k): float(v) for k, v in result.quantiles.items()},
        "samples_summary": {
            "mean": float(result.samples.mean()),
            "std": float(result.samples.std()),
            "n": len(result.samples),
        },
    }


def backtest(prices: pd.Series, start_year: int = 2010) -> dict:
    """Expanding-window backtest of the quantile forecast model.

    Evaluates January predictions (forecasting February) to match the
    Kalshi use case. At each January from start_year onward, trains on
    all prior data and predicts the following month's price.

    Args:
        prices: BLS egg prices with DatetimeIndex (monthly).
        start_year: First year to include in backtest.

    Returns:
        Dict with predictions list, direction_accuracy, mape,
        coverage_90, coverage_50, brier_score, and accuracy.
    """
    from .data import fetch_all_commodities, fetch_cpi_expectations, fetch_ppi_eggs
    from .forecaster import (
        _get_feature_columns,
        build_forecast_features,
        predict_quantiles,
        sample_from_quantiles,
        train_quantile_models,
    )

    commodities = fetch_all_commodities()
    cpi = fetch_cpi_expectations()
    try:
        ppi_eggs = fetch_ppi_eggs()
    except Exception:
        ppi_eggs = None

    full_df = build_forecast_features(commodities, cpi, ppi_eggs=ppi_eggs)

    # Evaluate only January egg rows (predicting February)
    egg_df = full_df[full_df["commodity"] == "eggs"].copy()
    egg_df = egg_df.sort_values("date")
    jan_eggs = egg_df[
        (egg_df["date"].dt.month == 1) &
        (egg_df["date"].dt.year >= start_year) &
        egg_df["target"].notna()
    ]

    predictions = []
    feature_cols = _get_feature_columns(full_df)

    for _, row in jan_eggs.iterrows():
        # Train on all data before this date (all commodities)
        train_df = full_df[full_df["date"] < row["date"]].dropna(subset=["target"])
        if len(train_df) < 100:
            continue

        models = train_quantile_models(train_df)
        features_row = row[feature_cols].to_frame().T
        q_preds = predict_quantiles(models, features_row)
        samples = sample_from_quantiles(q_preds)

        actual_price = row["target"]
        current_price = row["price_level"]
        predicted_prob = float((samples > current_price).mean())
        actual_up = actual_price > current_price

        predictions.append(
            {
                "date": str(row["date"].date()),
                "year": int(row["date"].year),
                "month": int(row["date"].month),
                "predicted_prob": round(predicted_prob, 3),
                "predicted_price": round(float(q_preds[0.5]), 3),
                "actual_price": round(float(actual_price), 3),
                "current_price": round(float(current_price), 3),
                "actual_up": bool(actual_up),
                "correct": bool((predicted_prob > 0.5) == actual_up),
                "error": round(float(abs(q_preds[0.5] - actual_price)), 3),
                "in_90_interval": bool(
                    q_preds[0.05] <= actual_price <= q_preds[0.95]
                ),
                "in_50_interval": bool(
                    q_preds[0.25] <= actual_price <= q_preds[0.75]
                ),
            }
        )

    if not predictions:
        return {
            "predictions": [],
            "direction_accuracy": 0,
            "mape": 0,
            "coverage_90": 0,
            "coverage_50": 0,
            "brier_score": 0,
            "accuracy": 0,
        }

    n = len(predictions)
    direction_accuracy = sum(p["correct"] for p in predictions) / n
    mape = sum(p["error"] / max(p["actual_price"], 0.01) for p in predictions) / n
    coverage_90 = sum(p["in_90_interval"] for p in predictions) / n
    coverage_50 = sum(p["in_50_interval"] for p in predictions) / n
    brier_score = sum(
        (p["predicted_prob"] - (1.0 if p["actual_up"] else 0.0)) ** 2
        for p in predictions
    ) / n

    return {
        "predictions": predictions,
        "direction_accuracy": round(direction_accuracy, 3),
        "mape": round(mape, 3),
        "coverage_90": round(coverage_90, 3),
        "coverage_50": round(coverage_50, 3),
        "brier_score": round(brier_score, 3),
        "accuracy": round(direction_accuracy, 3),
    }


# ---------------------------------------------------------------------------
# BSTS model interface
# ---------------------------------------------------------------------------


def estimate_probability_bsts(
    prices: pd.Series,
    commodity: str = "eggs",
) -> dict:
    """Estimate probability of price increase using BSTS structural model.

    Uses UnobservedComponents (local level + seasonal + Almon PDL covariates
    + AR(1) errors) with approximate posterior predictive draws.

    Args:
        prices: BLS price series with DatetimeIndex (monthly).
        commodity: Commodity key from SERIES_CONFIGS.

    Returns:
        Dict with same schema as estimate_probability() for backward
        compatibility: probability, point_forecast, intervals, components,
        model_info, quantiles, samples_summary.
    """
    from .bsts_forecaster import run_bsts_forecast

    result = run_bsts_forecast(commodity=commodity)

    return {
        "probability": result.p_up,
        "raw_probability": result.p_up,
        "calibration": {
            "calibrated_p_up": result.p_up,
            "raw_p_up": result.p_up,
            "base_rate": result.p_up,
            "model_weight": 1.0,
            "brier_score": 0.0,
        },
        "point_forecast": result.point_forecast,
        "intervals": {
            str(k): {"low": float(v[0]), "high": float(v[1])}
            for k, v in result.intervals.items()
        },
        "components": result.component_contributions,
        "model_info": {
            "type": "bsts_ucm",
            "n_quantile_models": 0,
            "n_samples": len(result.posterior_draws),
            "feature_importances": result.covariate_coefficients,
        },
        "quantiles": {str(k): float(v) for k, v in result.quantiles.items()},
        "samples_summary": {
            "mean": float(result.posterior_draws.mean()),
            "std": float(result.posterior_draws.std()),
            "n": len(result.posterior_draws),
        },
    }


def backtest_bsts(
    prices: pd.Series,
    commodity: str = "eggs",
    start_year: int = 2010,
) -> dict:
    """Expanding-window backtest of the BSTS structural model.

    Evaluates January predictions (forecasting February) to match the
    Kalshi use case. At each January from start_year onward, fits BSTS
    on all prior data and predicts the following month's price.

    Args:
        prices: BLS price series with DatetimeIndex (monthly).
        commodity: Commodity key from SERIES_CONFIGS.
        start_year: First year to include in backtest.

    Returns:
        Dict with same schema as backtest(): predictions list,
        direction_accuracy, mape, coverage_90, coverage_50, brier_score.
    """
    from .bsts_forecaster import (
        build_almon_basis,
        build_covariate_matrix,
        fit_bsts,
        forecast_with_draws,
    )
    from .data import fetch_bls_series
    from .series_config import SERIES_CONFIGS

    config = SERIES_CONFIGS[commodity]

    # Fetch full target series
    target = fetch_bls_series(config.fred_id)
    target = target.sort_index()

    # Fetch and align covariates for the full period
    full_covariates = build_covariate_matrix(target, config)

    # Evaluate January predictions
    predictions = []
    jan_dates = [
        d for d in target.index
        if d.month == 1 and d.year >= start_year
    ]

    for jan_date in jan_dates:
        # Need February data to evaluate
        feb_candidates = target[
            (target.index.month == 2) &
            (target.index.year == jan_date.year)
        ]
        if feb_candidates.empty:
            continue

        actual_feb_price = float(feb_candidates.iloc[0])
        current_jan_price = float(target.loc[jan_date])

        # Train on all data up to and including this January
        train_target = target[target.index <= jan_date]
        if len(train_target) < 36:  # Need at least 3 years
            continue

        train_covariates = None
        if not full_covariates.empty:
            train_cov = full_covariates[full_covariates.index <= jan_date]
            if not train_cov.empty:
                train_covariates = train_cov

        try:
            bsts_result = fit_bsts(train_target, train_covariates, config)
            fc = forecast_with_draws(bsts_result, n_draws=2000, horizon=1)

            predicted_prob = fc.p_up
            predicted_price = fc.point_forecast
            q05, q95 = fc.intervals[90]
            q25, q75 = fc.intervals[50]
        except Exception:
            continue

        actual_up = actual_feb_price > current_jan_price

        predictions.append({
            "date": str(jan_date.date()),
            "year": int(jan_date.year),
            "month": 1,
            "predicted_prob": round(predicted_prob, 3),
            "predicted_price": round(predicted_price, 3),
            "actual_price": round(actual_feb_price, 3),
            "current_price": round(current_jan_price, 3),
            "actual_up": bool(actual_up),
            "correct": bool((predicted_prob > 0.5) == actual_up),
            "error": round(abs(predicted_price - actual_feb_price), 3),
            "in_90_interval": bool(q05 <= actual_feb_price <= q95),
            "in_50_interval": bool(q25 <= actual_feb_price <= q75),
        })

    if not predictions:
        return {
            "predictions": [],
            "direction_accuracy": 0,
            "mape": 0,
            "coverage_90": 0,
            "coverage_50": 0,
            "brier_score": 0,
            "accuracy": 0,
        }

    n = len(predictions)
    direction_accuracy = sum(p["correct"] for p in predictions) / n
    mape = sum(p["error"] / max(p["actual_price"], 0.01) for p in predictions) / n
    coverage_90 = sum(p["in_90_interval"] for p in predictions) / n
    coverage_50 = sum(p["in_50_interval"] for p in predictions) / n
    brier_score = sum(
        (p["predicted_prob"] - (1.0 if p["actual_up"] else 0.0)) ** 2
        for p in predictions
    ) / n

    return {
        "predictions": predictions,
        "direction_accuracy": round(direction_accuracy, 3),
        "mape": round(mape, 3),
        "coverage_90": round(coverage_90, 3),
        "coverage_50": round(coverage_50, 3),
        "brier_score": round(brier_score, 3),
        "accuracy": round(direction_accuracy, 3),
    }
