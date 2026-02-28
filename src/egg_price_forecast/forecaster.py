"""LightGBM multi-quantile regression forecaster for commodity prices.

Trains separate LightGBM models for each quantile to produce probabilistic
price forecasts. Uses cross-commodity features, lags, momentum, and
seasonality to predict the next month's price.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, stats

from egg_price_forecast.data import fetch_all_commodities, fetch_cpi_expectations


def build_forecast_features(
    commodities: dict[str, pd.Series],
    cpi: pd.Series,
    ppi_eggs: pd.Series | None = None,
) -> pd.DataFrame:
    """Build feature matrix from multi-commodity price data and CPI expectations.

    For each commodity and month, computes lag features, momentum indicators,
    seasonality encoding, CPI expectations, and cross-commodity momentum.

    Args:
        commodities: Dict mapping commodity name to monthly price Series.
        cpi: Monthly CPI expectation Series with DatetimeIndex.

    Returns:
        DataFrame with feature columns, target, commodity name, and date.
    """
    all_rows = []

    # Pre-compute monthly percentage changes for cross-commodity momentum
    mom_changes: dict[str, pd.Series] = {}
    for name, series in commodities.items():
        mom_changes[name] = series.pct_change()

    for name, series in commodities.items():
        series = series.sort_index()
        # Align CPI to commodity dates via forward-fill
        cpi_aligned = cpi.reindex(series.index, method="ffill")
        # Align PPI eggs to commodity dates via forward-fill
        if ppi_eggs is not None:
            ppi_aligned = ppi_eggs.reindex(series.index, method="ffill")
        else:
            ppi_aligned = pd.Series(np.nan, index=series.index)

        for i in range(12, len(series) - 1):
            date = series.index[i]
            month = date.month

            # Lag features (i >= 12, so all lookbacks are in-bounds)
            lags = {f"lag_{lag}": series.iloc[i - lag] for lag in [1, 2, 3, 4, 5, 6]}
            lags["lag_12"] = series.iloc[i - 12]

            # Percentage change features
            current = series.iloc[i]
            pct_1m = (
                (current - series.iloc[i - 1]) / series.iloc[i - 1]
                if series.iloc[i - 1] != 0
                else np.nan
            )
            pct_3m = (
                (current - series.iloc[i - 3]) / series.iloc[i - 3]
                if series.iloc[i - 3] != 0
                else np.nan
            )
            pct_12m = (
                (current - series.iloc[i - 12]) / series.iloc[i - 12]
                if series.iloc[i - 12] != 0
                else np.nan
            )

            # Seasonality encoding
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            # CPI expectation
            cpi_val = cpi_aligned.iloc[i] if i < len(cpi_aligned) else np.nan

            # Cross-commodity momentum: average MoM % change across OTHER
            # commodities for this date
            other_changes = []
            for other_name, changes in mom_changes.items():
                if other_name == name:
                    continue
                if date in changes.index and not pd.isna(changes.loc[date]):
                    other_changes.append(changes.loc[date])
            cross_momentum = (
                np.mean(other_changes) if other_changes else 0.0
            )

            row = {
                **lags,
                "pct_change_1m": pct_1m,
                "pct_change_3m": pct_3m,
                "pct_change_12m": pct_12m,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "cpi_expectation": cpi_val,
                "cross_commodity_momentum": cross_momentum,
                "is_eggs": 1 if name == "eggs" else 0,
                "price_level": current,
                "ppi_eggs": ppi_aligned.iloc[i] if i < len(ppi_aligned) else np.nan,
                "ppi_pct_change": (
                    (ppi_aligned.iloc[i] - ppi_aligned.iloc[i - 1]) / ppi_aligned.iloc[i - 1]
                    if (i >= 1 and i < len(ppi_aligned) and ppi_aligned.iloc[i - 1] != 0 and not pd.isna(ppi_aligned.iloc[i - 1]) and not pd.isna(ppi_aligned.iloc[i]))
                    else np.nan
                ),
                "wholesale_retail_gap": (
                    current / ppi_aligned.iloc[i]
                    if (i < len(ppi_aligned) and ppi_aligned.iloc[i] != 0 and not pd.isna(ppi_aligned.iloc[i]))
                    else np.nan
                ),
                "target": series.iloc[i + 1],
                "commodity": name,
                "date": date,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.dropna(
        subset=[
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
            "target",
        ]
    )
    return df.reset_index(drop=True)


FEATURE_COLS: list[str] = [
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
    "ppi_eggs",
    "ppi_pct_change",
    "wholesale_retail_gap",
]

_NON_FEATURE_COLS = frozenset({"target", "commodity", "date"})


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything except target, commodity, date)."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def train_quantile_models(
    df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
) -> dict[float, lgb.Booster]:
    """Train one LightGBM quantile regression model per quantile.

    Uses the last 20% of data (by time order) as the validation set for
    early stopping.

    Args:
        df: Feature DataFrame from build_forecast_features.
        quantiles: Tuple of quantile levels to model.

    Returns:
        Dict mapping each quantile float to its trained LGBMBooster.
    """
    feature_cols = _get_feature_columns(df)

    # Time-based train/val split (last 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    models: dict[float, lgb.Booster] = {}

    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks,
        )
        models[q] = booster

    return models


def predict_quantiles(
    models: dict[float, lgb.Booster],
    features: pd.DataFrame,
) -> dict[float, float]:
    """Predict price quantiles for a single observation.

    Applies a monotonic-crossing fix so that predicted quantile values
    are non-decreasing.

    Args:
        models: Dict mapping quantile level to trained LightGBM model.
        features: Single-row DataFrame of feature values.

    Returns:
        Dict mapping quantile level to predicted price.
    """
    feature_cols = _get_feature_columns(features)
    X = features[feature_cols].values

    raw_preds = {}
    for q, model in sorted(models.items()):
        pred = model.predict(X)[0]
        raw_preds[q] = float(pred)

    # Fix quantile crossing: enforce monotonicity
    sorted_qs = sorted(raw_preds.keys())
    sorted_vals = [raw_preds[q] for q in sorted_qs]

    # Make values monotonically non-decreasing
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] < sorted_vals[i - 1]:
            sorted_vals[i] = sorted_vals[i - 1]

    return dict(zip(sorted_qs, sorted_vals))


def sample_from_quantiles(
    quantile_preds: dict[float, float],
    n: int = 1000,
) -> np.ndarray:
    """Sample from a distribution fitted to the predicted quantiles.

    Fits a skew-normal distribution by minimizing the squared distance
    between predicted quantile values and the distribution's quantiles.

    Args:
        quantile_preds: Dict mapping quantile level to predicted value.
        n: Number of samples to draw.

    Returns:
        Array of n price samples, clipped to be non-negative.
    """
    qs = np.array(sorted(quantile_preds.keys()))
    vals = np.array([quantile_preds[q] for q in qs])

    median_val = float(np.median(vals))
    spread = float(np.ptp(vals)) or 1.0

    def objective(params: np.ndarray) -> float:
        loc, scale, skew = params
        if scale <= 0:
            return 1e10
        predicted = stats.skewnorm.ppf(qs, skew, loc=loc, scale=scale)
        return float(np.sum((predicted - vals) ** 2))

    result = optimize.minimize(
        objective,
        x0=[median_val, spread / 3.0, 0.0],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    loc, scale, skew = result.x
    scale = max(scale, 1e-6)

    samples = stats.skewnorm.rvs(skew, loc=loc, scale=scale, size=n)
    samples = np.clip(samples, 0, None)
    return samples


@dataclass
class ForecastResult:
    """Container for forecast output.

    Attributes:
        point_forecast: Median (q50) prediction.
        intervals: Prediction intervals, e.g. {90: (q05, q95), 50: (q25, q75)}.
        quantiles: Full quantile predictions mapping level to value.
        samples: Array of N simulated prices.
        p_up: Fraction of samples above the current price.
        current_price: Latest observed egg price.
        feature_importances: Feature name to importance score mapping.
    """

    point_forecast: float
    intervals: dict
    quantiles: dict[float, float]
    samples: np.ndarray
    p_up: float
    current_price: float
    feature_importances: dict[str, float]


def run_forecast(current_price: float | None = None) -> ForecastResult:
    """Run the full forecasting pipeline.

    Fetches data, builds features, trains quantile models, predicts for
    the most recent egg observation, and returns a ForecastResult.

    Args:
        current_price: Override for the current egg price. If None,
            uses the latest value from the data.

    Returns:
        ForecastResult with point forecast, intervals, samples, etc.
    """
    # Fetch data
    commodities = fetch_all_commodities()
    try:
        cpi = fetch_cpi_expectations()
    except Exception:
        # Fallback: empty CPI series
        cpi = pd.Series(dtype=float)

    try:
        from egg_price_forecast.data import fetch_ppi_eggs
        ppi_eggs = fetch_ppi_eggs()
    except Exception:
        ppi_eggs = None

    # Build features
    df = build_forecast_features(commodities, cpi, ppi_eggs=ppi_eggs)

    # Train models
    models = train_quantile_models(df)

    # Get the last egg row for prediction
    egg_df = df[df["commodity"] == "eggs"]
    if egg_df.empty:
        raise ValueError("No egg data found in feature matrix")

    last_row = egg_df.iloc[[-1]]

    # Current price
    if current_price is None:
        current_price = float(last_row["price_level"].iloc[0])

    # Predict quantiles
    quantile_preds = predict_quantiles(models, last_row)

    # Sample from distribution
    samples = sample_from_quantiles(quantile_preds, n=5000)

    # Compute p_up
    p_up = float(np.mean(samples > current_price))

    # Feature importances from median model
    median_model = models[0.50]
    feature_cols = _get_feature_columns(df)
    importances = median_model.feature_importance()
    feature_importances = dict(zip(feature_cols, importances.tolist()))

    return ForecastResult(
        point_forecast=quantile_preds[0.50],
        intervals={
            90: (quantile_preds[0.05], quantile_preds[0.95]),
            50: (quantile_preds[0.25], quantile_preds[0.75]),
        },
        quantiles=quantile_preds,
        samples=samples,
        p_up=p_up,
        current_price=current_price,
        feature_importances=feature_importances,
    )
