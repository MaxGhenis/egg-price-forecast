"""Export model data as JSON for the visualization frontend."""

import json
from pathlib import Path

from egg_price_forecast.data import fetch_bls_eggs
from egg_price_forecast.features import (
    compute_mom_changes,
    decline_streak,
    monthly_seasonal_rate,
    recent_momentum,
)
from egg_price_forecast.model import backtest, estimate_probability


def export():
    prices = fetch_bls_eggs()

    result = estimate_probability(prices)
    bt = backtest(prices, start_year=2000)

    latest_price = round(float(prices.iloc[-1]), 3)

    # Price history — last 10 years for the chart, full for context
    recent = prices[prices.index >= "2015-01-01"]
    price_history = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(v), 3)}
        for d, v in recent.items()
    ]

    # Full monthly data for sparkline
    all_prices = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(v), 3)}
        for d, v in prices.items()
    ]

    # MoM changes for recent period
    mom = compute_mom_changes(prices)
    recent_mom = mom[mom.index >= "2020-01-01"].dropna()
    mom_data = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "abs_change": round(float(row["abs_change"]), 3),
            "pct_change": round(float(row["pct_change"]) * 100, 1),
        }
        for d, row in recent_mom.iterrows()
    ]

    # February history — every Feb vs Jan
    feb_history = []
    for year in sorted(prices.index.year.unique()):
        jan = prices[(prices.index.year == year) & (prices.index.month == 1)]
        feb = prices[(prices.index.year == year) & (prices.index.month == 2)]
        if len(jan) == 1 and len(feb) == 1:
            jan_p = float(jan.iloc[0])
            feb_p = float(feb.iloc[0])
            feb_history.append(
                {
                    "year": int(year),
                    "jan_price": round(jan_p, 3),
                    "feb_price": round(feb_p, 3),
                    "change": round(feb_p - jan_p, 3),
                    "pct_change": round((feb_p - jan_p) / jan_p * 100, 1),
                    "increased": feb_p > jan_p,
                }
            )

    # Kalshi edge
    market_yes = 0.31
    edge = result["probability"] - market_yes
    kelly = edge / (1 - market_yes) if edge > 0 else 0

    data = {
        "generated_at": prices.index[-1].strftime("%Y-%m-%d"),
        "latest_price": latest_price,
        "latest_month": prices.index[-1].strftime("%Y-%m"),
        "total_observations": len(prices),
        "forecast": {
            "probability": round(result["probability"], 4),
            "components": {
                k: round(v, 4) for k, v in result["components"].items()
            },
        },
        "price_forecast": {
            "point_forecast": result["point_forecast"],
            "current_price": latest_price,
            "intervals": result["intervals"],
            "quantiles": result["quantiles"],
            "samples_summary": result["samples_summary"],
        },
        "model_info": {
            "type": result["model_info"]["type"],
            "n_quantile_models": result["model_info"]["n_quantile_models"],
            "n_samples": result["model_info"]["n_samples"],
            "feature_importances": {
                k: round(v, 4)
                for k, v in result["model_info"]["feature_importances"].items()
            },
        },
        "kalshi": {
            "ticker": "KXEGGS-FEB26-0",
            "market_yes_price": market_yes,
            "model_estimate": round(result["probability"], 4),
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly, 4),
            "recommendation": (
                "BUY YES"
                if edge > 0.05
                else ("BUY NO" if edge < -0.05 else "NO EDGE")
            ),
        },
        "backtest": {
            "accuracy": round(bt["direction_accuracy"], 4),
            "direction_accuracy": round(bt["direction_accuracy"], 4),
            "mape": round(bt["mape"], 3),
            "coverage_90": round(bt["coverage_90"], 3),
            "coverage_50": round(bt["coverage_50"], 3),
            "brier_score": round(bt["brier_score"], 4),
            "predictions": bt["predictions"],
        },
        "price_history": price_history,
        "all_prices": all_prices,
        "mom_changes": mom_data,
        "feb_history": feb_history,
        "decline_streak": decline_streak(prices, "2026-02"),
        "seasonal_rate": round(monthly_seasonal_rate(prices, 2), 4),
        "recent_momentum": round(recent_momentum(prices, "2026-02"), 4),
    }

    out = Path(__file__).parent / "viz" / "src" / "data" / "forecast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Exported to {out}")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"  Point forecast: ${result['point_forecast']:.2f}")
    print(f"  Edge: {edge:+.1%}")


if __name__ == "__main__":
    export()
