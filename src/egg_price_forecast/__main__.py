"""CLI entrypoint: prints probability estimate and Kalshi edge."""

import argparse

from egg_price_forecast.data import fetch_bls_eggs


def main():
    parser = argparse.ArgumentParser(description="Egg price forecast")
    parser.add_argument(
        "--model",
        choices=["bsts", "lgbm"],
        default="bsts",
        help="Model type: bsts (structural time series) or lgbm (LightGBM quantile)",
    )
    args = parser.parse_args()

    print("Fetching BLS egg price data...")
    prices = fetch_bls_eggs()
    print(f"  Got {len(prices)} monthly observations, latest: {prices.index[-1].strftime('%Y-%m')} = ${prices.iloc[-1]:.3f}")

    model_label = "BSTS structural" if args.model == "bsts" else "LightGBM quantile"
    print(f"\nEstimating probability of Feb 2026 increase ({model_label})...")

    if args.model == "bsts":
        from egg_price_forecast.model import estimate_probability_bsts
        result = estimate_probability_bsts(prices)
    else:
        from egg_price_forecast.model import estimate_probability
        result = estimate_probability(prices)

    prob = result["probability"]
    point = result["point_forecast"]
    intervals = result["intervals"]

    print(f"\n{'='*50}")
    print(f"P(Feb 2026 increase): {prob:.1%}")
    print(f"Point forecast:       ${point:.2f}")
    if "90" in intervals:
        i90 = intervals["90"]
        print(f"90% interval:         [${i90['low']:.2f}, ${i90['high']:.2f}]")
    print(f"\nTop components:")
    for name, value in result["components"].items():
        print(f"  {name}: {value:.4f}")

    # Kalshi edge calculation
    market_yes_price = 0.31  # Current YES at 31c
    edge = prob - market_yes_price
    kelly = edge / (1 - market_yes_price) if edge > 0 else 0

    print(f"\n{'='*50}")
    print(f"Kalshi KXEGGS-FEB26-0:")
    print(f"  Market YES price: ${market_yes_price:.2f}")
    print(f"  Model estimate:   {prob:.1%}")
    print(f"  Edge:             {edge:+.1%}")
    print(f"  Kelly fraction:   {kelly:.1%}")

    if edge > 0.05:
        print(f"\n  -> BUY YES (positive edge)")
    elif edge < -0.05:
        print(f"\n  -> BUY NO (negative edge)")
    else:
        print(f"\n  -> No clear edge, stay out")


if __name__ == "__main__":
    main()
