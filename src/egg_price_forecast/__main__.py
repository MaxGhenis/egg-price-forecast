"""CLI entrypoint: prints probability estimate and Kalshi edge."""

from egg_price_forecast.data import fetch_bls_eggs
from egg_price_forecast.model import estimate_probability


def main():
    print("Fetching BLS egg price data...")
    prices = fetch_bls_eggs()
    print(f"  Got {len(prices)} monthly observations, latest: {prices.index[-1].strftime('%Y-%m')} = ${prices.iloc[-1]:.3f}")

    print("\nEstimating probability of Feb 2026 increase...")
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
    print(f"\nTop feature importances:")
    for name, value in result["components"].items():
        print(f"  {name}: {value:.1f}")

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
