"""CLI entrypoint: prints probability estimate and Kalshi edge."""

from egg_price_forecast.data import fetch_bls_eggs, get_wholesale_prices
from egg_price_forecast.model import estimate_probability


def main():
    print("Fetching BLS egg price data...")
    prices = fetch_bls_eggs()
    print(f"  Got {len(prices)} monthly observations, latest: {prices.index[-1].strftime('%Y-%m')} = ${prices.iloc[-1]:.3f}")

    print("\nFetching wholesale prices...")
    wholesale = get_wholesale_prices()
    if wholesale is not None:
        print(f"  Wholesale price: ${wholesale:.2f}/dozen")
    else:
        print("  Could not fetch wholesale data (will proceed without)")

    print("\nEstimating probability of Feb 2026 increase...")
    result = estimate_probability(prices, wholesale=wholesale)

    print(f"\n{'='*50}")
    print(f"P(Feb 2026 increase): {result['probability']:.1%}")
    print(f"Confidence interval:  [{result['ci_low']:.1%}, {result['ci_high']:.1%}]")
    print(f"\nComponent breakdown:")
    for name, value in result["components"].items():
        print(f"  {name}: {value:.3f}")

    # Kalshi edge calculation
    market_yes_price = 0.31  # Current YES at 31c
    edge = result["probability"] - market_yes_price
    kelly = edge / (1 - market_yes_price) if edge > 0 else 0

    print(f"\n{'='*50}")
    print(f"Kalshi KXEGGS-FEB26-0:")
    print(f"  Market YES price: ${market_yes_price:.2f}")
    print(f"  Model estimate:   {result['probability']:.1%}")
    print(f"  Edge:             {edge:+.1%}")
    print(f"  Kelly fraction:   {kelly:.1%}")

    if edge > 0.05:
        print(f"\n  → BUY YES (positive edge)")
    elif edge < -0.05:
        print(f"\n  → BUY NO (negative edge)")
    else:
        print(f"\n  → No clear edge, stay out")


if __name__ == "__main__":
    main()
