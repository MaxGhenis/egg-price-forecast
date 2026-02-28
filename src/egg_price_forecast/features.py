"""Feature engineering for egg price forecasting.

Computes features used by the probability model: month-over-month changes,
decline streaks, HPAI analog scoring, seasonal patterns, and wholesale-retail gaps.
"""

import numpy as np
import pandas as pd


def compute_mom_changes(prices: pd.Series) -> pd.DataFrame:
    """Compute month-over-month absolute and percentage changes.

    Args:
        prices: Series with DatetimeIndex of monthly prices.

    Returns:
        DataFrame with columns 'abs_change' and 'pct_change',
        same DatetimeIndex as input. First row is NaN.
    """
    abs_change = prices.diff()
    pct_change = prices.pct_change()

    return pd.DataFrame(
        {"abs_change": abs_change, "pct_change": pct_change},
        index=prices.index,
    )


def decline_streak(prices: pd.Series, target_month: str) -> int:
    """Count consecutive months of price decline ending before target_month.

    Args:
        prices: Series with DatetimeIndex of monthly prices.
        target_month: Format "YYYY-MM". Count streak ending at the month
            immediately before this.

    Returns:
        Number of consecutive declining months. 0 if the most recent
        month before target saw an increase.
    """
    target = pd.Timestamp(target_month + "-01")

    # Filter to months before target
    before_target = prices[prices.index < target].sort_index()

    if len(before_target) < 2:
        return 0

    # Compute month-over-month changes
    changes = before_target.diff()

    # Count consecutive declines from the end
    streak = 0
    for val in reversed(changes.values):
        if pd.isna(val):
            break
        if val < 0:
            streak += 1
        else:
            break

    return streak


def hpai_analog_score(prices: pd.Series, target_month: str) -> float:
    """Compare current price trajectory to 2023 HPAI recovery pattern.

    The 2023 pattern: prices spiked in Jan 2023 then declined Feb-Aug 2023.
    This function computes the correlation between recent 3-month price
    changes and the corresponding changes during the 2023 recovery.

    Args:
        prices: Series with DatetimeIndex spanning at least 2023.
        target_month: Format "YYYY-MM".

    Returns:
        Score in [0, 1] representing similarity to the 2023 recovery.
        0 = no similarity, 1 = perfect match.
    """
    target = pd.Timestamp(target_month + "-01")
    changes = prices.pct_change().dropna()

    # 2023 analog period: Feb-Apr 2023 (3 months of the recovery decline)
    analog_start = pd.Timestamp("2023-02-01")
    analog_end = pd.Timestamp("2023-05-01")
    analog_changes = changes[
        (changes.index >= analog_start) & (changes.index < analog_end)
    ]

    if len(analog_changes) < 3:
        return 0.5  # Neutral if insufficient data

    # Recent 3 months of changes ending before target
    recent_changes = changes[changes.index < target].tail(3)

    if len(recent_changes) < 3:
        return 0.5

    # Correlation between recent and analog trajectories
    corr = np.corrcoef(recent_changes.values, analog_changes.values[:3])[0, 1]

    if np.isnan(corr):
        return 0.5

    # Map correlation from [-1, 1] to [0, 1]
    score = (corr + 1.0) / 2.0
    return float(np.clip(score, 0.0, 1.0))


def monthly_seasonal_rate(prices: pd.Series, month: int) -> float:
    """Historical rate of price increases for a given calendar month.

    For each year in the data, checks whether the price in the target month
    was higher than the preceding month.

    Args:
        prices: Series with DatetimeIndex of monthly prices.
        month: Calendar month (1-12).

    Returns:
        Float in [0, 1]: fraction of years where price increased in that month.
    """
    increases = 0
    total = 0

    prev_month = month - 1 if month > 1 else 12

    for year in prices.index.year.unique():
        prev_year = year if month > 1 else year - 1
        prev_mask = (prices.index.year == prev_year) & (
            prices.index.month == prev_month
        )
        curr_mask = (prices.index.year == year) & (prices.index.month == month)

        prev_prices = prices[prev_mask]
        curr_prices = prices[curr_mask]

        if len(prev_prices) == 1 and len(curr_prices) == 1:
            total += 1
            if curr_prices.iloc[0] > prev_prices.iloc[0]:
                increases += 1

    if total == 0:
        return 0.0

    return float(increases / total)


def recent_momentum(prices: pd.Series, target_month: str, window: int = 3) -> float:
    """Average month-over-month percentage change over recent months.

    Args:
        prices: Series with DatetimeIndex of monthly prices.
        target_month: Format "YYYY-MM". Computes momentum from the months
            immediately before this.
        window: Number of months to average over.

    Returns:
        Average MoM percentage change (e.g. -0.05 for -5%).
    """
    target = pd.Timestamp(target_month + "-01")
    before_target = prices[prices.index < target].sort_index()

    if len(before_target) < window + 1:
        return 0.0

    pct_changes = before_target.pct_change().dropna()
    recent = pct_changes.tail(window)

    if len(recent) == 0:
        return 0.0

    return float(recent.mean())


def february_seasonal_rate(prices: pd.Series) -> float:
    """Historical rate of February price increases (Feb > Jan).

    Looks at all years in the data where both January and February
    prices exist, and computes the fraction where February was higher.

    Args:
        prices: Series with DatetimeIndex of monthly prices.

    Returns:
        Float in [0, 1]: fraction of Februaries with price increase.
    """
    increases = 0
    total = 0

    # Group by year and check Jan -> Feb
    for year in prices.index.year.unique():
        jan_mask = (prices.index.year == year) & (prices.index.month == 1)
        feb_mask = (prices.index.year == year) & (prices.index.month == 2)

        jan_prices = prices[jan_mask]
        feb_prices = prices[feb_mask]

        if len(jan_prices) == 1 and len(feb_prices) == 1:
            total += 1
            if feb_prices.iloc[0] > jan_prices.iloc[0]:
                increases += 1

    if total == 0:
        return 0.0

    return float(increases / total)


def wholesale_retail_gap(wholesale: float, retail: float) -> float:
    """Compute the ratio of retail to wholesale price.

    A higher ratio indicates more room for retail prices to increase
    (or that wholesale hasn't yet passed through to retail).

    Args:
        wholesale: Wholesale price in $/dozen.
        retail: Retail price in $/dozen.

    Returns:
        Float ratio: retail / wholesale.
    """
    return float(retail / wholesale)
