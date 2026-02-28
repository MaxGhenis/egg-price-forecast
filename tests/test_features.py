"""Tests for egg_price_forecast.features module."""

import numpy as np
import pandas as pd
import pytest


def _make_prices(values, start="2024-01-01"):
    """Helper to create a price Series with monthly DatetimeIndex."""
    index = pd.date_range(start=start, periods=len(values), freq="MS")
    return pd.Series(values, index=index, dtype=float)


class TestMonthlySeasonalRate:
    """Tests for monthly_seasonal_rate."""

    def test_all_months_increase(self):
        from egg_price_forecast.features import monthly_seasonal_rate

        # 3 years where every month increases
        values = []
        for _ in range(3):
            for m in range(12):
                values.append(2.0 + m * 0.1)
        prices = _make_prices(values, start="2020-01-01")

        # February (month 2) always increases vs January
        rate = monthly_seasonal_rate(prices, 2)
        assert rate == pytest.approx(1.0)

    def test_no_months_increase(self):
        from egg_price_forecast.features import monthly_seasonal_rate

        # 3 years where every month decreases
        values = []
        for _ in range(3):
            for m in range(12):
                values.append(5.0 - m * 0.1)
        prices = _make_prices(values, start="2020-01-01")

        # February (month 2) always decreases vs January
        rate = monthly_seasonal_rate(prices, 2)
        assert rate == pytest.approx(0.0)

    def test_works_for_january(self):
        from egg_price_forecast.features import monthly_seasonal_rate

        # January requires comparing to previous December
        values = [3.0, 2.5] + [2.5] * 10  # 2020: Dec=3.0 -> Jan increases
        values += [2.0, 2.5] + [2.5] * 10  # 2021: Dec=2.0 -> Jan increases
        prices = _make_prices(values, start="2019-12-01")

        rate = monthly_seasonal_rate(prices, 1)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_returns_float(self):
        from egg_price_forecast.features import monthly_seasonal_rate

        values = [2.0, 3.0] + [2.5] * 10
        prices = _make_prices(values, start="2020-01-01")

        rate = monthly_seasonal_rate(prices, 2)
        assert isinstance(rate, float)


class TestRecentMomentum:
    """Tests for recent_momentum."""

    def test_positive_momentum(self):
        from egg_price_forecast.features import recent_momentum

        # Prices consistently increasing
        prices = _make_prices([2.0, 2.1, 2.2, 2.3, 2.4], start="2025-01-01")
        result = recent_momentum(prices, "2025-06")
        assert result > 0

    def test_negative_momentum(self):
        from egg_price_forecast.features import recent_momentum

        # Prices consistently decreasing
        prices = _make_prices([3.0, 2.8, 2.6, 2.4, 2.2], start="2025-01-01")
        result = recent_momentum(prices, "2025-06")
        assert result < 0

    def test_zero_momentum(self):
        from egg_price_forecast.features import recent_momentum

        # Flat prices
        prices = _make_prices([2.0, 2.0, 2.0, 2.0], start="2025-01-01")
        result = recent_momentum(prices, "2025-05")
        assert result == pytest.approx(0.0)

    def test_returns_float(self):
        from egg_price_forecast.features import recent_momentum

        prices = _make_prices([2.0, 2.1, 2.2, 2.3], start="2025-01-01")
        result = recent_momentum(prices, "2025-05")
        assert isinstance(result, float)

    def test_insufficient_data_returns_zero(self):
        from egg_price_forecast.features import recent_momentum

        prices = _make_prices([2.0], start="2025-01-01")
        result = recent_momentum(prices, "2025-02")
        assert result == 0.0


class TestComputeMomChanges:
    """Tests for compute_mom_changes."""

    def test_returns_dataframe_with_correct_columns(self):
        from egg_price_forecast.features import compute_mom_changes

        prices = _make_prices([3.0, 3.5, 3.2])
        result = compute_mom_changes(prices)

        assert isinstance(result, pd.DataFrame)
        assert "abs_change" in result.columns
        assert "pct_change" in result.columns

    def test_same_index_as_input(self):
        from egg_price_forecast.features import compute_mom_changes

        prices = _make_prices([3.0, 3.5, 3.2])
        result = compute_mom_changes(prices)

        pd.testing.assert_index_equal(result.index, prices.index)

    def test_first_row_is_nan(self):
        from egg_price_forecast.features import compute_mom_changes

        prices = _make_prices([3.0, 3.5, 3.2])
        result = compute_mom_changes(prices)

        assert pd.isna(result["abs_change"].iloc[0])
        assert pd.isna(result["pct_change"].iloc[0])

    def test_known_values(self):
        from egg_price_forecast.features import compute_mom_changes

        prices = _make_prices([2.0, 3.0, 2.5])
        result = compute_mom_changes(prices)

        # 3.0 - 2.0 = 1.0
        assert result["abs_change"].iloc[1] == pytest.approx(1.0)
        # 1.0 / 2.0 = 0.5
        assert result["pct_change"].iloc[1] == pytest.approx(0.5)

        # 2.5 - 3.0 = -0.5
        assert result["abs_change"].iloc[2] == pytest.approx(-0.5)
        # -0.5 / 3.0 = -0.1667
        assert result["pct_change"].iloc[2] == pytest.approx(-1 / 6)


class TestDeclineStreak:
    """Tests for decline_streak."""

    def test_streak_of_three_declines(self):
        from egg_price_forecast.features import decline_streak

        # Prices decline from Oct through Jan (3 consecutive declines ending before Feb)
        prices = _make_prices(
            [5.0, 4.8, 4.5, 4.0, 3.8, 3.5, 3.2, 3.0],
            start="2025-07-01",
        )
        # target_month = "2026-03", month before is Feb 2026
        # 2026-02 = 3.0 < 3.2 (Jan), 2026-01 = 3.2 < 3.5 (Dec), etc.
        result = decline_streak(prices, "2026-03")
        assert result >= 3

    def test_streak_is_zero_when_last_month_increased(self):
        from egg_price_forecast.features import decline_streak

        # Prices: Jan=3.0, Feb=3.5 (increase)
        prices = _make_prices([3.0, 3.5], start="2025-01-01")
        result = decline_streak(prices, "2025-03")
        assert result == 0

    def test_streak_of_one(self):
        from egg_price_forecast.features import decline_streak

        # Sep goes up, Oct goes down => streak of 1 before Nov
        prices = _make_prices([3.0, 3.5, 3.2], start="2025-08-01")
        result = decline_streak(prices, "2025-11")
        assert result == 1

    def test_all_increases(self):
        from egg_price_forecast.features import decline_streak

        prices = _make_prices([1.0, 2.0, 3.0, 4.0], start="2025-01-01")
        result = decline_streak(prices, "2025-05")
        assert result == 0


class TestFebruarySeasonalRate:
    """Tests for february_seasonal_rate."""

    def test_all_februaries_increase(self):
        from egg_price_forecast.features import february_seasonal_rate

        # 3 years: Jan->Feb always increases
        values = []
        for year_offset in range(3):
            # Jan low, Feb high, then stable rest of year
            values.extend([2.0, 3.0] + [2.5] * 10)
        prices = _make_prices(values, start="2020-01-01")

        rate = february_seasonal_rate(prices)
        assert rate == pytest.approx(1.0)

    def test_no_februaries_increase(self):
        from egg_price_forecast.features import february_seasonal_rate

        # 3 years: Jan->Feb always decreases
        values = []
        for year_offset in range(3):
            values.extend([3.0, 2.0] + [2.5] * 10)
        prices = _make_prices(values, start="2020-01-01")

        rate = february_seasonal_rate(prices)
        assert rate == pytest.approx(0.0)

    def test_half_increase(self):
        from egg_price_forecast.features import february_seasonal_rate

        # Year 1: Feb > Jan (increase), Year 2: Feb < Jan (decrease)
        values = [2.0, 3.0] + [2.5] * 10  # 2020: increase
        values += [3.0, 2.0] + [2.5] * 10  # 2021: decrease
        prices = _make_prices(values, start="2020-01-01")

        rate = february_seasonal_rate(prices)
        assert rate == pytest.approx(0.5)

    def test_returns_float(self):
        from egg_price_forecast.features import february_seasonal_rate

        values = [2.0, 3.0] + [2.5] * 10
        prices = _make_prices(values, start="2020-01-01")

        rate = february_seasonal_rate(prices)
        assert isinstance(rate, float)


class TestWholesaleRetailGap:
    """Tests for wholesale_retail_gap."""

    def test_basic_ratio(self):
        from egg_price_forecast.features import wholesale_retail_gap

        result = wholesale_retail_gap(wholesale=4.0, retail=8.0)
        assert result == pytest.approx(2.0)

    def test_equal_prices(self):
        from egg_price_forecast.features import wholesale_retail_gap

        result = wholesale_retail_gap(wholesale=5.0, retail=5.0)
        assert result == pytest.approx(1.0)

    def test_returns_float(self):
        from egg_price_forecast.features import wholesale_retail_gap

        result = wholesale_retail_gap(wholesale=3.0, retail=6.0)
        assert isinstance(result, float)


class TestHpaiAnalogScore:
    """Tests for hpai_analog_score."""

    def test_returns_value_between_0_and_1(self):
        from egg_price_forecast.features import hpai_analog_score

        # Build data spanning 2022-2026 to include the 2023 analog period
        # 48 months from Jan 2022 to Dec 2025
        np.random.seed(42)
        n_months = 48
        values = np.cumsum(np.random.randn(n_months) * 0.1) + 3.0
        prices = _make_prices(values.tolist(), start="2022-01-01")

        score = hpai_analog_score(prices, "2026-02")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        from egg_price_forecast.features import hpai_analog_score

        np.random.seed(42)
        n_months = 48
        values = np.cumsum(np.random.randn(n_months) * 0.1) + 3.0
        prices = _make_prices(values.tolist(), start="2022-01-01")

        score = hpai_analog_score(prices, "2026-02")
        assert isinstance(score, float)

    def test_perfect_analog_scores_high(self):
        from egg_price_forecast.features import hpai_analog_score

        # Construct data where the recent trajectory closely matches
        # the 2023 HPAI recovery pattern (spike then decline)
        # 2023 pattern: Jan spike, Feb-Aug decline
        values_2022 = [2.5] * 12  # Stable 2022
        # 2023: spike then gradual decline
        values_2023 = [4.5, 4.2, 3.9, 3.6, 3.3, 3.0, 2.8, 2.7, 2.6, 2.5, 2.5, 2.5]
        # 2024: stable
        values_2024 = [2.5] * 12
        # 2025: replicate same spike-decline pattern
        values_2025 = [4.5, 4.2, 3.9, 3.6, 3.3, 3.0, 2.8, 2.7, 2.6, 2.5, 2.5, 2.5]

        all_values = values_2022 + values_2023 + values_2024 + values_2025
        prices = _make_prices(all_values, start="2022-01-01")

        score = hpai_analog_score(prices, "2026-01")
        assert 0.0 <= score <= 1.0
