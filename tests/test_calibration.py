"""Tests for egg_price_forecast.calibration module (TDD)."""

import pytest
from egg_price_forecast.calibration import calibrate_probability


class TestModelWeight:
    """Tests for model weight calculation from brier score."""

    def test_perfect_model_weight(self):
        """Brier score of 0 → weight capped at 0.9."""
        result = calibrate_probability(raw_p_up=0.7, base_rate=0.5, brier_score=0.0)
        assert result["model_weight"] == pytest.approx(0.9)

    def test_terrible_model_weight(self):
        """Brier score of 1 → weight clamped to 0.3."""
        result = calibrate_probability(raw_p_up=0.7, base_rate=0.5, brier_score=1.0)
        assert result["model_weight"] == pytest.approx(0.3)

    def test_medium_model_weight(self):
        """Brier score of 0.25 → w = 1 - 0.25 = 0.75 (within clamp range)."""
        result = calibrate_probability(raw_p_up=0.7, base_rate=0.5, brier_score=0.25)
        assert result["model_weight"] == pytest.approx(0.75)

    def test_weight_clamped_at_upper_bound(self):
        """Brier score below 0.1 → weight still capped at 0.9."""
        result = calibrate_probability(raw_p_up=0.7, base_rate=0.5, brier_score=0.05)
        assert result["model_weight"] == pytest.approx(0.9)

    def test_weight_clamped_at_lower_bound(self):
        """Brier score above 0.7 → weight still floored at 0.3."""
        result = calibrate_probability(raw_p_up=0.7, base_rate=0.5, brier_score=0.8)
        assert result["model_weight"] == pytest.approx(0.3)


class TestCalibratedProbability:
    """Tests for the calibrated_p_up output value."""

    def test_known_value(self):
        """Known-value check: raw=0.70, base=0.35, brier=0.325.

        w = 1 - 0.325 = 0.675
        calibrated = 0.675 * 0.70 + 0.325 * 0.35 = 0.4725 + 0.11375 = 0.58625
        """
        result = calibrate_probability(
            raw_p_up=0.70, base_rate=0.35, brier_score=0.325
        )
        assert result["calibrated_p_up"] == pytest.approx(0.58625)

    def test_perfect_model_trusts_raw_heavily(self):
        """Brier=0 → w=0.9, calibrated leans strongly toward raw_p_up."""
        raw = 0.8
        base = 0.4
        result = calibrate_probability(raw_p_up=raw, base_rate=base, brier_score=0.0)
        expected = 0.9 * raw + 0.1 * base
        assert result["calibrated_p_up"] == pytest.approx(expected)

    def test_terrible_model_trusts_base_rate_heavily(self):
        """Brier=1 → w=0.3, calibrated leans strongly toward base_rate."""
        raw = 0.8
        base = 0.4
        result = calibrate_probability(raw_p_up=raw, base_rate=base, brier_score=1.0)
        expected = 0.3 * raw + 0.7 * base
        assert result["calibrated_p_up"] == pytest.approx(expected)

    def test_calibrated_p_up_in_range(self):
        """calibrated_p_up is always in [0, 1] for typical inputs."""
        for raw, base, brier in [
            (0.0, 0.5, 0.25),
            (1.0, 0.5, 0.25),
            (0.5, 0.0, 0.25),
            (0.5, 1.0, 0.25),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.3, 0.6, 0.5),
        ]:
            result = calibrate_probability(
                raw_p_up=raw, base_rate=base, brier_score=brier
            )
            assert 0.0 <= result["calibrated_p_up"] <= 1.0, (
                f"calibrated_p_up={result['calibrated_p_up']} out of range "
                f"for raw={raw}, base={base}, brier={brier}"
            )


class TestEdgeCases:
    """Tests for boundary values of raw_p_up and base_rate."""

    def test_raw_p_up_zero(self):
        """raw_p_up=0.0 produces a valid, non-negative calibrated value."""
        result = calibrate_probability(raw_p_up=0.0, base_rate=0.5, brier_score=0.25)
        assert result["calibrated_p_up"] >= 0.0
        # w=0.75: 0.75*0.0 + 0.25*0.5 = 0.125
        assert result["calibrated_p_up"] == pytest.approx(0.125)

    def test_raw_p_up_one(self):
        """raw_p_up=1.0 produces a valid, at-most-1.0 calibrated value."""
        result = calibrate_probability(raw_p_up=1.0, base_rate=0.5, brier_score=0.25)
        assert result["calibrated_p_up"] <= 1.0
        # w=0.75: 0.75*1.0 + 0.25*0.5 = 0.875
        assert result["calibrated_p_up"] == pytest.approx(0.875)

    def test_base_rate_zero(self):
        """base_rate=0.0 is a valid input; calibrated stays non-negative."""
        result = calibrate_probability(raw_p_up=0.6, base_rate=0.0, brier_score=0.25)
        assert result["calibrated_p_up"] >= 0.0
        # w=0.75: 0.75*0.6 + 0.25*0.0 = 0.45
        assert result["calibrated_p_up"] == pytest.approx(0.45)

    def test_base_rate_one(self):
        """base_rate=1.0 is a valid input; calibrated stays at most 1.0."""
        result = calibrate_probability(raw_p_up=0.6, base_rate=1.0, brier_score=0.25)
        assert result["calibrated_p_up"] <= 1.0
        # w=0.75: 0.75*0.6 + 0.25*1.0 = 0.45 + 0.25 = 0.70
        assert result["calibrated_p_up"] == pytest.approx(0.70)


class TestReturnedKeys:
    """Tests for the structure of the returned dict."""

    def test_all_required_keys_present(self):
        """Return value contains all five expected keys."""
        result = calibrate_probability(raw_p_up=0.6, base_rate=0.5, brier_score=0.25)
        expected_keys = {
            "calibrated_p_up",
            "raw_p_up",
            "base_rate",
            "model_weight",
            "brier_score",
        }
        assert set(result.keys()) == expected_keys

    def test_passthrough_values_preserved(self):
        """raw_p_up, base_rate, and brier_score are echoed back unchanged."""
        raw, base, brier = 0.62, 0.41, 0.18
        result = calibrate_probability(
            raw_p_up=raw, base_rate=base, brier_score=brier
        )
        assert result["raw_p_up"] == pytest.approx(raw)
        assert result["base_rate"] == pytest.approx(base)
        assert result["brier_score"] == pytest.approx(brier)
