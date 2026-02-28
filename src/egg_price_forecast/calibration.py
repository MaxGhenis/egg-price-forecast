"""Probability calibration for egg price forecasts.

Blends the raw model probability with the historical base rate,
weighted by model quality (Brier score).
"""


def calibrate_probability(
    raw_p_up: float,
    base_rate: float,
    brier_score: float,
) -> dict:
    """Calibrate raw probability by blending with historical base rate.

    Uses a weighted average where the model weight is derived from the
    Brier score: better-calibrated models get more weight.

    w = clip(1 - brier_score, 0.3, 0.9)
    calibrated_p_up = w * raw_p_up + (1 - w) * base_rate

    Args:
        raw_p_up: Raw P(up) from the simulation model.
        base_rate: Historical rate of Feb > Jan price increases.
        brier_score: Brier score from backtest (0=perfect, 1=worst).

    Returns:
        Dict with keys: calibrated_p_up, raw_p_up, base_rate,
        model_weight, brier_score.
    """
    w = max(0.3, min(0.9, 1.0 - brier_score))
    calibrated_p_up = w * raw_p_up + (1.0 - w) * base_rate

    return {
        "calibrated_p_up": calibrated_p_up,
        "raw_p_up": raw_p_up,
        "base_rate": base_rate,
        "model_weight": w,
        "brier_score": brier_score,
    }
