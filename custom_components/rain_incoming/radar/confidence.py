from __future__ import annotations


def pop_to_threshold_multiplier(pop_pct: float | None) -> float:
    """Map forecast PoP percentage to confidence threshold multiplier.

    Higher multiplier = higher bar (harder to trigger).
    None (missing forecast) returns 1.0 (no penalty).
    """
    if pop_pct is None:
        return 1.0
    if pop_pct < 5:
        return 3.0
    if pop_pct < 30:
        return 2.0
    return 1.0
