from __future__ import annotations

from ..const import SATELLITE_CLOUD_PRESENCE_FRACTION


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


def cloud_to_threshold_multiplier(cloud_fraction: float | None) -> float:
    """Map satellite cloud fraction to confidence threshold multiplier.

    None (satellite unavailable) returns 1.0 (fail open).
    Clear sky (fraction below the cloud-presence threshold) raises the bar
    to 3.0 - any radar return there is likely noise.
    """
    if cloud_fraction is None:
        return 1.0
    if cloud_fraction < SATELLITE_CLOUD_PRESENCE_FRACTION:
        return 3.0
    return 1.0
