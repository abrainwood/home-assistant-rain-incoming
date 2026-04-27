from __future__ import annotations

import pytest

from custom_components.rain_incoming.const import (
    SATELLITE_CLOUD_PRESENCE_FRACTION,
)
from custom_components.rain_incoming.radar.confidence import (
    cloud_to_threshold_multiplier,
    pop_to_threshold_multiplier,
)


class TestPopToThresholdMultiplier:
    @pytest.mark.parametrize(
        "pop_pct, expected",
        [
            (None, 1.0),    # missing forecast → no adjustment
            (0.0, 3.0),     # 0 <= pop < 5 → raise bar significantly
            (4.99, 3.0),    # just below 5 boundary → still 3.0
            (5.0, 2.0),     # at 5 boundary → drop to 2.0
            (29.99, 2.0),   # just below 30 boundary → still 2.0
            (30.0, 1.0),    # at 30 boundary → no adjustment
            (100.0, 1.0),   # high PoP → no adjustment (strong forecast confirms radar)
        ],
    )
    def test_mapping(self, pop_pct, expected):
        """Boundary-correct mapping from PoP percentage to threshold multiplier."""
        assert pop_to_threshold_multiplier(pop_pct) == expected


class TestCloudToThresholdMultiplier:
    """Maps satellite cloud-fraction to a detection threshold multiplier.

    Mirrors `pop_to_threshold_multiplier`'s asymmetric semantics: when the
    satellite signal is missing (None), fail open (1.0). When the sky is
    clear, raise the bar (3.0) - any radar return is likely noise. When
    clouds are present, trust the radar normally (1.0).
    """

    @pytest.mark.parametrize(
        "cloud_fraction, expected",
        [
            (None, 1.0),                                          # satellite unavailable: fail open
            (0.0, 3.0),                                           # totally clear: raise bar
            (SATELLITE_CLOUD_PRESENCE_FRACTION - 0.01, 3.0),      # just below threshold: still raise bar
            (SATELLITE_CLOUD_PRESENCE_FRACTION, 1.0),             # at boundary: clouds present
            (SATELLITE_CLOUD_PRESENCE_FRACTION + 0.01, 1.0),      # just above boundary: clouds present
            (1.0, 1.0),                                           # totally cloudy: trust radar
        ],
    )
    def test_mapping(self, cloud_fraction, expected):
        """Boundary-correct mapping from cloud fraction to threshold multiplier.

        Boundary case (== threshold) maps to 1.0 because the predicate is
        "clouds present" (>=), not "more than threshold" (>)."""
        assert cloud_to_threshold_multiplier(cloud_fraction) == expected
