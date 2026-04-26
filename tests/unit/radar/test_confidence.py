from __future__ import annotations

import pytest

from custom_components.rain_incoming.radar.confidence import pop_to_threshold_multiplier


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
