from __future__ import annotations

import numpy as np
import pytest

from custom_components.rain_incoming.radar.detector import TrackedCell
from custom_components.rain_incoming.radar.qc.speed_sanity import score_speed


def _make_cell(velocity_kmh: float) -> TrackedCell:
    return TrackedCell(lat=0.0, lon=0.0, velocity_kmh=velocity_kmh, bearing=90.0)


class TestScoreSpeed:
    def test_sane_speed_scores_one(self):
        """Cell at 50 km/h (within 3-130 range) should score 1.0."""
        cells = [_make_cell(50.0)]
        labeled = np.ones((16, 16), dtype=np.int32)
        score = score_speed(cells, labeled)
        assert score[8, 8] == pytest.approx(1.0)

    def test_too_fast_scores_low(self):
        """Cell at 200 km/h (above 130) should score 0.1."""
        cells = [_make_cell(200.0)]
        labeled = np.ones((16, 16), dtype=np.int32)
        score = score_speed(cells, labeled)
        assert score[8, 8] == pytest.approx(0.1)

    def test_stationary_scores_low(self):
        """Stationary cell (< 3 km/h) should score 0.2."""
        cells = [_make_cell(1.0)]
        labeled = np.ones((16, 16), dtype=np.int32)
        score = score_speed(cells, labeled)
        assert score[8, 8] == pytest.approx(0.2)

    def test_untracked_pixels_score_half(self):
        """Pixels not in any cell should score 0.5."""
        cells = [_make_cell(50.0)]
        labeled = np.zeros((16, 16), dtype=np.int32)  # no cells
        score = score_speed(cells, labeled)
        assert score[8, 8] == pytest.approx(0.5)

    def test_output_shape_matches_labeled(self):
        """Output shape must match labeled array."""
        cells = [_make_cell(50.0)]
        labeled = np.ones((20, 30), dtype=np.int32)
        score = score_speed(cells, labeled)
        assert score.shape == (20, 30)

    def test_values_between_zero_and_one(self):
        """All values must be in [0.0, 1.0]."""
        cells = [_make_cell(50.0), _make_cell(200.0), _make_cell(1.0)]
        labeled = np.zeros((16, 16), dtype=np.int32)
        labeled[0:5, :] = 1
        labeled[5:10, :] = 2
        labeled[10:15, :] = 3
        score = score_speed(cells, labeled)
        assert score.min() >= 0.0
        assert score.max() <= 1.0
