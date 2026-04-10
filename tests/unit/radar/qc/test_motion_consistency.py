from __future__ import annotations

import numpy as np
import pytest

from custom_components.rain_incoming.radar.detector import TrackedCell
from custom_components.rain_incoming.radar.qc.motion_consistency import score_motion_consistency


def _make_cell(velocity_kmh: float = 50.0, coherent: bool = True) -> TrackedCell:
    return TrackedCell(
        lat=0.0, lon=0.0,
        velocity_kmh=velocity_kmh, bearing=90.0,
        confidence=1.0 if coherent else 0.3,
    )


class TestScoreMotionConsistency:
    def test_coherent_cell_scores_one(self):
        """A cell that passed directional coherence should score 1.0."""
        # Coherent cells have confidence >= 0.4 (set by detector)
        cells = [_make_cell(coherent=True)]
        labeled = np.ones((16, 16), dtype=np.int32)
        score = score_motion_consistency(cells, labeled)
        assert score[8, 8] == pytest.approx(1.0)

    def test_non_coherent_cell_scores_low(self):
        """A cell that failed directional coherence should score 0.3."""
        cells = [_make_cell(coherent=False)]
        labeled = np.ones((16, 16), dtype=np.int32)
        score = score_motion_consistency(cells, labeled)
        assert score[8, 8] == pytest.approx(0.3)

    def test_untracked_pixels_score_half(self):
        """Untracked echo pixels should score 0.5."""
        cells = [_make_cell()]
        labeled = np.zeros((16, 16), dtype=np.int32)
        score = score_motion_consistency(cells, labeled)
        assert score[8, 8] == pytest.approx(0.5)

    def test_output_shape_matches_labeled(self):
        """Output shape must match labeled array."""
        cells = [_make_cell()]
        labeled = np.ones((20, 30), dtype=np.int32)
        score = score_motion_consistency(cells, labeled)
        assert score.shape == (20, 30)

    def test_values_between_zero_and_one(self):
        """All values must be in [0.0, 1.0]."""
        cells = [_make_cell(coherent=True), _make_cell(coherent=False)]
        labeled = np.zeros((16, 16), dtype=np.int32)
        labeled[0:8, :] = 1
        labeled[8:16, :] = 2
        score = score_motion_consistency(cells, labeled)
        assert score.min() >= 0.0
        assert score.max() <= 1.0
