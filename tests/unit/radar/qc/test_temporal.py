from __future__ import annotations

import numpy as np
import pytest

from custom_components.incoming_rain.radar.qc.temporal import score_temporal


class TestScoreTemporal:
    def test_pixel_in_all_frames_scores_one(self):
        """Pixel present in all 6 frames should score 1.0."""
        grids = [np.full((16, 16), 0.5, dtype=np.float32) for _ in range(6)]
        score = score_temporal(grids, threshold=0.01, min_frames=3)
        # Present in all 6 frames, 6/3 = 2.0 clipped to 1.0
        assert score[8, 8] == pytest.approx(1.0)

    def test_pixel_in_one_of_six_frames_scores_low(self):
        """Pixel in only 1 of 6 frames -> score ~0.33."""
        grids = [np.zeros((16, 16), dtype=np.float32) for _ in range(6)]
        # Only in the latest frame
        grids[-1][8, 8] = 0.5
        score = score_temporal(grids, threshold=0.01, min_frames=3)
        assert score[8, 8] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_pixel_in_three_of_six_frames_scores_one(self):
        """Pixel in 3 of 6 frames -> score 1.0 (meets min_frames=3)."""
        grids = [np.zeros((16, 16), dtype=np.float32) for _ in range(6)]
        grids[-1][8, 8] = 0.5
        grids[-2][8, 8] = 0.5
        grids[-3][8, 8] = 0.5
        score = score_temporal(grids, threshold=0.01, min_frames=3)
        assert score[8, 8] == pytest.approx(1.0)

    def test_pixel_not_in_latest_frame_scores_zero(self):
        """Pixel not in the latest frame should score 0.0."""
        grids = [np.full((16, 16), 0.5, dtype=np.float32) for _ in range(6)]
        grids[-1] = np.zeros((16, 16), dtype=np.float32)
        score = score_temporal(grids, threshold=0.01, min_frames=3)
        assert score[8, 8] == 0.0

    def test_empty_grid_list_no_crash(self):
        """Empty grid list should return empty array without crashing."""
        score = score_temporal([], threshold=0.01, min_frames=3)
        assert isinstance(score, np.ndarray)
        assert score.size == 0

    def test_output_shape_matches_grid(self):
        """Output shape should match input grid shape."""
        grids = [np.full((20, 30), 0.5, dtype=np.float32) for _ in range(4)]
        score = score_temporal(grids)
        assert score.shape == (20, 30)

    def test_values_between_zero_and_one(self):
        """All output values must be in [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        grids = [rng.random((16, 16)).astype(np.float32) * 0.5 for _ in range(5)]
        score = score_temporal(grids)
        assert score.min() >= 0.0
        assert score.max() <= 1.0
