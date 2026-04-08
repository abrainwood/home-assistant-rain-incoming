from __future__ import annotations

import numpy as np
import pytest

from custom_components.incoming_rain.radar.qc.texture import score_texture


class TestScoreTexture:
    def test_smooth_gradient_high_confidence(self):
        """A uniform block of echo (like real rain) should get high confidence."""
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[20:44, 20:44] = 0.5  # uniform block
        score = score_texture(grid)
        # Interior of smooth area should have high confidence
        assert score[32, 32] > 0.8

    def test_speckle_low_confidence(self):
        """Salt-and-pepper noise (like AP/clutter) should get low confidence."""
        rng = np.random.default_rng(42)
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[rng.random((64, 64)) > 0.7] = 0.3  # sparse random echoes
        score = score_texture(grid)
        # Speckly area should have low mean confidence
        echo_pixels = grid > 0
        mean_confidence = score[echo_pixels].mean()
        assert mean_confidence < 0.5

    def test_no_echo_zero_confidence(self):
        """No-echo pixels (grid == 0) should always get 0.0 confidence."""
        grid = np.zeros((64, 64), dtype=np.float32)
        score = score_texture(grid)
        assert (score == 0.0).all()

    def test_uniform_nonzero_high_confidence(self):
        """Perfectly uniform echo = zero std dev = max confidence."""
        grid = np.full((64, 64), 0.3, dtype=np.float32)
        score = score_texture(grid)
        assert score.mean() > 0.9

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        grid = np.random.default_rng(0).random((100, 80)).astype(np.float32)
        score = score_texture(grid)
        assert score.shape == grid.shape

    def test_values_between_zero_and_one(self):
        """All output values must be in [0.0, 1.0]."""
        grid = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        score = score_texture(grid)
        assert score.min() >= 0.0
        assert score.max() <= 1.0

    def test_custom_thresholds(self):
        """Custom low/high thresholds should be respected."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        score = score_texture(grid, low_threshold=0.0, high_threshold=0.01)
        # Uniform block with very tight thresholds - should still be high confidence
        assert score[16, 16] > 0.9

    def test_custom_kernel_size(self):
        """Different kernel sizes should produce valid results."""
        grid = np.random.default_rng(7).random((64, 64)).astype(np.float32)
        score_small = score_texture(grid, kernel_size=3)
        score_large = score_texture(grid, kernel_size=9)
        # Both should be valid
        assert score_small.shape == grid.shape
        assert score_large.shape == grid.shape
        # Larger kernel should produce smoother (higher mean) confidence
        # because it averages over more pixels
        echo_mask = grid > 0
        assert score_large[echo_mask].mean() >= score_small[echo_mask].mean() - 0.1
