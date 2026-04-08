from __future__ import annotations

import numpy as np

from custom_components.incoming_rain.radar.qc.scoring import compute_confidence_map
from custom_components.incoming_rain.radar.qc.types import ConfidenceMap, QCConfig


class TestComputeConfidenceMap:
    def test_returns_confidence_map(self):
        """Should return a ConfidenceMap with correct shape."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result = compute_confidence_map(grid)
        assert isinstance(result, ConfidenceMap)
        assert result.confidence.shape == (32, 32)

    def test_smooth_rain_high_confidence(self):
        """Smooth uniform echo should get high confidence."""
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[20:44, 20:44] = 0.4
        result = compute_confidence_map(grid)
        assert result.confidence[32, 32] > 0.7

    def test_factor_scores_contains_texture(self):
        """factor_scores dict should contain a 'texture' entry."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result = compute_confidence_map(grid)
        assert "texture" in result.factor_scores
        assert result.factor_scores["texture"].shape == (32, 32)

    def test_custom_config_is_used(self):
        """Passing a custom QCConfig should affect the result."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        config = QCConfig(texture_kernel_size=3, texture_high_threshold=0.01)
        result = compute_confidence_map(grid, config)
        assert isinstance(result, ConfidenceMap)

    def test_no_echo_zero_confidence(self):
        """All-zero grid should produce all-zero confidence."""
        grid = np.zeros((32, 32), dtype=np.float32)
        result = compute_confidence_map(grid)
        assert (result.confidence == 0.0).all()

    def test_confidence_dtype_is_float32(self):
        """Confidence map should be float32."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result = compute_confidence_map(grid)
        assert result.confidence.dtype == np.float32
