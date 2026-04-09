from __future__ import annotations

import logging

import numpy as np

from custom_components.incoming_rain.radar.qc.scoring import compute_confidence_map
from custom_components.incoming_rain.radar.qc.types import ConfidenceMap, QCConfig


class TestModelPrecipitationModifier:
    """Asymmetric model treatment: 'confirms rain' is a strong boost,
    'says dry' is a mild skepticism. Radar might see virga or cells
    the model hasn't picked up yet."""

    def test_model_dry_applies_mild_penalty(self):
        """0.0mm -> confidence * 0.85 (mild skepticism, not a veto)."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_normal = compute_confidence_map(grid)
        result_dry = compute_confidence_map(grid, model_precipitation_mm=0.0)
        mask = result_normal.confidence > 0
        ratio = result_dry.confidence[mask] / result_normal.confidence[mask]
        np.testing.assert_allclose(ratio, 0.85, atol=0.01)

    def test_model_trace_no_change(self):
        """0.3mm (trace) -> no change either way."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_normal = compute_confidence_map(grid)
        result_trace = compute_confidence_map(grid, model_precipitation_mm=0.3)
        np.testing.assert_array_almost_equal(result_trace.confidence, result_normal.confidence)

    def test_model_confirms_rain_boosts_confidence(self):
        """1.0mm -> confidence * 1.15 (model confirms, boost)."""
        # Use noisy grid so base confidence is well below 1.0 (avoids cap)
        rng = np.random.default_rng(42)
        grid = rng.random((32, 32)).astype(np.float32) * 0.3 + 0.1
        result_normal = compute_confidence_map(grid)
        result_rain = compute_confidence_map(grid, model_precipitation_mm=1.0)
        mask = result_normal.confidence > 0.01
        ratio = result_rain.confidence[mask] / result_normal.confidence[mask]
        np.testing.assert_allclose(ratio.mean(), 1.15, atol=0.05)

    def test_model_boost_capped_at_one(self):
        """Boost can't push confidence above 1.0."""
        grid = np.full((32, 32), 0.95, dtype=np.float32)
        result = compute_confidence_map(grid, model_precipitation_mm=2.0)
        assert result.confidence.max() <= 1.0

    def test_model_none_no_change(self):
        """None (API failure) -> unchanged (fail open)."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_normal = compute_confidence_map(grid)
        result_none = compute_confidence_map(grid, model_precipitation_mm=None)
        np.testing.assert_array_equal(result_none.confidence, result_normal.confidence)


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

    def test_unmatched_weight_keys_logs_warning(self, caplog):
        """Weight keys that don't match any factor should log a warning."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        config = QCConfig(weights={"texture": 1.0, "bogus_factor": 0.5})
        with caplog.at_level(logging.WARNING):
            compute_confidence_map(grid, config)
        assert "bogus_factor" in caplog.text


class TestColdStartPenalty:
    """Cold-start: when clutter map is immature, reduce confidence to avoid
    false positives from unlearned clutter patterns."""

    def test_cold_start_reduces_confidence(self):
        """maturity=0 should reduce confidence by 15%."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_normal = compute_confidence_map(grid, clutter_maturity=0.0)
        result_mature = compute_confidence_map(grid, clutter_maturity=1.0)
        mask = result_mature.confidence > 0
        ratio = result_normal.confidence[mask] / result_mature.confidence[mask]
        np.testing.assert_allclose(ratio, 0.85, atol=0.01)

    def test_mature_clutter_no_penalty(self):
        """maturity=1.0 should not reduce confidence."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_none = compute_confidence_map(grid)  # default maturity=0.0
        result_mature = compute_confidence_map(grid, clutter_maturity=1.0)
        # Mature should have no cold-start penalty (same as no-clutter baseline
        # since clutter_freq is None so clutter weight is redistributed anyway)
        mask = result_mature.confidence > 0
        # Both should be identical when clutter_freq is None
        # (maturity only affects cold-start penalty, not weight redistribution)
        assert result_mature.confidence[mask].mean() >= result_none.confidence[mask].mean()

    def test_cold_start_half_mature(self):
        """maturity=0.5 should have no penalty (boundary)."""
        grid = np.full((32, 32), 0.5, dtype=np.float32)
        result_half = compute_confidence_map(grid, clutter_maturity=0.5)
        result_mature = compute_confidence_map(grid, clutter_maturity=1.0)
        mask = result_mature.confidence > 0
        np.testing.assert_allclose(
            result_half.confidence[mask],
            result_mature.confidence[mask],
            atol=0.01,
        )
