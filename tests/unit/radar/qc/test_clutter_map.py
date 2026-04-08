from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from custom_components.incoming_rain.radar.qc.clutter_map import (
    ClutterMap,
    get_clutter_frequency,
    load_clutter_map,
    save_clutter_map,
    update_clutter_map,
)
from custom_components.incoming_rain.radar.qc.clutter import score_clutter
from custom_components.incoming_rain.radar.qc.scoring import compute_confidence_map
from custom_components.incoming_rain.radar.qc.types import QCConfig


class TestUpdateClutterMap:
    def test_frequency_approaches_one_after_many_identical_updates(self):
        """After many updates with echo at same pixel, frequency should approach 1.0."""
        cm = ClutterMap(
            echo_count=np.zeros((16, 16), dtype=np.float32),
            update_count=0.0,
        )
        grid = np.zeros((16, 16), dtype=np.float32)
        grid[8, 8] = 0.5
        for _ in range(200):
            update_clutter_map(cm, grid)
        freq = get_clutter_frequency(cm)
        assert freq[8, 8] > 0.9

    def test_frequency_decays_toward_zero_after_empty_updates(self):
        """After many empty updates, frequency should decay toward 0.0.

        With a 14-day half-life (decay=0.99966, ~2016 updates per half-life),
        we need many more empty updates to see significant decay.
        """
        cm = ClutterMap(
            echo_count=np.zeros((16, 16), dtype=np.float32),
            update_count=0.0,
        )
        grid = np.zeros((16, 16), dtype=np.float32)
        grid[8, 8] = 0.5
        # Build up clutter
        for _ in range(100):
            update_clutter_map(cm, grid)
        freq_before = get_clutter_frequency(cm)[8, 8]
        assert freq_before > 0.5

        # Now empty updates - need ~7000 (~5 half-lives) to decay below 0.1
        empty = np.zeros((16, 16), dtype=np.float32)
        for _ in range(7000):
            update_clutter_map(cm, empty)
        freq_after = get_clutter_frequency(cm)[8, 8]
        assert freq_after < 0.1

    def test_frequency_values_between_zero_and_one(self):
        """All frequency values must be in [0.0, 1.0]."""
        cm = ClutterMap(
            echo_count=np.zeros((16, 16), dtype=np.float32),
            update_count=0.0,
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            grid = (rng.random((16, 16)) > 0.7).astype(np.float32) * 0.5
            update_clutter_map(cm, grid)
        freq = get_clutter_frequency(cm)
        assert freq.min() >= 0.0
        assert freq.max() <= 1.0


class TestSaveLoadClutterMap:
    def test_save_load_round_trip(self):
        """Save then load should preserve data."""
        cm = ClutterMap(
            echo_count=np.random.default_rng(7).random((16, 16)).astype(np.float32),
            update_count=42.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_clutter_map(cm, path)
            loaded = load_clutter_map(path)
            assert loaded is not None
            np.testing.assert_array_almost_equal(loaded.echo_count, cm.echo_count)
            assert loaded.update_count == pytest.approx(cm.update_count)
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_none(self):
        """Loading a non-existent file should return None."""
        result = load_clutter_map("/nonexistent/path/clutter.npz")
        assert result is None


class TestScoreClutter:
    def test_high_clutter_frequency_low_score(self):
        """Pixels at high clutter locations should get low confidence."""
        grid = np.full((16, 16), 0.5, dtype=np.float32)
        clutter_freq = np.full((16, 16), 0.8, dtype=np.float32)
        score = score_clutter(grid, clutter_freq)
        assert score[8, 8] == pytest.approx(0.2)

    def test_no_clutter_full_score(self):
        """Pixels at locations with no clutter history should get 1.0."""
        grid = np.full((16, 16), 0.5, dtype=np.float32)
        clutter_freq = np.zeros((16, 16), dtype=np.float32)
        score = score_clutter(grid, clutter_freq)
        assert score[8, 8] == pytest.approx(1.0)

    def test_no_echo_zero_score(self):
        """Non-echo pixels should get 0.0 regardless of clutter frequency."""
        grid = np.zeros((16, 16), dtype=np.float32)
        clutter_freq = np.full((16, 16), 0.5, dtype=np.float32)
        score = score_clutter(grid, clutter_freq)
        assert (score == 0.0).all()


class TestClutterMaturityGating:
    def test_new_map_reduces_clutter_weight(self):
        """With a brand new clutter map (update_count=0), clutter weight should be
        effectively zero and other weights should compensate."""
        grid = np.full((16, 16), 0.5, dtype=np.float32)
        clutter_freq = np.full((16, 16), 0.9, dtype=np.float32)

        # With clutter at full weight (maturity=1.0), confidence should be lower
        config_with_clutter = QCConfig(
            weights={"texture": 0.40, "temporal": 0.35, "clutter": 0.25}
        )
        result_mature = compute_confidence_map(
            grid,
            config=config_with_clutter,
            grids=[grid],
            clutter_freq=clutter_freq,
            clutter_maturity=1.0,
        )

        # With clutter at zero maturity, clutter weight should be redistributed
        result_immature = compute_confidence_map(
            grid,
            config=config_with_clutter,
            grids=[grid],
            clutter_freq=clutter_freq,
            clutter_maturity=0.0,
        )

        # Immature should have higher confidence (clutter penalty not applied)
        assert result_immature.confidence[8, 8] > result_mature.confidence[8, 8]
