"""
Interaction tests: Radar signal x QC factors.

These test the decision boundaries where false positives and negatives live.
Each scenario runs synthetic data through the full QC + detection pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.rain_incoming.radar.detector import detect
from custom_components.rain_incoming.radar.qc import compute_confidence_map
from custom_components.rain_incoming.radar.qc.clutter_map import ClutterMap, get_clutter_frequency

from .test_detector import LAT, LON, default_config, make_frame, ts


# --- Grid builders ---


def _smooth_blob(
    shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
    intensity: float = 0.5,
) -> np.ndarray:
    """A uniform rectangular blob - looks like real rain to the texture scorer."""
    grid = np.zeros(shape, dtype=np.float32)
    grid[row_slice, col_slice] = intensity
    return grid


def _speckle_grid(
    shape: tuple[int, int],
    density: float = 0.05,
    intensity: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Randomly scattered pixels - looks like noise to texture scorer."""
    rng = np.random.default_rng(seed)
    grid = np.zeros(shape, dtype=np.float32)
    mask = rng.random(shape) < density
    grid[mask] = intensity
    return grid


# --- Helpers ---

GRID_SHAPE = (64, 64)


def _run_pipeline(
    grids: list[np.ndarray],
    clutter_freq: np.ndarray | None = None,
    clutter_maturity: float = 1.0,
    config=None,
):
    """Build frames, compute confidence maps, run detect. Returns DetectionResult."""
    cfg = config or default_config()
    bounds = cfg.analysis_bounds

    frames = [
        make_frame(ts(-20 + i * 10), g, bounds)
        for i, g in enumerate(grids)
    ]

    confidence_maps = [
        compute_confidence_map(
            g,
            grids=grids,
            clutter_freq=clutter_freq,
            clutter_maturity=clutter_maturity,
        ).confidence
        for g in grids
    ]

    return detect(frames, (LAT, LON), cfg, confidence_maps=confidence_maps)


# ---- 1. Strong smooth rain + QC high + model confirms ----


class TestRadarHighQC_Detected:
    """Strong smooth rain, QC says high confidence - should be detected."""

    def test_detected_with_full_confidence(self):
        # 10x10 smooth rain blob centred on location (row 32, col 32),
        # present in all 3 frames
        blob = slice(27, 37), slice(27, 37)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.5) for _ in range(3)]

        result = _run_pipeline(grids)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1
        assert result.tracked_cells[0].confidence > 0.6


# ---- 4. Weak speckly returns + low QC ----


class TestRadarWeakQC_SpeckleVsSmooth:
    """Weak speckly returns get lower confidence than smooth rain."""

    def test_speckle_has_lower_confidence_than_smooth(self):
        # Compare: speckle confidence vs smooth blob confidence -
        # speckle should be significantly lower due to texture penalty.
        grids_speckle = [
            _speckle_grid(GRID_SHAPE, density=0.05, intensity=0.3, seed=42 + i)
            for i in range(3)
        ]
        grids_smooth = [
            _smooth_blob(GRID_SHAPE, slice(22, 42), slice(22, 42), intensity=0.3)
            for _ in range(3)
        ]

        cm_speckle = compute_confidence_map(
            grids_speckle[-1], grids=grids_speckle,
        )
        cm_smooth = compute_confidence_map(
            grids_smooth[-1], grids=grids_smooth,
        )

        speckle_mean = float(cm_speckle.confidence[cm_speckle.confidence > 0].mean())
        smooth_mean = float(cm_smooth.confidence[22:42, 22:42].mean())

        # Speckle's texture penalty should dominate
        assert speckle_mean < smooth_mean


# ---- 5. Weak speckly returns + low QC ----


class TestRadarWeakQC_NoiseSuppressed:
    """Weak speckly returns, low QC score - noise should be suppressed."""

    def test_noise_suppressed_by_texture_penalty(self):
        # Very sparse speckle with weak intensity = low effective intensity.
        # Even with moderate texture scores, the combination of weak raw
        # intensity and texture penalty keeps effective intensity near threshold,
        # and speckle cells are too small/incoherent to pass tracking filters.
        grids = [_speckle_grid(GRID_SHAPE, density=0.01, intensity=0.12, seed=42 + i) for i in range(3)]

        result = _run_pipeline(grids)

        assert result.rain_incoming is False


# ---- 6. AP noise (smooth + persistent) ----


class TestAPNoise_SmoothPersistent:
    """AP noise that looks smooth and persists across frames.
    Without the clutter map, this is hard to distinguish from real rain.
    The clutter map (after maturing over days) will catch it."""

    def _make_ap_grids(self):
        """Large smooth area of weak returns, present in all frames."""
        blob = slice(20, 44), slice(20, 44)  # 24x24 area
        return [_smooth_blob(GRID_SHAPE, *blob, intensity=0.15) for _ in range(3)]

    def test_ap_confidence_not_perfect(self):
        # AP passes texture+temporal, but confidence is NOT 1.0
        # (edge effects in texture scoring reduce it slightly)
        grids = self._make_ap_grids()

        cm = compute_confidence_map(grids[-1], grids=grids)
        ap_region = cm.confidence[20:44, 20:44]
        mean_conf = float(ap_region[ap_region > 0].mean())
        # Confidence should be high but not perfect 1.0
        assert mean_conf < 1.0


# ---- 8. No radar returns ----


class TestNoRadar:
    """No radar returns at all - nothing to detect."""

    def test_no_detection_without_radar_signal(self):
        grids = [np.zeros(GRID_SHAPE, dtype=np.float32) for _ in range(3)]

        result = _run_pipeline(grids)

        assert result.rain_incoming is False
        assert result.tracked_cells == []


# ---- 9. Stationary clutter + matured clutter map ----


class TestStationaryClutter_MaturedMap:
    """Ground clutter at fixed pixels, clutter map has matured."""

    def test_clutter_suppressed_by_map(self):
        # Same pixels lit in every grid (simulating ground clutter)
        # Ground clutter is typically weak. With matured clutter map driving
        # confidence down, the effective intensity falls below threshold.
        clutter_spot = slice(30, 36), slice(30, 36)
        grids = [_smooth_blob(GRID_SHAPE, *clutter_spot, intensity=0.12) for _ in range(3)]

        # Simulate a matured clutter map: those pixels have frequency 0.95
        clutter_freq = np.zeros(GRID_SHAPE, dtype=np.float32)
        clutter_freq[clutter_spot] = 0.95

        result = _run_pipeline(
            grids,
            clutter_freq=clutter_freq,
            clutter_maturity=1.0,
        )

        # Clutter score = 1 - 0.95 = 0.05 for those pixels (very low)
        # The clutter factor alone should drive this below threshold
        assert result.rain_incoming is False


# ---- 10. Moving rain cell + stationary noise + model confirms ----


class TestMovingCell_PlusStationaryNoise:
    """Real moving rain cell + stationary noise blobs. Model confirms rain."""

    def _make_grids(self):
        """3 frames: large moving cell approaching location + stationary noise blob.

        The cell must be large enough (16px wide) that interior pixels get
        high texture scores with the 5x5 kernel. The cell approaches
        the location (row 32, col 32) from the west and arrives overhead
        in the last frame. The stationary blob is far away at col 48.
        """
        grids = []
        cell_h = 12
        cell_w = 16
        # Cell moves east: centroid covers row 32 (location row).
        # Rows 26:38 have centroid at row 32. In last frame, col 16:32 has
        # centroid at col 24. 4px/frame stays under the 120 km/h speed cap.
        for i, col in enumerate([8, 12, 16]):
            grid = np.zeros(GRID_SHAPE, dtype=np.float32)
            # Moving rain cell (large, strong) centered on row 32
            grid[26:26 + cell_h, col:col + cell_w] = 0.8
            # Stationary noise blob (at col 48, all frames)
            grid[26:26 + cell_h, 48:48 + cell_w] = 0.4
            grids.append(grid)
        return grids

    def test_moving_cell_tracked(self):
        # Moving cell approaches the location from the west. In the last frame,
        # the cell overlaps the location pixel. High intensity * moderate
        # confidence exceeds thresholds.
        grids = self._make_grids()
        result = _run_pipeline(grids)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1

    def test_stationary_noise_has_higher_temporal_score(self):
        # The stationary noise blob appears at the same pixels in every frame,
        # so it gets a higher temporal score than the moving cell.
        # This is a fundamental property: temporal scoring rewards persistence
        # at the same pixel location. Moving cells get differentiated by
        # tracking (velocity, coherence) not by per-pixel temporal score.
        grids = self._make_grids()

        cm = compute_confidence_map(grids[-1], grids=grids)

        # Moving cell at col 16 in last frame: temporal ~1/3 (only in last frame
        # at those pixels)
        moving_temporal = float(cm.factor_scores["temporal"][26:38, 16:32].mean())
        # Stationary noise at col 48: temporal = 3/3 = 1.0 (all frames)
        static_temporal = float(cm.factor_scores["temporal"][26:38, 48:64].mean())

        assert static_temporal > moving_temporal
        assert static_temporal == pytest.approx(1.0, abs=0.01)


# ---- 11. AP noise + cold start (maturity=0) ----


class TestAPNoise_ColdStart:
    """AP noise on a fresh installation with zero clutter maturity.
    The cold-start penalty (0.85x) provides some confidence reduction."""

    def test_cold_start_reduces_ap_confidence(self):
        """Weak AP returns should have reduced confidence on cold start."""
        # Weak AP-like smooth blob at location
        blob = slice(20, 44), slice(20, 44)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.15) for _ in range(3)]

        result = _run_pipeline(
            grids,
            clutter_maturity=0.0,  # fresh install
        )

        # Cold-start (0.85x) confidence reduction
        # With base confidence ~0.99 for smooth blob, effective conf ~0.85
        if result.rain_incoming:
            for cell in result.tracked_cells:
                assert cell.confidence < 0.90
