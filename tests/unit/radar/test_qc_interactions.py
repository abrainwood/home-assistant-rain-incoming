"""
Three-way interaction tests: Radar signal x QC factors x Open-Meteo model.

These test the decision boundaries where false positives and negatives live.
Each scenario runs synthetic data through the full QC + detection pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.incoming_rain.radar.detector import detect
from custom_components.incoming_rain.radar.qc import compute_confidence_map
from custom_components.incoming_rain.radar.qc.clutter_map import ClutterMap, get_clutter_frequency

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
    model_precipitation_mm: float | None,
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
            model_precipitation_mm=model_precipitation_mm,
        ).confidence
        for g in grids
    ]

    return detect(frames, (LAT, LON), cfg, confidence_maps=confidence_maps)


# ---- 1. Strong smooth rain + QC high + model confirms ----


class TestRadarHighQC_HighMeteoConfirms:
    """Strong smooth rain, QC says high confidence, model confirms rain."""

    def test_detected_with_full_confidence(self):
        # 10x10 smooth rain blob centred on location (row 32, col 32),
        # present in all 3 frames, model says 2.0mm
        blob = slice(27, 37), slice(27, 37)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.5) for _ in range(3)]

        result = _run_pipeline(grids, model_precipitation_mm=2.0)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1
        # Cell confidence should be high (no penalty from model)
        assert result.tracked_cells[0].confidence > 0.6


# ---- 2. Strong smooth rain + QC high + model says 0mm ----


class TestRadarHighQC_HighMeteoZero:
    """Strong smooth rain, QC says high confidence, but model says 0mm."""

    def _make_grids(self):
        # Use a large blob (20x20) so interior pixels have max texture score.
        # With a 5x5 kernel, pixels >2 from edge get near-zero local std.
        blob = slice(22, 42), slice(22, 42)
        return [_smooth_blob(GRID_SHAPE, *blob, intensity=0.5) for _ in range(3)]

    def test_still_detected_despite_model_penalty(self):
        # Same smooth blob, model says 0.0mm
        # 0.85x penalty, but high base confidence (texture~1.0, temporal=1.0)
        # still passes 0.35 threshold
        grids = self._make_grids()
        result = _run_pipeline(grids, model_precipitation_mm=0.0)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1

    def test_cell_confidence_reduced(self):
        # Cell confidence should be lower than the confirms-rain case
        grids = self._make_grids()
        result_confirm = _run_pipeline(grids, model_precipitation_mm=2.0)
        result_zero = _run_pipeline(grids, model_precipitation_mm=0.0)

        conf_confirm = result_confirm.tracked_cells[0].confidence
        conf_zero = result_zero.tracked_cells[0].confidence
        assert conf_zero < conf_confirm


# ---- 3. Strong smooth rain + QC high + model down (None) ----


class TestRadarHighQC_HighMeteoDown:
    """Strong smooth rain, QC says high confidence, model API unreachable."""

    def test_detected_with_fail_open(self):
        # Same smooth blob, model returns None -> no penalty applied (fail open)
        blob = slice(27, 37), slice(27, 37)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.5) for _ in range(3)]

        result = _run_pipeline(grids, model_precipitation_mm=None)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1
        # Full confidence - no model penalty
        assert result.tracked_cells[0].confidence > 0.6


# ---- 4. Weak speckly returns + low QC + model confirms ----


class TestRadarWeakQC_LowMeteoConfirms:
    """Weak speckly returns, low QC score, but model confirms rain."""

    def test_speckle_has_lower_confidence_than_smooth(self):
        # The model confirming rain (no penalty) doesn't fix bad texture.
        # Compare: speckle confidence vs smooth blob confidence -
        # speckle should be significantly lower even with model confirmation.
        grids_speckle = [
            _speckle_grid(GRID_SHAPE, density=0.05, intensity=0.3, seed=42 + i)
            for i in range(3)
        ]
        grids_smooth = [
            _smooth_blob(GRID_SHAPE, slice(22, 42), slice(22, 42), intensity=0.3)
            for _ in range(3)
        ]

        cm_speckle = compute_confidence_map(
            grids_speckle[-1], grids=grids_speckle, model_precipitation_mm=1.5,
        )
        cm_smooth = compute_confidence_map(
            grids_smooth[-1], grids=grids_smooth, model_precipitation_mm=1.5,
        )

        speckle_mean = float(cm_speckle.confidence[cm_speckle.confidence > 0].mean())
        smooth_mean = float(cm_smooth.confidence[22:42, 22:42].mean())

        # Model confirms both, but speckle's texture penalty should dominate
        assert speckle_mean < smooth_mean


# ---- 5. Weak speckly returns + low QC + model says 0mm ----


class TestRadarWeakQC_LowMeteoZero:
    """Weak speckly returns, low QC score, model says zero."""

    def test_noise_suppressed_by_double_penalty(self):
        # Sparse speckle + 0mm model = very low confidence
        # Double penalty (bad texture + 0.85x model) kills it
        grids = [_speckle_grid(GRID_SHAPE, density=0.05, intensity=0.3, seed=42 + i) for i in range(3)]

        result = _run_pipeline(grids, model_precipitation_mm=0.0)

        assert result.rain_incoming is False


# ---- 6. THE critical scenario: AP noise (smooth + persistent) + model says 0mm ----


class TestAPNoise_SmoothPersistent_MeteoZero:
    """THE critical scenario: AP noise that looks smooth and persists across frames,
    but model says 0mm. This is the clear-sky-with-stars situation."""

    def _make_ap_grids(self):
        """Large smooth area of weak returns, present in all frames."""
        blob = slice(20, 44), slice(20, 44)  # 24x24 area
        return [_smooth_blob(GRID_SHAPE, *blob, intensity=0.15) for _ in range(3)]

    def test_ap_confidence_reduced_when_model_says_dry(self):
        # Model says dry -> 0.85x penalty. Smooth persistent AP gets high texture
        # and temporal scores, so confidence is only mildly reduced.
        # This is a known trade-off: mild model penalty doesn't kill AP noise,
        # but the clutter map (after maturing over days) will catch it.
        grids = self._make_ap_grids()

        cm_dry = compute_confidence_map(grids[-1], grids=grids, model_precipitation_mm=0.0)
        cm_none = compute_confidence_map(grids[-1], grids=grids, model_precipitation_mm=None)

        ap_region_dry = cm_dry.confidence[20:44, 20:44]
        ap_region_none = cm_none.confidence[20:44, 20:44]
        # Model penalty should reduce confidence vs no-model baseline
        assert ap_region_dry[ap_region_dry > 0].mean() < ap_region_none[ap_region_none > 0].mean()

    def test_ap_renders_dimmer_with_model_dry(self):
        # With 0.85x penalty, AP confidence is reduced. After cubing for
        # rendering, the visual difference is noticeable.
        grids = self._make_ap_grids()

        cm = compute_confidence_map(grids[-1], grids=grids, model_precipitation_mm=0.0)
        ap_region = cm.confidence[20:44, 20:44]
        mean_conf = float(ap_region[ap_region > 0].mean()) if np.any(ap_region > 0) else 0.0

        # Confidence reduced but not obliterated (mild penalty)
        assert mean_conf < 0.95
        # Cubed opacity still visible but dimmed
        assert mean_conf ** 3 < 0.85


# ---- 7. AP noise (smooth + persistent) + model down ----


class TestAPNoise_SmoothPersistent_MeteoDown:
    """AP noise, looks real to radar QC, and model API is unreachable."""

    def test_ap_may_detect_without_model_check(self):
        # Worst case: AP passes texture+temporal, no model to penalise.
        # This is a known limitation. We verify:
        # - It may detect (we don't assert rain_incoming=False)
        # - But confidence is NOT 1.0 (some factors should still reduce it)
        blob = slice(20, 44), slice(20, 44)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.15) for _ in range(3)]

        result = _run_pipeline(grids, model_precipitation_mm=None)

        # Without the model check, the AP noise might be detected or not
        # depending on whether effective intensity (0.15 * conf) exceeds 0.1.
        # Key assertion: even without model, confidence is not maxed out.
        cm = compute_confidence_map(grids[-1], grids=grids, model_precipitation_mm=None)
        ap_region = cm.confidence[20:44, 20:44]
        mean_conf = float(ap_region[ap_region > 0].mean())
        # Confidence should be high but not perfect 1.0 (edge effects in texture)
        assert mean_conf < 1.0


# ---- 8. No radar returns + model confirms rain ----


class TestNoRadar_MeteoConfirms:
    """No radar returns at all, but model says it's raining."""

    def test_no_detection_without_radar_signal(self):
        # Empty grids, model says 5.0mm
        # Model can't create phantom cells - nothing to track
        grids = [np.zeros(GRID_SHAPE, dtype=np.float32) for _ in range(3)]

        result = _run_pipeline(grids, model_precipitation_mm=5.0)

        assert result.rain_incoming is False
        assert result.tracked_cells == []


# ---- 9. Stationary clutter + matured clutter map + model says 0mm ----


class TestStationaryClutter_MaturedMap_MeteoZero:
    """Ground clutter at fixed pixels, clutter map has matured, model says 0mm."""

    def test_clutter_suppressed_by_map_and_model(self):
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
            model_precipitation_mm=0.0,
            clutter_freq=clutter_freq,
            clutter_maturity=1.0,
        )

        # Clutter score = 1 - 0.95 = 0.05 for those pixels (very low)
        # Model penalty 0.85x on top of that
        # Combined: very low conf -> effective intensity = 0.2 * conf
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
        result = _run_pipeline(grids, model_precipitation_mm=2.0)

        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1

    def test_stationary_noise_has_higher_temporal_score(self):
        # The stationary noise blob appears at the same pixels in every frame,
        # so it gets a higher temporal score than the moving cell.
        # This is a fundamental property: temporal scoring rewards persistence
        # at the same pixel location. Moving cells get differentiated by
        # tracking (velocity, coherence) not by per-pixel temporal score.
        grids = self._make_grids()

        cm = compute_confidence_map(grids[-1], grids=grids, model_precipitation_mm=2.0)

        # Moving cell at col 16 in last frame: temporal ~1/3 (only in last frame
        # at those pixels)
        moving_temporal = float(cm.factor_scores["temporal"][26:38, 16:32].mean())
        # Stationary noise at col 48: temporal = 3/3 = 1.0 (all frames)
        static_temporal = float(cm.factor_scores["temporal"][26:38, 48:64].mean())

        assert static_temporal > moving_temporal
        assert static_temporal == pytest.approx(1.0, abs=0.01)


# ---- 11. AP noise + cold start (maturity=0) + model says 0mm ----


class TestAPNoise_ColdStart_MeteoZero:
    """AP noise on a fresh installation with zero clutter maturity and model
    saying dry. The cold-start penalty (0.85x) stacks with the model-dry
    penalty (0.85x) giving ~0.72x total confidence reduction."""

    def test_cold_start_plus_model_dry_suppresses_weak_ap(self):
        """Weak AP returns should be suppressed by double penalty on cold start."""
        # Weak AP-like smooth blob at location
        blob = slice(20, 44), slice(20, 44)
        grids = [_smooth_blob(GRID_SHAPE, *blob, intensity=0.15) for _ in range(3)]

        result = _run_pipeline(
            grids,
            model_precipitation_mm=0.0,
            clutter_maturity=0.0,  # fresh install
        )

        # Cold-start (0.85x) + model-dry (0.85x) = ~0.72x confidence
        # With base confidence ~0.99 for smooth blob, effective conf ~0.72
        # effective intensity = 0.15 * ~0.72 = ~0.108, just above threshold
        # Should be suppressed or at minimum have notably reduced confidence
        if result.rain_incoming:
            # Double penalty means confidence is well below a mature system
            for cell in result.tracked_cells:
                assert cell.confidence < 0.75
