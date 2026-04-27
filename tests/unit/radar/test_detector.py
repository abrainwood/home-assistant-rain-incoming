from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.rain_incoming.providers.base import BoundingBox, RadarFrame
from custom_components.rain_incoming.providers.open_meteo import max_pop_in_window
from custom_components.rain_incoming.radar.confidence import pop_to_threshold_multiplier
from custom_components.rain_incoming.const import STRONG_BYPASS_INTENSITY
from custom_components.rain_incoming.radar.detector import (
    Confidence,
    DetectionResult,
    DetectorConfig,
    TrackedCell,
    detect,
)

# --- Helpers ---

LAT, LON = -33.701, 151.209  # Terry Hills


def make_frame(timestamp: datetime, grid: np.ndarray, bounds: BoundingBox) -> RadarFrame:
    """Create a mock RadarFrame returning a fixed intensity grid."""
    frame = MagicMock(spec=RadarFrame)
    frame.timestamp = timestamp
    frame.get_intensity_grid.return_value = grid.astype(np.float32)
    frame.get_intensity_at.return_value = float(grid.mean())
    return frame


def ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 4, 7, 10, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return base + timedelta(minutes=offset_minutes)


def default_config() -> DetectorConfig:
    return DetectorConfig(
        lookahead_seconds=3600,
        intensity_threshold=0.1,
        min_cell_area_pixels=1,
        min_temporal_frames=2,
        max_angular_variance=0.5,
        max_storm_speed_kmh=120.0,
        proximity_radius_km=5.0,
        analysis_bounds=BoundingBox(
            lat_min=LAT - 1.5,
            lat_max=LAT + 1.5,
            lon_min=LON - 1.5,
            lon_max=LON + 1.5,
        ),
        grid_width=64,
        grid_height=64,
    )


# --- Tests ---

class TestDetectUnavailable:
    def test_no_frames_returns_unavailable(self):
        result = detect(frames=[], location=(LAT, LON), config=default_config())
        assert result.confidence == Confidence.UNAVAILABLE
        assert result.rain_incoming is False
        assert result.arrival_time is None

    def test_one_frame_returns_unavailable(self):
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(0), grid, default_config().analysis_bounds)]
        result = detect(frames=frames, location=(LAT, LON), config=default_config())
        assert result.confidence == Confidence.UNAVAILABLE


class TestDetectDegradedConfidence:
    def test_two_frames_returns_degraded_confidence(self):
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [
            make_frame(ts(-10), grid, default_config().analysis_bounds),
            make_frame(ts(0), grid, default_config().analysis_bounds),
        ]
        result = detect(frames=frames, location=(LAT, LON), config=default_config())
        assert result.confidence == Confidence.DEGRADED

    def test_three_or_more_frames_returns_normal_confidence(self):
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(-20 + i * 10), grid, default_config().analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=default_config())
        assert result.confidence == Confidence.NORMAL


class TestDetectNoRain:
    def test_empty_frames_produce_no_rain_incoming(self):
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(-20 + i * 10), grid, default_config().analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=default_config())
        assert result.rain_incoming is False
        assert result.arrival_time is None

    def test_rain_at_location_is_incoming_with_arrival_now(self):
        """Rain already overhead is reported as incoming with arrival_time = last frame time."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[32, 32] = 0.8  # location is at centre of bounds
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.arrival_time == frames[-1].timestamp


class TestDetectRainApproaching:
    def _make_approaching_frames(self, cfg: DetectorConfig) -> list[RadarFrame]:
        """
        Simulate a rain cell moving east toward Terry Hills.
        Location is at pixel (32, 32). Cell starts at col 18 and moves right by 4 px/frame.
        At ~104 km/h this stays under the 120 km/h speed cap.
        It will reach the location within the lookahead window.
        """
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            # 3x3 rain cell
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        return frames

    def test_rain_approaching_from_west_detected(self):
        cfg = default_config()
        frames = self._make_approaching_frames(cfg)
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.arrival_time is not None

    def test_arrival_time_is_in_the_future(self):
        cfg = default_config()
        frames = self._make_approaching_frames(cfg)
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.arrival_time > frames[-1].timestamp


class TestDetectorPopMultiplier:
    # Deliberately weak: must be above intensity_threshold (0.1) but below
    # STRONG_BYPASS_INTENSITY (0.2) so S7b bypass never fires in this class.
    _WEAK_INTENSITY = 0.15

    def _make_approaching_frames(self, cfg: DetectorConfig) -> list[RadarFrame]:
        """Simulate a weak rain cell moving east toward Terry Hills.

        Intensity is deliberately < STRONG_BYPASS_INTENSITY so the S7b bypass
        doesn't apply and the multiplier gate behavior is cleanly tested.
        """
        assert self._WEAK_INTENSITY < STRONG_BYPASS_INTENSITY, (
            "Test cell intensity must be below STRONG_BYPASS_INTENSITY or S7b bypass will fire"
        )
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = self._WEAK_INTENSITY
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        return frames

    def test_multiplier_raises_frame_requirement(self):
        """PoP multiplier raises effective min_temporal_frames, blocking weak detections."""
        frames = self._make_approaching_frames(default_config())

        # With multiplier=1.0 (default): effective frames = ceil(2 * 1.0) = 2, we have 3 → detects
        cfg_baseline = default_config()
        result_baseline = detect(frames=frames, location=(LAT, LON), config=cfg_baseline)
        assert result_baseline.rain_incoming is True

        # With multiplier=2.0: effective frames = ceil(2 * 2.0) = 4, we have 3 → doesn't detect
        cfg_multiplied = default_config()
        cfg_multiplied.pop_multiplier = 2.0
        result_multiplied = detect(frames=frames, location=(LAT, LON), config=cfg_multiplied)
        assert result_multiplied.rain_incoming is False

    def test_forecast_wiring_low_pop_suppresses_weak_detection(self):
        """Integration: low-PoP forecast suppresses 2-frame detection via multiplier chain."""
        frames = self._make_approaching_frames(default_config())

        # Build hourly forecast: low PoP now, high PoP later
        forecast = [
            (ts(0), 3.0),    # now: 3% PoP
            (ts(60), 60.0),  # +1h: 60% PoP
        ]

        # Window covering "now" → max PoP = 3.0 → multiplier = 3.0
        window_max = max_pop_in_window(forecast, ts(0), ts(30))
        assert window_max == 3.0
        multiplier = pop_to_threshold_multiplier(window_max)
        assert multiplier == 3.0

        cfg_suppressed = default_config()
        cfg_suppressed.pop_multiplier = multiplier
        result = detect(frames=frames, location=(LAT, LON), config=cfg_suppressed)
        # With multiplier=3.0: effective frames = ceil(2 * 3.0) = 6, we have 3 → suppressed
        assert result.rain_incoming is False

    def test_forecast_wiring_high_pop_allows_detection(self):
        """Integration: high-PoP forecast allows weak detection via multiplier chain."""
        frames = self._make_approaching_frames(default_config())

        # Build hourly forecast: high PoP covering the window
        forecast = [
            (ts(0), 60.0),   # now: 60% PoP
            (ts(60), 45.0),  # +1h: 45% PoP
        ]

        # Window covering "now" → max PoP = 60.0 → multiplier = 1.0
        window_max = max_pop_in_window(forecast, ts(0), ts(30))
        assert window_max == 60.0
        multiplier = pop_to_threshold_multiplier(window_max)
        assert multiplier == 1.0

        cfg_allowed = default_config()
        cfg_allowed.pop_multiplier = multiplier
        result = detect(frames=frames, location=(LAT, LON), config=cfg_allowed)
        # With multiplier=1.0: effective frames = ceil(2 * 1.0) = 2, we have 3 → detects
        assert result.rain_incoming is True


class TestDetectRainReceding:
    def test_rain_moving_away_not_detected(self):
        cfg = default_config()
        # Cell starts at col 32 (near location) and moves west (away)
        # 4px/frame = ~104 km/h, under speed cap
        frames = []
        for i, col_offset in enumerate([32, 28, 24]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False


class TestDetectOutsideLookahead:
    def test_rain_arriving_after_lookahead_not_detected(self):
        cfg = DetectorConfig(
            **{**default_config().__dict__, "lookahead_seconds": 300}  # 5 min only
        )
        # Cell is far away, moving slowly - won't arrive within 5 minutes
        frames = []
        for i, col_offset in enumerate([5, 6, 7]):  # barely moving
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False


class TestDetectParallelMiss:
    def test_cell_moving_parallel_misses_location(self):
        """
        Cell moving east but far north of the location.
        Its trajectory is directionally coherent but never enters the proximity circle.
        """
        cfg = default_config()
        frames = []
        # Cell at rows 5-7 (far north of location at row 32), moving east
        # 4px/frame = ~104 km/h, under speed cap
        for i, col in enumerate([20, 24, 28]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[5:8, col:col + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False
        assert result.arrival_time is None


class TestMaxApproachingIntensity:
    def test_no_rain_returns_zero_intensity(self):
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.max_approaching_intensity == 0.0

    def test_approaching_rain_records_max_intensity(self):
        cfg = default_config()
        frames = []
        for i, col in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col:col + 3] = 0.6
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.max_approaching_intensity == pytest.approx(0.6, abs=0.05)

    def test_overhead_rain_sets_intensity(self):
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.9
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.max_approaching_intensity == pytest.approx(0.9, abs=0.05)


class TestTrackedCells:
    def test_tracked_cells_populated_for_approaching_rain(self):
        """When rain is approaching, tracked_cells should have at least one entry."""
        cfg = default_config()
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1

    def test_tracked_cells_empty_for_no_rain(self):
        """When there's no rain, tracked_cells should be empty."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.tracked_cells == []

    def test_tracked_cell_has_valid_lat_lon(self):
        """TrackedCell lat/lon should be within the analysis bounds."""
        cfg = default_config()
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        bounds = cfg.analysis_bounds
        for cell in result.tracked_cells:
            assert bounds.lat_min <= cell.lat <= bounds.lat_max
            assert bounds.lon_min <= cell.lon <= bounds.lon_max

    def test_tracked_cell_has_reasonable_speed_and_bearing(self):
        """TrackedCell speed should be positive and bearing should be 0-360."""
        cfg = default_config()
        frames = []
        # Cell moving east
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert len(result.tracked_cells) >= 1
        cell = result.tracked_cells[0]
        assert cell.velocity_kmh > 0
        assert 0 <= cell.bearing < 360
        # Cell is moving east, so bearing should be ~90 degrees
        assert 45 < cell.bearing < 135

    def test_tracked_cells_empty_when_fewer_than_two_frames(self):
        """With < 2 frames, no detection runs, so tracked_cells should be empty."""
        cfg = default_config()
        result = detect(frames=[], location=(LAT, LON), config=cfg)
        assert result.tracked_cells == []

    def test_overhead_rain_produces_tracked_cell(self):
        """Rain already at the location should also appear in tracked_cells."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.9
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert len(result.tracked_cells) >= 1


class TestRainAtLocation:
    """DetectionResult.rain_at_location distinguishes overhead from approaching."""

    def test_overhead_rain_sets_rain_at_location_true(self):
        """Rain directly at the location must set rain_at_location=True."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.9  # rain at centre = at location
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.rain_at_location is True

    def test_approaching_rain_sets_rain_at_location_false(self):
        """Rain approaching but not yet overhead must set rain_at_location=False."""
        cfg = default_config()
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.rain_at_location is False

    def test_no_rain_sets_rain_at_location_false(self):
        """No rain at all must set rain_at_location=False."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False
        assert result.rain_at_location is False

    # NOTE: We investigated filtering single-frame transient rain at the
    # location (6 of Penrith's 75 FAs). The filter also removes real
    # "rain just arrived" events where rain appears in 1 frame with no
    # cell tracking support. Net result: -2 FA but -4 hits. Not worth it.
    # Real rain arriving looks identical to transient noise when isolated
    # from cell tracking. A smarter approach (e.g., satellite cloud
    # confirmation) is needed to distinguish them.


class TestSpeedCap:
    def test_fast_cell_rejected_by_speed_cap(self):
        """A cell moving at 200+ km/h should NOT trigger rain_incoming."""
        cfg = default_config()  # max_storm_speed_kmh=120
        # Cell moves 8 px/frame at 10-min intervals -> ~208 km/h (exceeds 120)
        # Cell is far north (row 8) so rain-at-location check won't fire
        frames = []
        for i, col_offset in enumerate([10, 18, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[8:11, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False
        assert result.tracked_cells == []


class TestConfidenceMaps:
    def test_low_confidence_noise_not_detected(self):
        """Low-confidence noise (conf=0.2, intensity=0.15) -> effective 0.03 -> below threshold."""
        cfg = default_config()
        # Create grid with borderline intensity
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.15  # just above threshold of 0.1

        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]

        # Confidence map that heavily penalizes
        confidence_map = np.full((64, 64), 0.2, dtype=np.float32)
        confidence_maps = [confidence_map] * 3

        result = detect(frames=frames, location=(LAT, LON), config=cfg, confidence_maps=confidence_maps)
        # effective = 0.15 * 0.2 = 0.03 < threshold 0.1 -> no detection
        assert result.rain_incoming is False

    def test_high_confidence_rain_detected(self):
        """High-confidence rain (conf=0.9, intensity=0.15) -> effective 0.135 -> above threshold."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.15

        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]

        # High confidence
        confidence_map = np.full((64, 64), 0.9, dtype=np.float32)
        confidence_maps = [confidence_map] * 3

        result = detect(frames=frames, location=(LAT, LON), config=cfg, confidence_maps=confidence_maps)
        # effective = 0.15 * 0.9 = 0.135 > threshold 0.1 -> detected
        assert result.rain_incoming is True

    def test_confidence_maps_none_backward_compat(self):
        """confidence_maps=None should work exactly as before."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.9
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]
        result = detect(frames=frames, location=(LAT, LON), config=cfg, confidence_maps=None)
        assert result.rain_incoming is True

    def test_tracked_cell_has_confidence_field(self):
        """TrackedCell should have a confidence field."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.9
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]

        confidence_map = np.full((64, 64), 0.8, dtype=np.float32)
        confidence_maps = [confidence_map] * 3

        result = detect(frames=frames, location=(LAT, LON), config=cfg, confidence_maps=confidence_maps)
        assert len(result.tracked_cells) >= 1
        cell = result.tracked_cells[0]
        assert hasattr(cell, 'confidence')
        assert 0.0 <= cell.confidence <= 1.0

    def test_low_confidence_cell_not_reported_as_incoming(self):
        """A cell with confidence < 0.35 should not trigger rain_incoming."""
        cfg = default_config()

        # We need: intensity * confidence >= 0.1 and confidence < 0.35
        # So intensity > 0.29 with confidence 0.34
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[31:34, 31:34] = 0.3
        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]

        confidence_map = np.full((64, 64), 0.34, dtype=np.float32)
        confidence_maps = [confidence_map] * 3

        result = detect(frames=frames, location=(LAT, LON), config=cfg, confidence_maps=confidence_maps)
        # effective = 0.3 * 0.34 = 0.102 > threshold 0.1 -> cells detected
        # but cell confidence 0.34 < 0.35 -> rain_incoming should be False
        assert result.rain_incoming is False


class TestQCDoesNotSuppressRealRain:
    def test_smooth_persistent_rain_detected_with_qc(self):
        """A smooth rain cell over the location in all frames should NOT be suppressed by QC."""
        cfg = default_config()
        # Create smooth rain blob centered on location (row 32, col 32)
        frames = []
        for i in range(3):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[28:36, 28:36] = 0.5  # smooth uniform block over location
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))

        # Compute confidence maps
        grids = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
        for g in grids:
            g[28:36, 28:36] = 0.5
        from custom_components.rain_incoming.radar.qc import compute_confidence_map
        confidence_maps = [compute_confidence_map(g, grids=grids).confidence for g in grids]

        result = detect(frames, (LAT, LON), cfg, confidence_maps=confidence_maps)
        # The smooth blob should survive QC and be detected as rain at location
        assert result.rain_incoming is True
        assert len(result.tracked_cells) > 0

    def test_approaching_rain_detected_with_qc(self):
        """A strong approaching cell with QC confidence maps should be tracked."""
        # Cell must be large enough (>5x5) to survive texture scoring with a 5x5 kernel.
        cfg = default_config()
        cell_size = 8
        frames = []
        for i, col in enumerate([16, 20, 24]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[28:28 + cell_size, col:col + cell_size] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))

        grids = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
        for i, col in enumerate([16, 20, 24]):
            grids[i][28:28 + cell_size, col:col + cell_size] = 0.8

        from custom_components.rain_incoming.radar.qc import compute_confidence_map
        confidence_maps = [
            compute_confidence_map(g, grids=grids).confidence
            for g in grids
        ]

        result = detect(frames, (LAT, LON), cfg, confidence_maps=confidence_maps)
        assert len(result.tracked_cells) > 0
        assert result.tracked_cells[0].velocity_kmh > 0

    def test_large_approaching_cell_full_detection(self):
        """A large approaching cell should get full detection with QC."""
        cfg = default_config()
        # Large cell (16x16) so interior pixels have high texture confidence
        cell_size = 16
        frames = []
        for i, col in enumerate([8, 12, 16]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[24:24 + cell_size, col:col + cell_size] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))

        grids = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
        for i, col in enumerate([8, 12, 16]):
            grids[i][24:24 + cell_size, col:col + cell_size] = 0.8

        from custom_components.rain_incoming.radar.qc import compute_confidence_map
        confidence_maps = [
            compute_confidence_map(g, grids=grids).confidence
            for g in grids
        ]

        result = detect(frames, (LAT, LON), cfg, confidence_maps=confidence_maps)
        assert result.rain_incoming is True


class TestOverheadRainDetection:
    def test_large_rain_cell_centered_nearby_detected_as_overhead(self):
        """A 20km-wide rain cell centered 10km from location should trigger overhead.

        When a cell's centroid is outside the proximity radius but the cell is
        large enough to cover the location, it should still be detected.
        """
        cfg = default_config()
        # Location is at pixel ~(32, 32). Create a large blob centered at (32, 22)
        # which is ~10 pixels away but extends to cover (32, 32).
        frames = []
        for i in range(3):
            grid = np.zeros((64, 64), dtype=np.float32)
            # Large rain cell: 24 pixels wide, centered at col 22
            # Covers cols 10-34, which includes location at col 32
            grid[24:40, 10:34] = 0.6
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.arrival_time == frames[-1].timestamp

    def test_rain_at_location_in_recent_frames_triggers_incoming(self):
        """Rain over the location in recent frames should trigger rain_incoming
        even if it's gone in the very last frame.

        This is the Shepparton scenario: rain was directly overhead in frames 3-4
        but dissipated by frame 5. The detector should still report rain_incoming.
        """
        cfg = default_config()
        frames = []
        # Frame 0: no rain
        grid0 = np.zeros((64, 64), dtype=np.float32)
        frames.append(make_frame(ts(-50), grid0, cfg.analysis_bounds))
        # Frame 1: no rain
        frames.append(make_frame(ts(-40), grid0, cfg.analysis_bounds))
        # Frame 2: rain approaching - near but not at location
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[28:36, 24:30] = 0.5
        frames.append(make_frame(ts(-30), grid2, cfg.analysis_bounds))
        # Frame 3: rain at location
        grid3 = np.zeros((64, 64), dtype=np.float32)
        grid3[28:36, 28:36] = 0.5
        frames.append(make_frame(ts(-20), grid3, cfg.analysis_bounds))
        # Frame 4: rain at location
        grid4 = np.zeros((64, 64), dtype=np.float32)
        grid4[28:36, 28:36] = 0.4
        frames.append(make_frame(ts(-10), grid4, cfg.analysis_bounds))
        # Frame 5 (last): rain has dissipated/moved away
        grid5 = np.zeros((64, 64), dtype=np.float32)
        frames.append(make_frame(ts(0), grid5, cfg.analysis_bounds))

        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True



class TestClosingDistanceFallback:
    """Tests for the closing-distance fallback path in the detector.

    The fallback triggers when velocity projection doesn't predict arrival
    but the cell's track history shows it consistently closing distance.
    """

    def _make_oblique_frames(
        self, cfg: DetectorConfig, positions: list[tuple[int, int]]
    ) -> list[RadarFrame]:
        """Build frames with a cell at each (row, col) position, 10 min apart."""
        frames = []
        for i, (r, c) in enumerate(positions):
            grid = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.float32)
            grid[r : r + 3, c : c + 3] = 0.8
            frames.append(make_frame(ts(-10 * (len(positions) - 1 - i)), grid, cfg.analysis_bounds))
        return frames

    def test_oblique_approach_detected_via_closing_distance(self):
        """Cell moving obliquely but consistently closing distance should trigger.

        The cell moves NE from the SW quadrant. Its velocity vector doesn't
        intersect the proximity circle, but each frame is closer to the
        location than the last - the closing-distance fallback should fire.
        """
        cfg = default_config()
        # Location at pixel (32, 32). Cell starts far SW and moves NE.
        # Velocity vector points NE but passes well west of location.
        positions = [
            (50, 14),
            (47, 15),
            (44, 16),
            (41, 17),
            (38, 18),
            (35, 19),
        ]
        frames = self._make_oblique_frames(cfg, positions)
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.arrival_time is not None

    def test_cell_that_reverses_not_detected(self):
        """Cell that approached then reversed should NOT trigger.

        The cell closes ~25 km over the first frames then starts moving away.
        Overall first-vs-last distance still shows closing, but the most
        recent frame is farther than the previous one.
        """
        cfg = default_config()
        # Location at pixel (32, 32).
        # Cell approaches from SW then reverses direction in the last frame.
        positions = [
            (50, 14),
            (47, 15),
            (44, 16),
            (41, 17),
            (38, 18),
            (41, 17),  # reversed - now moving away
        ]
        frames = self._make_oblique_frames(cfg, positions)
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False


class TestLeadingEdgeArrival:
    """Tests for the leading-edge arrival fallback.

    Large storm fronts (squall lines) can have incoherent centroid motion
    even while their leading edge clearly approaches the location. The
    leading-edge arrival path uses the minimum pixel distance from any
    cell pixel to the target across the track, instead of centroid-to-
    centroid velocity, so these systems are still detected.
    """

    def _leading_edge_config(self) -> DetectorConfig:
        """Config tuned for ~1 km/pixel.

        Bounds span 0.58 deg lat (~64 km) and 0.70 deg lon (~64 km at
        the test latitude), giving a 64x64 grid with ~1 km/pixel. With
        proximity_radius_km=15 this leaves room for a wide cell whose
        centroid is much further from the location than its leading edge.
        """
        return DetectorConfig(
            lookahead_seconds=3600,
            intensity_threshold=0.1,
            min_cell_area_pixels=1,
            min_temporal_frames=2,
            max_angular_variance=0.5,
            max_storm_speed_kmh=120.0,
            proximity_radius_km=15.0,
            analysis_bounds=BoundingBox(
                lat_min=LAT - 0.29,
                lat_max=LAT + 0.29,
                lon_min=LON - 0.35,
                lon_max=LON + 0.35,
            ),
            grid_width=64,
            grid_height=64,
        )

    def _make_band_frames(
        self,
        cfg: DetectorConfig,
        bands: list[tuple[int, int, int, int]],
        interval_minutes: int = 10,
    ) -> list[RadarFrame]:
        """Build frames each containing a single rectangular band.

        ``bands`` is a list of (row_start, row_end_exclusive, col_start,
        col_end_exclusive). Frames are spaced ``interval_minutes`` apart,
        oldest first, with the final frame at t=0.
        """
        frames = []
        n = len(bands)
        for i, (r0, r1, c0, c1) in enumerate(bands):
            grid = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.float32)
            grid[r0:r1, c0:c1] = 0.8
            frames.append(
                make_frame(
                    ts(-interval_minutes * (n - 1 - i)),
                    grid,
                    cfg.analysis_bounds,
                )
            )
        return frames

    def test_incoherent_centroid_with_approaching_edge_detected(self):
        """Large band with zigzag centroid but a leading edge that closes
        across the track should fire via the leading-edge fallback.

        Frame layout (64x64 grid, ~1 km/pixel, location at pixel (32, 32),
        proximity_radius_km=15.0). All cells stay above row 12 so the
        rain-at-location proximity-box check (rows 17-47, cols 17-47)
        does not fire and detection must come from the new fallback.

          F0 (-30m): rows 0-2,  cols 0-15  -> centroid (1, 7.5)
          F1 (-20m): rows 3-5,  cols 8-23  -> centroid (4, 15.5)
          F2 (-10m): rows 6-8,  cols 0-15  -> centroid (7, 7.5)
          F3   (0m): rows 9-11, cols 8-23  -> centroid (10, 15.5)

        Centroid velocity zigzags east/west each frame. Circular variance
        across the three velocities is ~0.53 > max_angular_variance=0.5,
        so coherence fails and the centroid-projection path returns None.

        Leading edge (closest cell pixel) to (32, 32):
          F0 (2,15) ~34.5 km, F1 (5,23) ~28.5 km,
          F2 (8,15) ~29.4 km, F3 (11,23) ~22.9 km.

        Edge closes ~11.6 km over the track (>10 km noise floor) and
        the most recent step (F2->F3) is closing. Closing rate ~23 km/h
        is well under max_storm_speed_kmh=120. ETA ~20 min, within
        lookahead_seconds=3600.
        """
        cfg = self._leading_edge_config()
        frames = self._make_band_frames(
            cfg,
            [
                (0, 3, 0, 16),
                (3, 6, 8, 24),
                (6, 9, 0, 16),
                (9, 12, 8, 24),
            ],
        )
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is True
        assert result.arrival_time is not None

    def test_leading_edge_closing_faster_than_speed_cap_rejected(self):
        """Edge closing implausibly fast (>max_storm_speed_kmh) is rejected.

        A growing cell whose southern edge sweeps from ~34 km to ~24 km in
        5 minutes implies an edge speed of ~125 km/h, above the 120 km/h
        cap. Centroid motion is coherent and slow (~78 km/h) so the
        centroid speed cap does not fire; the rejection must come from
        the leading-edge fallback's own cap.

        Frames spaced 2.5 minutes apart so the closing rate exceeds the
        cap while the per-frame centroid drift stays within it.

          F0 (-5m):   rows 0-2,  cols 5-15  centroid (1, 10),    edge (2, 15)
          F1 (-2.5m): rows 0-7,  cols 5-15  centroid (3.5, 10),  edge (7, 15)
          F2  (0m):   rows 0-15, cols 5-15  centroid (7.5, 10),  edge (15, 15)

        With ~1 km/pixel edge dist goes 34.5 -> 30.2 -> 24.0 km. Total
        closing 10.4 km in 5 min -> 125 km/h, just above the 120 cap.
        """
        cfg = self._leading_edge_config()
        # Use half-minute precision via _make_band_frames is awkward; build
        # the timestamps inline here.
        from datetime import timedelta
        bands = [
            (0, 3, 5, 16),
            (0, 8, 5, 16),
            (0, 16, 5, 16),
        ]
        offsets_s = [-300, -150, 0]
        frames = []
        for (r0, r1, c0, c1), off in zip(bands, offsets_s):
            grid = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.float32)
            grid[r0:r1, c0:c1] = 0.8
            frames.append(
                make_frame(
                    ts(0) + timedelta(seconds=off),
                    grid,
                    cfg.analysis_bounds,
                )
            )
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False

    def test_leading_edge_retreating_on_last_step_rejected(self):
        """Edge that closed overall but is retreating on the most recent
        step must NOT be reported as incoming.

        Even if the first-vs-last edge distance shows >10 km of closing,
        a cell that has just begun moving away should not be projected
        as an arrival - it could be a passing front that missed.

        Frame layout (incoherent centroid, leading edge in pixels):
          F0 (-30m): row 0, cols 0-3        edge (0, 3)   dist ~42.5 km
          F1 (-20m): rows 3-5, cols 0-15    edge (5, 15)  dist ~31.9 km
          F2 (-10m): rows 6-8, cols 0-15    edge (8, 15)  dist ~29.4 km
          F3   (0m): rows 3-5, cols 0-15    edge (5, 15)  dist ~31.9 km

        Edge closes 42.5 -> 31.9 -> 29.4 (advancing) then 29.4 -> 31.9
        (retreating). The most-recent-step guard rejects this.
        """
        cfg = self._leading_edge_config()
        frames = self._make_band_frames(
            cfg,
            [
                (0, 1, 0, 4),
                (3, 6, 0, 16),
                (6, 9, 0, 16),
                (3, 6, 0, 16),
            ],
        )
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False

    def test_leading_edge_already_inside_proximity_returns_zero_eta(self):
        """When the leading edge has already crossed inside the proximity
        radius, the fallback returns ETA=0 (rain is here now).

        Tested directly because the surrounding pipeline routes any cell
        whose pixels fall in the proximity box through the rain-at-
        location check first; this branch is reachable when a cell's
        centroid is far but its edge is already in proximity.
        """
        from custom_components.rain_incoming.radar.detector import (
            _leading_edge_arrival,
        )
        cfg = self._leading_edge_config()
        from datetime import timedelta
        base = ts(0)
        zero = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.float32)
        frames = [
            make_frame(base + timedelta(minutes=-20), zero, cfg.analysis_bounds),
            make_frame(base + timedelta(minutes=-10), zero, cfg.analysis_bounds),
            make_frame(base, zero, cfg.analysis_bounds),
        ]
        # Pixel positions (r0, r1_excl, c0, c1_excl). Edge dists ~30, 25, ~13 km.
        positions = [(2, 3, 12, 13), (7, 8, 13, 14), (22, 23, 22, 23)]
        labeled_grids = []
        for (r0, r1, c0, c1) in positions:
            grid = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.int32)
            grid[r0:r1, c0:c1] = 1
            labeled_grids.append(grid)
        track = [
            (0, 1, (2.0, 13.0)),
            (1, 1, (7.5, 13.0)),
            (2, 1, (22.5, 22.5)),
        ]
        eta = _leading_edge_arrival(
            track, frames, cfg, loc_row=32, loc_col=32,
            per_frame_labeled=labeled_grids,
            km_per_row=1.0, km_per_col=1.0,
        )
        assert eta == 0.0

    def test_leading_edge_eta_beyond_lookahead_not_detected(self):
        """When the projected ETA exceeds lookahead_seconds, the fallback
        returns None.

        Tested directly: a cell whose edge closes >10 km but very slowly
        over a long track, leaving a large remaining distance, yields an
        ETA outside the lookahead window.
        """
        from custom_components.rain_incoming.radar.detector import (
            _leading_edge_arrival,
        )
        cfg = self._leading_edge_config()
        from datetime import timedelta
        base = ts(0)
        zero = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.float32)
        frames = [
            make_frame(base + timedelta(hours=-2), zero, cfg.analysis_bounds),
            make_frame(base + timedelta(hours=-1), zero, cfg.analysis_bounds),
            make_frame(base, zero, cfg.analysis_bounds),
        ]
        # Single-pixel cells; edge dists 32 -> 28 -> 21 km (closing 11 km
        # over 2 hours = 5.5 km/h). Remaining 21-15=6 km. ETA = 6/5.5*3600
        # = 3927s, just over the 3600s lookahead.
        positions = [(0, 1, 32, 33), (0, 1, 28, 29), (0, 1, 21, 22)]
        labeled_grids = []
        for (r0, r1, c0, c1) in positions:
            grid = np.zeros((cfg.grid_height, cfg.grid_width), dtype=np.int32)
            grid[r0:r1, c0:c1] = 1
            labeled_grids.append(grid)
        track = [
            (0, 1, (0.0, 32.5)),
            (1, 1, (0.0, 28.5)),
            (2, 1, (0.0, 21.5)),
        ]
        eta = _leading_edge_arrival(
            track, frames, cfg, loc_row=32, loc_col=32,
            per_frame_labeled=labeled_grids,
            km_per_row=1.0, km_per_col=1.0,
        )
        assert eta is None

    def test_leading_edge_stationary_not_detected(self):
        """A stationary cell whose leading edge does not advance must
        not fire the leading-edge fallback.

        All three frames contain the same band; centroid velocity is
        zero so coherence fails (zero velocities are treated as
        incoherent). The fallback engages but sees no closing motion
        and returns None.
        """
        cfg = self._leading_edge_config()
        frames = self._make_band_frames(
            cfg,
            [
                (3, 6, 0, 16),
                (3, 6, 0, 16),
                (3, 6, 0, 16),
            ],
        )
        result = detect(frames=frames, location=(LAT, LON), config=cfg)
        assert result.rain_incoming is False


class TestDetectDiagnostics:
    """Tests for the diagnostic trace feature that exposes frame analysis details."""

    def test_diagnostics_trace_records_cell_counts_per_frame(self):
        """When detect() is called with diagnostics=DiagnosticTrace(), the trace
        should record the cell count found in each frame.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Frame 0: empty grid (0 cells)
        grid0 = np.zeros((64, 64), dtype=np.float32)
        frame0 = make_frame(ts(-20), grid0, cfg.analysis_bounds)

        # Frame 1: empty grid (0 cells)
        grid1 = np.zeros((64, 64), dtype=np.float32)
        frame1 = make_frame(ts(-10), grid1, cfg.analysis_bounds)

        # Frame 2: one rain cell (2x2 patch at row 20, col 20 - away from location)
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[20:22, 20:22] = 0.5
        frame2 = make_frame(ts(0), grid2, cfg.analysis_bounds)

        frames = [frame0, frame1, frame2]
        trace = DiagnosticTrace()

        result = detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # Verify trace was populated
        assert len(trace.frames) == 3
        assert trace.frames[0].cell_count == 0
        assert trace.frames[1].cell_count == 0
        assert trace.frames[2].cell_count == 1

    def test_diagnostics_trace_records_tracks_with_frame_indices(self):
        """When detect() runs on frames containing a single rain cell moving across
        3 frames, the trace's tracks list should contain one TrackDiagnostic whose
        frame_indices is [0, 1, 2].
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
            TrackDiagnostic,
        )

        cfg = default_config()
        # Build 3 frames with a single cell moving east by 2 pixels per frame
        # Frame 0: cell at columns 20-21 (2x2 patch at row 20)
        grid0 = np.zeros((64, 64), dtype=np.float32)
        grid0[20:22, 20:22] = 0.5
        frame0 = make_frame(ts(-20), grid0, cfg.analysis_bounds)

        # Frame 1: cell at columns 22-23
        grid1 = np.zeros((64, 64), dtype=np.float32)
        grid1[20:22, 22:24] = 0.5
        frame1 = make_frame(ts(-10), grid1, cfg.analysis_bounds)

        # Frame 2: cell at columns 24-25
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[20:22, 24:26] = 0.5
        frame2 = make_frame(ts(0), grid2, cfg.analysis_bounds)

        frames = [frame0, frame1, frame2]
        trace = DiagnosticTrace()

        result = detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # Verify trace.tracks was populated with the cell track
        assert len(trace.tracks) == 1
        assert trace.tracks[0].frame_indices == [0, 1, 2]

    def test_diagnostics_records_dropped_track_too_short(self):
        """When a cell appears only in the last frame (single-frame track),
        the trace should record that track with status='dropped' and reason='too_short'.

        The detector requires min_temporal_frames=2. A single-frame track in the last
        frame meets the 'ends on last frame' condition but fails the minimum duration check.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Frame 0: empty grid (no cells)
        grid0 = np.zeros((64, 64), dtype=np.float32)
        frame0 = make_frame(ts(-20), grid0, cfg.analysis_bounds)

        # Frame 1: empty grid (no cells)
        grid1 = np.zeros((64, 64), dtype=np.float32)
        frame1 = make_frame(ts(-10), grid1, cfg.analysis_bounds)

        # Frame 2: one rain cell (2x2 patch at row 20, col 20 - away from location)
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[20:22, 20:22] = 0.5
        frame2 = make_frame(ts(0), grid2, cfg.analysis_bounds)

        frames = [frame0, frame1, frame2]
        trace = DiagnosticTrace()

        result = detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # The detector builds one track from the single cell in the last frame
        assert len(trace.tracks) == 1
        # Track contains only frame 2
        assert trace.tracks[0].frame_indices == [2]
        # Track is dropped because it's too short (only 1 frame, requires min 2)
        assert trace.tracks[0].status == "dropped"
        assert trace.tracks[0].reason == "too_short"

    def test_diagnostics_records_dropped_track_ended_early(self):
        """When a cell appears in early frames but vanishes before the last frame,
        the trace should record that track with status='dropped' and reason='ended_early'.

        A track that meets the minimum temporal duration but ends before the last frame
        is rejected because it doesn't span the observation window.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Frame 0: cell at row 20, columns 20-21
        grid0 = np.zeros((64, 64), dtype=np.float32)
        grid0[20:22, 20:22] = 0.5
        frame0 = make_frame(ts(-20), grid0, cfg.analysis_bounds)

        # Frame 1: cell at row 20, columns 22-23 (moved 2px east)
        grid1 = np.zeros((64, 64), dtype=np.float32)
        grid1[20:22, 22:24] = 0.5
        frame1 = make_frame(ts(-10), grid1, cfg.analysis_bounds)

        # Frame 2 (last): empty grid (cell vanished)
        grid2 = np.zeros((64, 64), dtype=np.float32)
        frame2 = make_frame(ts(0), grid2, cfg.analysis_bounds)

        frames = [frame0, frame1, frame2]
        trace = DiagnosticTrace()

        result = detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # The detector builds one track from frames 0-1
        assert len(trace.tracks) == 1
        # Track spans frames 0-1 (len=2, meets min_temporal_frames=2)
        assert trace.tracks[0].frame_indices == [0, 1]
        # But it ends on frame 1, not frame 2 (last_frame_idx), so it's dropped
        assert trace.tracks[0].status == "dropped"
        assert trace.tracks[0].reason == "ended_early"

    def test_diagnostics_records_decision_for_approaching_rain(self):
        """When detect() predicts incoming rain from an approaching cell, the trace's
        decision field should record rain_incoming=True and an arrival_minutes value
        matching the result.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Build 3 frames with a single cell moving east toward the location.
        # Location is at pixel (32, 32). Cell starts at col 20 and moves right by 2 px/frame.
        # At ~52 km/h this stays under the 120 km/h speed cap.
        # It will reach the location within the lookahead window.
        # Frame 0: cell at row 20, columns 20-21 (intensity 0.5)
        grid0 = np.zeros((64, 64), dtype=np.float32)
        grid0[30:33, 20:23] = 0.5
        frame0 = make_frame(ts(-20), grid0, cfg.analysis_bounds)

        # Frame 1: cell at row 20, columns 22-23 (moved 2px east)
        grid1 = np.zeros((64, 64), dtype=np.float32)
        grid1[30:33, 22:25] = 0.5
        frame1 = make_frame(ts(-10), grid1, cfg.analysis_bounds)

        # Frame 2: cell at row 20, columns 24-25 (moved 2px east)
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[30:33, 24:27] = 0.5
        frame2 = make_frame(ts(0), grid2, cfg.analysis_bounds)

        frames = [frame0, frame1, frame2]
        trace = DiagnosticTrace()

        result = detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # Verify the detector predicted rain incoming (approaching cell detected)
        assert result.rain_incoming is True
        assert result.arrival_time is not None

        # Verify trace.decision was populated
        assert trace.decision is not None

        # Mirror the result's rain_incoming status
        assert trace.decision.rain_incoming == result.rain_incoming

        # Verify arrival_minutes matches the calculation:
        # (arrival_time - last_frame_timestamp).total_seconds() / 60.0
        expected_arrival_minutes = (
            result.arrival_time - frames[-1].timestamp
        ).total_seconds() / 60.0
        assert trace.decision.arrival_minutes == pytest.approx(
            expected_arrival_minutes, abs=0.1
        )

        # Verify rain_at_location mirrors the result
        assert trace.decision.rain_at_location == result.rain_at_location

    def test_diagnostics_records_velocity_and_intensity_for_accepted_track(self):
        """When detect() runs on an approaching cell with constant intensity 0.8,
        the trace's accepted track records velocity_kmh > 0, initial_intensity and
        final_intensity both close to 0.8.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Build 3 frames where a cell at row 30-32 moves east 4px/frame with constant intensity 0.8
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset : col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))

        trace = DiagnosticTrace()
        detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        # Find the accepted track (there should be one)
        accepted_tracks = [t for t in trace.tracks if t.status == "accepted"]
        assert len(accepted_tracks) == 1
        track = accepted_tracks[0]

        # Velocity should be > 0 (cell is moving)
        assert track.velocity_kmh is not None
        assert track.velocity_kmh > 0

        # Intensity should be ~0.8 at both ends (constant intensity)
        assert track.initial_intensity is not None
        assert track.final_intensity is not None
        assert track.initial_intensity == pytest.approx(0.8, abs=0.01)
        assert track.final_intensity == pytest.approx(0.8, abs=0.01)

    def test_diagnostics_records_bearing_for_accepted_track(self):
        """A cell moving east (increasing column) should have bearing_degrees ~90.

        Bearing is compass convention: 0=N, 90=E, 180=S, 270=W.
        An east-moving cell (constant row, increasing col) should report ~90°.
        """
        from custom_components.rain_incoming.radar.detector import (
            DiagnosticTrace,
        )

        cfg = default_config()
        # Cell moves east: same row, increasing column
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset : col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))

        trace = DiagnosticTrace()
        detect(frames=frames, location=(LAT, LON), config=cfg, diagnostics=trace)

        accepted = [t for t in trace.tracks if t.status == "accepted"]
        assert len(accepted) == 1
        assert accepted[0].bearing_degrees is not None
        # East-moving cell: bearing should be 90° ±20° (grid is lat/lon not perfect square)
        assert accepted[0].bearing_degrees == pytest.approx(90.0, abs=20.0)


class TestTrackPeakIntensity:
    def test_returns_peak_intensity_across_track_frames(self):
        """_track_peak_intensity finds the max pixel value in the cell region across all track frames.

        Build a 2-frame scenario:
        - Frame 0: cell labeled=1 at rows 0-1, cols 0-1 with intensity 0.4
        - Frame 1: cell labeled=1 at rows 2-3, cols 2-3 with intensity 0.7 (the peak)
        - Noise pixels in other regions should not be picked up

        Expected result: 0.7
        """
        from custom_components.rain_incoming.radar.detector import _track_peak_intensity

        size = 5

        # Frame 0: label=1 occupies top-left 2x2; grid intensity 0.4 there
        labeled_0 = np.zeros((size, size), dtype=np.int32)
        labeled_0[0:2, 0:2] = 1
        grid_0 = np.zeros((size, size), dtype=np.float32)
        grid_0[0:2, 0:2] = 0.4
        grid_0[3, 3] = 0.9  # noise pixel not in label=1 region

        # Frame 1: label=1 occupies middle 2x2; grid intensity 0.7 there (the peak)
        labeled_1 = np.zeros((size, size), dtype=np.int32)
        labeled_1[2:4, 2:4] = 1
        grid_1 = np.zeros((size, size), dtype=np.float32)
        grid_1[2:4, 2:4] = 0.7

        # track: (frame_idx, label, centroid)
        track = [
            (0, 1, (0.5, 0.5)),
            (1, 1, (2.5, 2.5)),
        ]
        per_frame_labeled = [labeled_0, labeled_1]
        effective_grids = [grid_0, grid_1]

        result = _track_peak_intensity(track, per_frame_labeled, effective_grids)

        assert result == pytest.approx(0.7, abs=1e-6), (
            f"Expected peak intensity 0.7 (from frame 1 cell), got {result}"
        )


class TestStrongBypassGate:
    def _make_2frame_strong_approaching(self, cfg: DetectorConfig) -> list[RadarFrame]:
        """2-frame strong cell moving east toward location. Peak intensity 0.8."""
        frames = []
        for i, col_offset in enumerate([22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-10 + i * 10), grid, cfg.analysis_bounds))
        return frames

    def _make_3frame_strong_approaching(self, cfg: DetectorConfig) -> list[RadarFrame]:
        """3-frame strong cell moving east toward location. Peak intensity 0.8."""
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.8
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        return frames

    def _make_3frame_moderate_approaching(self, cfg: DetectorConfig) -> list[RadarFrame]:
        """3-frame moderate cell. Intensity 0.3 - above STRONG_BYPASS_INTENSITY (0.2)."""
        frames = []
        for i, col_offset in enumerate([18, 22, 26]):
            grid = np.zeros((64, 64), dtype=np.float32)
            grid[30:33, col_offset:col_offset + 3] = 0.3
            frames.append(make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds))
        return frames

    def test_strong_persistent_cell_bypasses_multiplier_gate(self):
        """S7b: a strong cell with persistence bypasses even high multipliers.

        Bypass requires BOTH high confidence in real rain (intensity above noise
        floor) AND persistence (more than the bare minimum frames). A 3-frame
        cell at peak 0.8 meets both - clearly real, clearly tracked.
        """
        cfg = default_config()
        cfg.pop_multiplier = 2.0
        frames = self._make_3frame_strong_approaching(cfg)

        result = detect(frames=frames, location=(LAT, LON), config=cfg)

        assert result.rain_incoming is True

    def test_two_frame_strong_cell_does_not_bypass(self):
        """Persistence criterion: a 2-frame track does not bypass even with high intensity.

        QC pipeline still leaves noise in - high intensity alone is not enough
        to overrule a low forecast PoP. The track must also have persisted
        beyond the bare minimum (min_temporal_frames).
        """
        cfg = default_config()
        cfg.pop_multiplier = 2.0
        frames = self._make_2frame_strong_approaching(cfg)

        result = detect(frames=frames, location=(LAT, LON), config=cfg)

        assert result.rain_incoming is False, (
            "2-frame track should not bypass the multiplier gate even at high intensity - "
            "persistence criterion (>= STRONG_BYPASS_MIN_FRAMES) is required"
        )

    def test_moderate_persistent_cell_bypasses(self):
        """Intensity criterion lowered to noise-floor margin, not heavy-rain threshold.

        A 3-frame cell at intensity 0.3 (3x the 0.1 detection floor) is high-confidence
        rain - the multiplier should not suppress it just because rain is light.
        """
        cfg = default_config()
        cfg.pop_multiplier = 2.0
        frames = self._make_3frame_moderate_approaching(cfg)

        result = detect(frames=frames, location=(LAT, LON), config=cfg)

        assert result.rain_incoming is True, (
            "3-frame cell at moderate intensity (0.3 >= STRONG_BYPASS_INTENSITY 0.2) "
            "should bypass even with high multiplier - confidence comes from "
            "above-noise + persistent, not from heavy intensity"
        )
