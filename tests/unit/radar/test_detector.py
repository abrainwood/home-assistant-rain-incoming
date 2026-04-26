from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.rain_incoming.providers.base import BoundingBox, RadarFrame
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

