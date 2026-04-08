from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.incoming_rain.providers.base import BoundingBox, RadarFrame
from custom_components.incoming_rain.radar.detector import (
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
        Location is at pixel (32, 32). Cell starts at col 10 and moves right by 8 px/frame.
        It will reach the location by frame 3, within the lookahead window.
        """
        frames = []
        for i, col_offset in enumerate([10, 18, 26]):
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
        frames = []
        for i, col_offset in enumerate([32, 24, 16]):
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
        for i, col in enumerate([20, 28, 36]):
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
        for i, col in enumerate([10, 18, 26]):
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
        for i, col_offset in enumerate([10, 18, 26]):
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
        for i, col_offset in enumerate([10, 18, 26]):
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
        for i, col_offset in enumerate([10, 18, 26]):
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
        """A cell with confidence < 0.4 should not trigger rain_incoming."""
        cfg = default_config()
        grid = np.zeros((64, 64), dtype=np.float32)
        # Cell right at location with moderate intensity but low confidence
        grid[31:34, 31:34] = 0.15

        frames = [make_frame(ts(-20 + i * 10), grid, cfg.analysis_bounds) for i in range(3)]

        # Confidence just high enough to pass threshold but cell confidence < 0.4
        # effective = 0.15 * 0.3 = 0.045 -> below threshold anyway
        # Let's use a value that makes effective > threshold but cell confidence < 0.4
        # We need: intensity * confidence >= 0.1 and confidence < 0.4
        # So intensity > 0.25 with confidence 0.39
        grid2 = np.zeros((64, 64), dtype=np.float32)
        grid2[31:34, 31:34] = 0.3
        frames2 = [make_frame(ts(-20 + i * 10), grid2, cfg.analysis_bounds) for i in range(3)]

        confidence_map = np.full((64, 64), 0.35, dtype=np.float32)
        confidence_maps = [confidence_map] * 3

        result = detect(frames=frames2, location=(LAT, LON), config=cfg, confidence_maps=confidence_maps)
        # effective = 0.3 * 0.35 = 0.105 > threshold 0.1 -> cells detected
        # but cell confidence 0.35 < 0.4 -> rain_incoming should be False
        assert result.rain_incoming is False
