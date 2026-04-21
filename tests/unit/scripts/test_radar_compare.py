"""Unit tests for scripts/radar_compare.py pure functions."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root so scripts/ can import custom_components without HA
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.radar_compare import (
    _build_analysis_bounds,
    _build_detector_config,
    _build_report,
    _classify_intensity,
)
from custom_components.rain_incoming.const import (
    INTENSITY_THRESHOLD,
    MAX_ANGULAR_VARIANCE_RADIANS,
    MAX_STORM_SPEED_KMH,
    MIN_CELL_AREA_PIXELS,
    MIN_TEMPORAL_FRAMES,
    PROXIMITY_RADIUS_KM,
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_TILE_SIZE,
    RAINVIEWER_ZOOM,
)
from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.radar.detector import DetectionResult, Confidence


# ---------------------------------------------------------------------------
# _classify_intensity
# ---------------------------------------------------------------------------

def test_classify_intensity_matches_production():
    """_classify_intensity must delegate to production _intensity_label.

    Tests against the production thresholds (0.0/0.20/0.45/0.75) not
    the old script-local thresholds which diverged from the HA sensor.
    """
    from custom_components.rain_incoming.sensor import _intensity_label

    # Verify delegation - same result for a representative sample
    for value in [0.0, 0.05, 0.10, 0.20, 0.21, 0.45, 0.46, 0.75, 0.76, 1.0]:
        assert _classify_intensity(value) == _intensity_label(value), (
            f"_classify_intensity({value}) = {_classify_intensity(value)} "
            f"but production _intensity_label = {_intensity_label(value)}"
        )


def test_classify_intensity_boundary():
    """Boundary values against production thresholds."""
    assert _classify_intensity(0.0) == "none"
    assert _classify_intensity(0.19) == "light"
    assert _classify_intensity(0.20) == "light"       # <= 0.20 is light
    assert _classify_intensity(0.21) == "moderate"     # > 0.20 is moderate
    assert _classify_intensity(0.45) == "moderate"     # <= 0.45 is moderate
    assert _classify_intensity(0.46) == "heavy"        # > 0.45 is heavy
    assert _classify_intensity(0.75) == "heavy"        # <= 0.75 is heavy
    assert _classify_intensity(0.76) == "extreme"      # > 0.75 is extreme
    assert _classify_intensity(1.0) == "extreme"


# ---------------------------------------------------------------------------
# _build_report
# ---------------------------------------------------------------------------

def _make_detection_result(rain_incoming=False, tracked_cells=None, arrival_time=None, max_intensity=0.0):
    return DetectionResult(
        rain_incoming=rain_incoming,
        arrival_time=arrival_time,
        confidence=Confidence.NORMAL,
        frame_count=4,
        max_approaching_intensity=max_intensity,
        tracked_cells=tracked_cells or [],
    )


def _make_grids_and_confs(num_frames: int = 4, size: int = 10):
    grids = [np.zeros((size, size), dtype=np.float32) for _ in range(num_frames)]
    confidence_maps = [np.ones((size, size), dtype=np.float32) for _ in range(num_frames)]
    return grids, confidence_maps


def test_build_report_includes_location():
    grids, confs = _make_grids_and_confs()
    result = _make_detection_result()
    local_time = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    report = _build_report(
        name="Sydney",
        lat=-33.87,
        lon=151.21,
        local_time=local_time,
        grids=grids,
        confidence_maps=confs,
        detection_result=result,
    )

    assert "Sydney" in report
    assert "-33.87" in report
    assert "151.21" in report


def test_build_report_includes_cell_count():
    grids, confs = _make_grids_and_confs()
    cell = MagicMock()
    cell.lat = -33.87
    cell.lon = 151.21
    cell.velocity_kmh = 45.0
    cell.bearing = 270.0
    cell.confidence = 0.85
    result = _make_detection_result(rain_incoming=True, tracked_cells=[cell, cell])
    local_time = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    report = _build_report(
        name="TestLocation",
        lat=-33.87,
        lon=151.21,
        local_time=local_time,
        grids=grids,
        confidence_maps=confs,
        detection_result=result,
    )

    assert "tracked_cells: 2" in report
    assert "Cells:" in report


def test_build_report_no_rain():
    grids, confs = _make_grids_and_confs()
    result = _make_detection_result(rain_incoming=False)
    local_time = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    report = _build_report(
        name="DryPlace",
        lat=-30.0,
        lon=150.0,
        local_time=local_time,
        grids=grids,
        confidence_maps=confs,
        detection_result=result,
    )

    assert "rain_incoming: False" in report
    assert "tracked_cells: 0" in report
    # No cells section should appear when there are no tracked cells
    assert "Cells:" not in report


# ---------------------------------------------------------------------------
# _build_analysis_bounds
# ---------------------------------------------------------------------------

def test_build_analysis_bounds_returns_bounding_box():
    lat, lon = -33.701, 151.209

    bounds = _build_analysis_bounds(lat, lon)

    assert isinstance(bounds, BoundingBox)
    # Centre lat/lon must be inside the returned box
    assert bounds.lat_min < lat < bounds.lat_max
    assert bounds.lon_min < lon < bounds.lon_max
    # Box must be non-degenerate
    assert bounds.lat_max > bounds.lat_min
    assert bounds.lon_max > bounds.lon_min


# ---------------------------------------------------------------------------
# _build_detector_config
# ---------------------------------------------------------------------------

def test_build_detector_config_uses_production_constants():
    lat, lon = -33.701, 151.209

    config = _build_detector_config(lat, lon)

    # Verify config uses the imported production constants, not hardcoded duplicates
    assert config.intensity_threshold == INTENSITY_THRESHOLD
    assert config.min_cell_area_pixels == MIN_CELL_AREA_PIXELS
    assert config.min_temporal_frames == MIN_TEMPORAL_FRAMES
    assert config.max_angular_variance == MAX_ANGULAR_VARIANCE_RADIANS
    assert config.max_storm_speed_kmh == MAX_STORM_SPEED_KMH
    assert config.proximity_radius_km == PROXIMITY_RADIUS_KM

    # Grid dimensions should be computed from RAINVIEWER_TILE_SIZE and RAINVIEWER_ANALYSIS_GRID
    expected_grid_side = RAINVIEWER_TILE_SIZE * (2 * RAINVIEWER_ANALYSIS_GRID + 1)
    assert config.grid_width == expected_grid_side
    assert config.grid_height == expected_grid_side

    # analysis_bounds should be a valid BoundingBox containing the location
    assert isinstance(config.analysis_bounds, BoundingBox)
    assert config.analysis_bounds.contains(lat, lon)
