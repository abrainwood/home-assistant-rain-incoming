"""
Golden data validation tests.

These test the full QC + detection pipeline against human-classified real
radar data. If these tests fail, our pipeline disagrees with what a human
sees on the radar.

Classification by Andrew (Product Manager) on 2026-04-09.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.incoming_rain.providers.base import BoundingBox, RadarFrame
from custom_components.incoming_rain.radar.detector import DetectorConfig, detect
from custom_components.incoming_rain.radar.qc.scoring import compute_confidence_map
from custom_components.incoming_rain.radar.qc.types import QCConfig

GOLDEN_DATA_DIR = Path(__file__).parent.parent.parent / "fixtures" / "golden_data"
WINDOW_SIZE = 6  # frames needed for detection


def _load_location(name: str):
    """Load grids and classifications for a location."""
    manifest_path = GOLDEN_DATA_DIR / f"manifest_20260409_{name}_2hr.json"
    grids_path = GOLDEN_DATA_DIR / f"grids_20260409_{name}_2hr.npz"

    with open(manifest_path) as f:
        manifest = json.load(f)

    data = np.load(grids_path)
    grids = [data[f"frame_{i}"] for i in range(len(manifest["frames"]))]

    return manifest, grids


def _make_mock_frame(grid: np.ndarray, timestamp: datetime) -> RadarFrame:
    """Create a mock RadarFrame that returns a pre-built grid."""
    frame = MagicMock(spec=RadarFrame)
    frame.timestamp = timestamp
    frame.get_intensity_grid.return_value = grid.astype(np.float32)
    return frame


def _classify(raw: str) -> bool:
    """Map a human classification string to expected rain_incoming value."""
    return raw.startswith("rain_incoming") or raw.startswith("rain_overhead")


def _run_detection(
    grids_window: list[np.ndarray],
    lat: float,
    lon: float,
    base_time_offset: int = 0,
):
    """Run QC + detection on a window of grids. Returns DetectionResult."""
    bounds = BoundingBox(
        lat_min=lat - 1.5,
        lat_max=lat + 1.5,
        lon_min=lon - 1.5,
        lon_max=lon + 1.5,
    )

    config = DetectorConfig(
        lookahead_seconds=3600,
        intensity_threshold=0.1,
        min_cell_area_pixels=4,
        min_temporal_frames=2,
        max_angular_variance=0.5,
        max_storm_speed_kmh=120.0,
        proximity_radius_km=15.0,
        analysis_bounds=bounds,
        grid_width=256,
        grid_height=256,
    )

    # Create mock frames with 10-minute intervals
    base = datetime(2026, 4, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=base_time_offset
    )
    frames = [
        _make_mock_frame(g, base + timedelta(minutes=i * 10))
        for i, g in enumerate(grids_window)
    ]

    # Run QC
    qc_config = QCConfig()
    confidence_maps = [
        compute_confidence_map(
            g, qc_config, grids=list(grids_window), clutter_maturity=1.0
        ).confidence
        for g in grids_window
    ]

    return detect(frames, (lat, lon), config, confidence_maps=confidence_maps)


class TestSheppartonGoldenData:
    """Shepparton: rain overhead throughout, transitioning to incoming."""

    @pytest.fixture(scope="class")
    def location_data(self):
        manifest, grids = _load_location("shepparton")
        lat = manifest["location"]["lat"]
        lon = manifest["location"]["lon"]
        return manifest, grids, lat, lon

    @pytest.mark.parametrize("frame_idx", range(WINDOW_SIZE - 1, 13))
    def test_detection_matches_classification(self, location_data, frame_idx):
        manifest, grids, lat, lon = location_data

        start = max(0, frame_idx - WINDOW_SIZE + 1)
        window = grids[start : frame_idx + 1]

        frame_info = manifest["frames"][frame_idx]
        classification = frame_info["classification"]
        expected_incoming = _classify(classification)
        timestamp = frame_info["timestamp_utc"]

        result = _run_detection(window, lat, lon, base_time_offset=start * 10)

        assert result.rain_incoming == expected_incoming, (
            f"Frame {frame_idx} ({timestamp}): "
            f"expected rain_incoming={expected_incoming} "
            f"(classified as '{classification}'), "
            f"got rain_incoming={result.rain_incoming}"
        )


class TestBrightGoldenData:
    """Bright: rain overhead throughout, with incoming rain mid-sequence."""

    @pytest.fixture(scope="class")
    def location_data(self):
        manifest, grids = _load_location("bright")
        lat = manifest["location"]["lat"]
        lon = manifest["location"]["lon"]
        return manifest, grids, lat, lon

    @pytest.mark.parametrize("frame_idx", range(WINDOW_SIZE - 1, 13))
    def test_detection_matches_classification(self, location_data, frame_idx):
        manifest, grids, lat, lon = location_data

        start = max(0, frame_idx - WINDOW_SIZE + 1)
        window = grids[start : frame_idx + 1]

        frame_info = manifest["frames"][frame_idx]
        classification = frame_info["classification"]
        expected_incoming = _classify(classification)
        timestamp = frame_info["timestamp_utc"]

        result = _run_detection(window, lat, lon, base_time_offset=start * 10)

        assert result.rain_incoming == expected_incoming, (
            f"Frame {frame_idx} ({timestamp}): "
            f"expected rain_incoming={expected_incoming} "
            f"(classified as '{classification}'), "
            f"got rain_incoming={result.rain_incoming}"
        )


class TestAlburyGoldenData:
    """Albury: rain receeding then incoming, transitioning to overhead."""

    @pytest.fixture(scope="class")
    def location_data(self):
        manifest, grids = _load_location("albury")
        lat = manifest["location"]["lat"]
        lon = manifest["location"]["lon"]
        return manifest, grids, lat, lon

    @pytest.mark.parametrize("frame_idx", range(WINDOW_SIZE - 1, 13))
    def test_detection_matches_classification(self, location_data, frame_idx):
        manifest, grids, lat, lon = location_data

        start = max(0, frame_idx - WINDOW_SIZE + 1)
        window = grids[start : frame_idx + 1]

        frame_info = manifest["frames"][frame_idx]
        classification = frame_info["classification"]
        expected_incoming = _classify(classification)
        timestamp = frame_info["timestamp_utc"]

        result = _run_detection(window, lat, lon, base_time_offset=start * 10)

        assert result.rain_incoming == expected_incoming, (
            f"Frame {frame_idx} ({timestamp}): "
            f"expected rain_incoming={expected_incoming} "
            f"(classified as '{classification}'), "
            f"got rain_incoming={result.rain_incoming}"
        )
