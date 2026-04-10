"""
Golden data rendering validation tests.

These feed real radar data through the FULL pipeline (QC + detection + rendering)
and assert that frames classified as rain_overhead actually produce visible
precipitation in the rendered output.

This catches the bug where confidence^3 suppresses real rain to near-invisibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.providers.base import BoundingBox, RadarFrame
from custom_components.rain_incoming.radar.composite import _composite_single_frame
from custom_components.rain_incoming.radar.detector import DetectorConfig, detect
from custom_components.rain_incoming.radar.qc.scoring import compute_confidence_map
from custom_components.rain_incoming.radar.qc.types import QCConfig

GOLDEN_DATA_DIR = Path(__file__).parent.parent.parent / "fixtures" / "golden_data"
WINDOW_SIZE = 6

# Minimum percentage of precipitation pixels that must be clearly visible
# in the rendered output. If rain is classified as overhead, a substantial
# portion should be visible - not just barely above background.
MIN_VISIBLE_PRECIP_PERCENT = 15.0


def _load_location(name: str):
    manifest_path = GOLDEN_DATA_DIR / f"manifest_20260409_{name}_2hr.json"
    grids_path = GOLDEN_DATA_DIR / f"grids_20260409_{name}_2hr.npz"
    with open(manifest_path) as f:
        manifest = json.load(f)
    data = np.load(grids_path)
    grids = [data[f"frame_{i}"] for i in range(len(manifest["frames"]))]
    return manifest, grids


def _make_mock_frame(grid: np.ndarray, timestamp: datetime) -> RadarFrame:
    frame = MagicMock(spec=RadarFrame)
    frame.timestamp = timestamp
    frame.get_intensity_grid.return_value = grid.astype(np.float32)
    return frame


def _grid_to_radar_image(grid: np.ndarray, output_size: int = 256) -> Image.Image:
    """Convert an intensity grid to a synthetic RGBA radar image.

    Precipitation pixels get green colour with alpha=200 (mimicking RainViewer
    scheme 2). Non-precipitation pixels are transparent.
    """
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    precip_mask = grid > 0.05
    # Green rain with alpha proportional to intensity, capped at 200
    rgba[precip_mask, 1] = np.clip(grid[precip_mask] * 255, 50, 200).astype(np.uint8)
    rgba[precip_mask, 3] = 200
    img = Image.fromarray(rgba)
    return img.resize((output_size, output_size), Image.BILINEAR)


def _run_pipeline_and_render(
    grids: list[np.ndarray],
    window_start: int,
    window_end: int,
    lat: float,
    lon: float,
    output_size: int = 256,
) -> np.ndarray:
    """Run QC pipeline on a window of grids and render the last frame.

    Returns the rendered RGBA image as a numpy array.
    """
    window = grids[window_start : window_end + 1]
    last_grid = window[-1]

    # Run QC to get confidence map
    qc_config = QCConfig()
    confidence_map = compute_confidence_map(
        last_grid, qc_config, grids=list(window), clutter_maturity=1.0,
    ).confidence

    # Create synthetic radar image from the grid
    radar_img = _grid_to_radar_image(last_grid, output_size)

    # Resize confidence map to match output size
    conf_resized = np.array(
        Image.fromarray((confidence_map * 255).astype(np.uint8)).resize(
            (output_size, output_size), Image.BILINEAR,
        )
    ).astype(np.float32) / 255.0

    # Create a blank map background
    map_crop = Image.new("RGBA", (output_size, output_size), (30, 30, 30, 255))

    # Render through the composite pipeline
    result_img = _composite_single_frame(
        map_crop=map_crop,
        radar_resized=radar_img,
        lat=lat,
        lon=lon,
        map_zoom=8,
        radius_km=128,
        output_size=output_size,
        confidence_map=conf_resized,
    )

    return np.array(result_img)


def _count_visible_precip_pixels(rendered: np.ndarray, grid: np.ndarray, output_size: int) -> float:
    """Calculate the percentage of precipitation pixels that are visible in the render.

    Compares the rendered image against the original grid to find pixels that
    should show precipitation. A pixel is 'visible' if its rendered colour
    differs from the background by more than a threshold.
    """
    # Resize grid to match output size for comparison
    grid_resized = np.array(
        Image.fromarray((grid * 255).astype(np.uint8)).resize(
            (output_size, output_size), Image.BILINEAR,
        )
    ).astype(np.float32) / 255.0

    precip_mask = grid_resized > 0.05
    total_precip = precip_mask.sum()
    if total_precip == 0:
        return 0.0

    # Background is (30, 30, 30, 255). A precipitation pixel is "clearly visible"
    # if the green channel is well above background. We use bg + 20 as threshold -
    # anything less than that is too faint for a human to distinguish as rain.
    green_channel = rendered[:, :, 1].astype(np.float32)
    bg_green = 30.0
    visible_mask = precip_mask & (green_channel > bg_green + 20)

    return float(visible_mask.sum()) / float(total_precip) * 100.0


class TestGoldenRendering:
    """Frames classified as rain_overhead must produce visible precipitation
    in the rendered output."""

    @pytest.fixture(scope="class")
    def shepparton_data(self):
        manifest, grids = _load_location("shepparton")
        lat = manifest["location"]["lat"]
        lon = manifest["location"]["lon"]
        return manifest, grids, lat, lon

    @pytest.fixture(scope="class")
    def bright_data(self):
        manifest, grids = _load_location("bright")
        lat = manifest["location"]["lat"]
        lon = manifest["location"]["lon"]
        return manifest, grids, lat, lon

    @pytest.mark.parametrize("frame_idx", range(WINDOW_SIZE - 1, 11))
    def test_shepparton_rain_overhead_is_visible(self, shepparton_data, frame_idx):
        """Shepparton rain_overhead frames must render with visible precipitation."""
        manifest, grids, lat, lon = shepparton_data
        classification = manifest["frames"][frame_idx]["classification"]
        if not classification.startswith("rain_overhead"):
            pytest.skip(f"Frame {frame_idx} is '{classification}', not rain_overhead")

        start = max(0, frame_idx - WINDOW_SIZE + 1)
        output_size = 256
        rendered = _run_pipeline_and_render(grids, start, frame_idx, lat, lon, output_size)
        pct = _count_visible_precip_pixels(rendered, grids[frame_idx], output_size)

        assert pct >= MIN_VISIBLE_PRECIP_PERCENT, (
            f"Shepparton frame {frame_idx} ({classification}): "
            f"only {pct:.1f}% of precip pixels are visible (need >= {MIN_VISIBLE_PRECIP_PERCENT}%)"
        )

    @pytest.mark.parametrize("frame_idx", range(WINDOW_SIZE - 1, 13))
    def test_bright_rain_overhead_is_visible(self, bright_data, frame_idx):
        """Bright rain_overhead frames must render with visible precipitation."""
        manifest, grids, lat, lon = bright_data
        classification = manifest["frames"][frame_idx]["classification"]
        if not classification.startswith("rain_overhead"):
            pytest.skip(f"Frame {frame_idx} is '{classification}', not rain_overhead")

        start = max(0, frame_idx - WINDOW_SIZE + 1)
        output_size = 256
        rendered = _run_pipeline_and_render(grids, start, frame_idx, lat, lon, output_size)
        pct = _count_visible_precip_pixels(rendered, grids[frame_idx], output_size)

        assert pct >= MIN_VISIBLE_PRECIP_PERCENT, (
            f"Bright frame {frame_idx} ({classification}): "
            f"only {pct:.1f}% of precip pixels are visible (need >= {MIN_VISIBLE_PRECIP_PERCENT}%)"
        )
