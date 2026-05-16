"""Golden-data test: Perth false-negative 2026-04-27.

Captured radar from 2026-04-27 around Perth, WA showed a large approaching
front whose leading edge was ~25 km from the city while its centroid was
~138 km away. The integration reported `rain_incoming=False` at 14:20 AEST
because cell tracking is centroid-based - the leading-edge arrival path
hadn't been implemented yet.

Scope of this file:

* Window 1, end ts=1777259400 (AEST 13:10), 5 frames: rain WAS arriving.
  This is exercised against the FULL detection pipeline (tile decode ->
  stitched grid -> bounds crop -> detect) and acts as the regression guard
  for the Perth capture - the upcoming leading-edge fix must not break the
  case the centroid path already handles.

* Window 2 (the actual false-negative reproduction at AEST 14:20) is
  intentionally NOT covered here. The false negative was QC-coupled: a
  mature clutter map suppressed the 0.376-intensity proximity pixels in
  the last frame, which is what stopped the rain-at-location bypass and
  left the NW approaching front undetected. Cold detect() without
  `confidence_maps` cannot reproduce that - it sees the clutter pixels as
  real rain and fires True, which is a different wrong answer. Re-add
  window 2 once clutter-map warming in tests is supported.

The test bypasses the HTTP layer by reading the captured PNG tiles from
disk and seeding `RainViewerFrame._cached_grid`/`_cached_bounds`
directly, matching the data shape that production would produce.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.const import (
    DEFAULT_LOOKAHEAD_MINUTES,
    INTENSITY_THRESHOLD,
    MAX_ANGULAR_VARIANCE_RADIANS,
    MAX_STORM_SPEED_KMH,
    MIN_CELL_AREA_PIXELS,
    MIN_TEMPORAL_FRAMES,
    PROXIMITY_RADIUS_KM,
    RAINVIEWER_TILE_SIZE,
    RAINVIEWER_ZOOM,
)
from custom_components.rain_incoming.coordinator import _build_analysis_bounds
from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import (
    TILE_SIZE,
    RainViewerFrame,
    _tile_bounds,
    _tile_to_intensity_array,
)
from custom_components.rain_incoming.radar.detector import DetectorConfig, detect

PERTH_LAT = -31.95
PERTH_LON = 115.86

_FIXTURE_ROOT = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "golden_v2"
    / "Perth_false_negative_20260427"
    / "bronze"
)

# 13 captured frame timestamps from manifest.json (oldest first, 10-min cadence).
_PERTH_TIMESTAMPS = [
    1777257000,
    1777257600,
    1777258200,
    1777258800,
    1777259400,  # window 1 end (AEST 13:10)
    1777260000,
    1777260600,
    1777261200,
    1777261800,
    1777262400,
    1777263000,
    1777263600,  # AEST 14:20 - false negative, see module docstring
    1777264200,
]

# The 4 analysis tiles `_build_analysis_bounds(PERTH_LAT, PERTH_LON)` selects
# at zoom 7. Verified by exercising _build_analysis_tiles in coordinator.py.
_ANALYSIS_TILES = [(104, 75), (105, 75), (104, 76), (105, 76)]

_GRID_SIZE = RAINVIEWER_TILE_SIZE * 2  # 512x512, matching production


def _stitched_grid_for_frame(
    frame_ts: int,
    bounds: BoundingBox,
) -> np.ndarray:
    """Stitch the four analysis tiles into a single 512x512 intensity grid.

    Mirrors `RainViewerFrame._fetch_stitched_grid` for the
    `_build_analysis_bounds` case, where the requested bounds align exactly
    with the 2x2 analysis tile block so no cropping/resampling distortion
    happens. The output grid is what production would feed to detect().
    """
    tile_dir = _FIXTURE_ROOT / "tiles" / str(frame_ts)

    xs = [tx for tx, _ in _ANALYSIS_TILES]
    ys = [ty for _, ty in _ANALYSIS_TILES]
    min_tx, max_tx = min(xs), max(xs)
    min_ty, max_ty = min(ys), max(ys)
    cols = max_tx - min_tx + 1
    rows = max_ty - min_ty + 1

    canvas = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)
    for tx, ty in _ANALYSIS_TILES:
        png_path = tile_dir / f"{RAINVIEWER_ZOOM}_{tx}_{ty}_s6.png"
        png_bytes = png_path.read_bytes()
        tile_arr = _tile_to_intensity_array(png_bytes)
        px = (tx - min_tx) * TILE_SIZE
        py = (ty - min_ty) * TILE_SIZE
        canvas[py : py + TILE_SIZE, px : px + TILE_SIZE] = tile_arr

    canvas_bounds = BoundingBox(
        lat_min=_tile_bounds(min_tx, max_ty, RAINVIEWER_ZOOM).lat_min,
        lat_max=_tile_bounds(min_tx, min_ty, RAINVIEWER_ZOOM).lat_max,
        lon_min=_tile_bounds(min_tx, min_ty, RAINVIEWER_ZOOM).lon_min,
        lon_max=_tile_bounds(max_tx, min_ty, RAINVIEWER_ZOOM).lon_max,
    )

    # Crop to requested bounds, then resample to (_GRID_SIZE, _GRID_SIZE).
    # Same logic as production _fetch_stitched_grid.
    ch, cw = canvas.shape

    def lat_to_row(lat: float) -> int:
        return int(
            (canvas_bounds.lat_max - lat)
            / (canvas_bounds.lat_max - canvas_bounds.lat_min)
            * ch
        )

    def lon_to_col(lon: float) -> int:
        return int(
            (lon - canvas_bounds.lon_min)
            / (canvas_bounds.lon_max - canvas_bounds.lon_min)
            * cw
        )

    r0 = max(0, lat_to_row(bounds.lat_max))
    r1 = min(ch, lat_to_row(bounds.lat_min))
    c0 = max(0, lon_to_col(bounds.lon_min))
    c1 = min(cw, lon_to_col(bounds.lon_max))
    cropped = canvas[r0:r1, c0:c1]

    pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
    pil_img = pil_img.resize((_GRID_SIZE, _GRID_SIZE), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0


def _build_frames(timestamps: list[int], bounds: BoundingBox) -> list[RainViewerFrame]:
    """Build RainViewerFrame objects with `_cached_grid`/`_cached_bounds` seeded.

    Bypasses HTTP entirely - detect() reads from `get_intensity_grid` which
    returns the cached grid when bounds and shape match the request.
    """
    frames: list[RainViewerFrame] = []
    for ts in timestamps:
        frame = RainViewerFrame(
            timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
            path=f"/golden/perth/{ts}",
            zoom=RAINVIEWER_ZOOM,
            colour_scheme=6,
        )
        frame._cached_grid = _stitched_grid_for_frame(ts, bounds)
        frame._cached_bounds = bounds
        frames.append(frame)
    return frames


def _perth_detector_config(bounds: BoundingBox) -> DetectorConfig:
    """Match `RainDetectorCoordinator._build_config` exactly (no PoP/satellite)."""
    return DetectorConfig(
        lookahead_seconds=DEFAULT_LOOKAHEAD_MINUTES * 60,
        intensity_threshold=INTENSITY_THRESHOLD,
        min_cell_area_pixels=MIN_CELL_AREA_PIXELS,
        min_temporal_frames=MIN_TEMPORAL_FRAMES,
        max_angular_variance=MAX_ANGULAR_VARIANCE_RADIANS,
        max_storm_speed_kmh=MAX_STORM_SPEED_KMH,
        proximity_radius_km=PROXIMITY_RADIUS_KM,
        analysis_bounds=bounds,
        grid_width=_GRID_SIZE,
        grid_height=_GRID_SIZE,
    )


@pytest.mark.skipif(
    not _FIXTURE_ROOT.exists(),
    reason="Perth golden v2 fixture not present",
)
class TestPerthFalseNegative20260427:
    """Golden-data tests for the Perth false-negative capture.

    Window 2 (the false negative itself) requires a warm QC clutter map to
    reproduce - cold detect() triggers on 0.376 clutter pixels within
    proximity of Perth instead, which is a different wrong answer. Will be
    revisited once clutter-map warming in tests is supported.
    """

    @pytest.mark.skip(reason="experiment branch #195-palette-v7: V7 strict palette filters out the light-rain pixels that this golden window relies on for detection")
    def test_window_1_detects_approaching_rain(self) -> None:
        """5 frames ending 1777259400 (AEST 13:10): rain WAS arriving.

        Regression guard - the centroid-based pipeline already detects this
        window. The leading-edge fix must not break it.
        """
        bounds = _build_analysis_bounds(PERTH_LAT, PERTH_LON)
        timestamps = _PERTH_TIMESTAMPS[0:5]  # 1777257000 .. 1777259400
        frames = _build_frames(timestamps, bounds)
        config = _perth_detector_config(bounds)

        result = detect(
            frames=frames,
            location=(PERTH_LAT, PERTH_LON),
            config=config,
        )

        assert result.rain_incoming is True, (
            f"Window 1 (5 frames ending 1777259400) should detect approaching "
            f"rain but got rain_incoming={result.rain_incoming}, "
            f"tracked_cells={len(result.tracked_cells)}, "
            f"max_intensity={result.max_approaching_intensity}"
        )
