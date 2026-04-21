"""Tests for CapturedRadarFrame - radar frames loaded from captured tile PNGs on disk."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from custom_components.rain_incoming.providers.base import BoundingBox


# Fixtures path for Melbourne_dry golden tiles
_FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "golden_v2" / "Melbourne_dry" / "bronze" / "tiles"

# Melbourne_dry analysis tiles: (x, y) at zoom 7
_ZOOM = 7
_DRY_TS = 1776725400      # all-zero precipitation
_NOISY_TS = 1776731400    # trace noise (0.05% coverage)
_ANALYSIS_TILES = [(115, 78), (116, 78), (115, 79), (116, 79)]


def _tile_paths_for(ts: int) -> dict[tuple[int, int], Path]:
    tile_dir = _FIXTURES / str(ts)
    return {
        (x, y): tile_dir / f"7_{x}_{y}_s2.png"
        for x, y in _ANALYSIS_TILES
    }


@pytest.fixture
def dry_frame():
    from scripts.backtest.captured_frame import CapturedRadarFrame
    ts = datetime.fromtimestamp(_DRY_TS, tz=timezone.utc)
    return CapturedRadarFrame(
        _timestamp=ts,
        _tile_paths=_tile_paths_for(_DRY_TS),
        _zoom=_ZOOM,
    )


@pytest.fixture
def noisy_frame():
    from scripts.backtest.captured_frame import CapturedRadarFrame
    ts = datetime.fromtimestamp(_NOISY_TS, tz=timezone.utc)
    return CapturedRadarFrame(
        _timestamp=ts,
        _tile_paths=_tile_paths_for(_NOISY_TS),
        _zoom=_ZOOM,
    )


def _analysis_bounds() -> BoundingBox:
    """Build the bounding box that covers the 2x2 analysis tile grid."""
    from custom_components.rain_incoming.providers.rainviewer import _tile_bounds
    top_left = _tile_bounds(115, 78, _ZOOM)   # NW corner tile
    bottom_right = _tile_bounds(116, 79, _ZOOM)  # SE corner tile
    return BoundingBox(
        lat_min=bottom_right.lat_min,
        lat_max=top_left.lat_max,
        lon_min=top_left.lon_min,
        lon_max=bottom_right.lon_max,
    )


# ---------------------------------------------------------------------------
# test_timestamp_property
# ---------------------------------------------------------------------------

def test_timestamp_property():
    from scripts.backtest.captured_frame import CapturedRadarFrame
    ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    frame = CapturedRadarFrame(
        _timestamp=ts,
        _tile_paths={},
        _zoom=7,
    )
    assert frame.timestamp == ts


# ---------------------------------------------------------------------------
# test_get_intensity_grid_loads_tiles
# ---------------------------------------------------------------------------

def test_get_intensity_grid_loads_tiles(dry_frame):
    bounds = _analysis_bounds()
    grid = dry_frame.get_intensity_grid(bounds, 512, 512)

    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.float32
    assert grid.shape == (512, 512)


# ---------------------------------------------------------------------------
# test_get_intensity_grid_caches_result
# ---------------------------------------------------------------------------

def test_get_intensity_grid_caches_result(dry_frame):
    bounds = _analysis_bounds()
    grid_first = dry_frame.get_intensity_grid(bounds, 512, 512)
    grid_second = dry_frame.get_intensity_grid(bounds, 512, 512)

    assert grid_first is grid_second  # same object, not a copy


# ---------------------------------------------------------------------------
# test_dry_tiles_produce_zero_grid
# ---------------------------------------------------------------------------

def test_dry_tiles_produce_zero_grid(dry_frame):
    bounds = _analysis_bounds()
    grid = dry_frame.get_intensity_grid(bounds, 512, 512)

    assert grid.max() == 0.0, f"Expected all zeros for dry timestamp, got max={grid.max()}"


# ---------------------------------------------------------------------------
# test_noisy_tiles_produce_nonzero_pixels
# ---------------------------------------------------------------------------

def test_noisy_tiles_produce_nonzero_pixels(noisy_frame):
    bounds = _analysis_bounds()
    grid = noisy_frame.get_intensity_grid(bounds, 512, 512)

    nonzero_count = int((grid > 0).sum())
    assert nonzero_count > 0, "Expected some non-zero pixels for noisy timestamp"


# ---------------------------------------------------------------------------
# test_missing_tile_produces_zeros
# ---------------------------------------------------------------------------

def test_missing_tile_produces_zeros():
    from scripts.backtest.captured_frame import CapturedRadarFrame
    ts = datetime.fromtimestamp(_DRY_TS, tz=timezone.utc)
    # Point all tile paths at a non-existent file
    missing_paths = {(x, y): Path("/tmp/does_not_exist.png") for x, y in _ANALYSIS_TILES}
    frame = CapturedRadarFrame(
        _timestamp=ts,
        _tile_paths=missing_paths,
        _zoom=_ZOOM,
    )
    bounds = _analysis_bounds()
    grid = frame.get_intensity_grid(bounds, 512, 512)

    assert isinstance(grid, np.ndarray)
    assert grid.shape == (512, 512)
    assert grid.max() == 0.0, "Missing tiles should produce an all-zero grid"


# ---------------------------------------------------------------------------
# test_get_intensity_at_returns_zero
# ---------------------------------------------------------------------------

def test_get_intensity_at_returns_zero(dry_frame):
    # get_intensity_at is a stub - always returns 0.0
    result = dry_frame.get_intensity_at(-37.8, 144.9)
    assert result == 0.0
