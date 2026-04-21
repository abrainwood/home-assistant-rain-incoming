"""Unit tests for scripts/radar_mockup.py."""
from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS
from custom_components.rain_incoming.const import (
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_COLOUR_SCHEME,
    RAINVIEWER_TILE_SIZE,
    RAINVIEWER_ZOOM,
)
from scripts.radar_mockup import (
    COLOUR_SCHEME,
    GRID_HALF,
    OUTPUT_SIZE,
    TILE_SIZE,
    ZOOM,
    _MAX_COLOUR_DIST,
    _PRECIP_COLOURS,
    crop_to_radius,
    draw_location_marker,
    draw_range_ring,
    location_to_pixel,
    fetch_radar_grid,
    fetch_tile_grid,
    filter_precipitation_only,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_rgba(width: int, height: int, rgba: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (width, height), rgba)
    return img


def _blank_rgba(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


# ---------------------------------------------------------------------------
# _PRECIP_COLOURS is derived from production - not re-encoded
# ---------------------------------------------------------------------------

def test_precip_colours_match_production():
    """_PRECIP_COLOURS must be the RGB component of the production table, nothing else."""
    expected = [(r, g, b) for r, g, b, _ in PRECIP_COLOURS]
    assert _PRECIP_COLOURS == expected


def test_constants_match_production():
    """Script constants must be imported from production, not hardcoded."""
    assert ZOOM == RAINVIEWER_ZOOM
    assert COLOUR_SCHEME == RAINVIEWER_COLOUR_SCHEME
    assert TILE_SIZE == RAINVIEWER_TILE_SIZE
    assert GRID_HALF == RAINVIEWER_ANALYSIS_GRID


# ---------------------------------------------------------------------------
# location_to_pixel
# ---------------------------------------------------------------------------

def test_location_to_pixel_center_tile_returns_canvas_center():
    """A location at the exact center of the tile grid should map to canvas center."""
    from custom_components.rain_incoming.radar.geo import lat_lon_to_tile
    from custom_components.rain_incoming.providers.rainviewer import _tile_bounds

    lat, lon = -33.701, 151.209
    cx, cy = lat_lon_to_tile(lat, lon, ZOOM)
    # Get the center of the center tile
    tb = _tile_bounds(cx, cy, ZOOM)
    center_lat = (tb.lat_min + tb.lat_max) / 2
    center_lon = (tb.lon_min + tb.lon_max) / 2

    px, py = location_to_pixel(center_lat, center_lon, cx, cy, ZOOM, GRID_HALF, TILE_SIZE)
    canvas_size = (2 * GRID_HALF + 1) * TILE_SIZE
    canvas_center = canvas_size // 2

    # Should be near the center. Not exact because the center of a tile
    # in lat/lon space isn't exactly at the midpoint of pixel space (Mercator
    # distortion). Allow a full tile's worth of tolerance.
    assert abs(px - canvas_center) < TILE_SIZE, f"x={px} expected near {canvas_center}"
    assert abs(py - canvas_center) < TILE_SIZE, f"y={py} expected near {canvas_center}"


def test_location_to_pixel_returns_within_canvas():
    """Result must be within canvas bounds for a location inside the grid."""
    from custom_components.rain_incoming.radar.geo import lat_lon_to_tile

    lat, lon = -33.701, 151.209
    cx, cy = lat_lon_to_tile(lat, lon, ZOOM)

    px, py = location_to_pixel(lat, lon, cx, cy, ZOOM, GRID_HALF, TILE_SIZE)
    canvas_size = (2 * GRID_HALF + 1) * TILE_SIZE

    assert 0 <= px < canvas_size, f"x={px} out of canvas bounds [0, {canvas_size})"
    assert 0 <= py < canvas_size, f"y={py} out of canvas bounds [0, {canvas_size})"


# ---------------------------------------------------------------------------
# filter_precipitation_only
# ---------------------------------------------------------------------------

def test_filter_precipitation_only_keeps_rain_pixels():
    """Pixels matching a known precipitation colour are preserved (with boosted alpha)."""
    # Pick the first precipitation colour from the production table
    pr, pg, pb, _ = PRECIP_COLOURS[0]
    size = 8
    img = _solid_rgba(size, size, (pr, pg, pb, 200))

    result = filter_precipitation_only(img)
    arr = np.array(result)

    # ALL pixels should be kept as precipitation (solid tile of precip colour)
    assert (arr[:, :, 3] > 0).all(), "Expected ALL precipitation pixels to be retained"
    # RGB values should be unchanged
    assert (arr[:, :, 0] == pr).all()
    assert (arr[:, :, 1] == pg).all()
    assert (arr[:, :, 2] == pb).all()


def test_filter_precipitation_only_clears_transparent():
    """Fully transparent tile returns an all-zero array."""
    img = _blank_rgba(16, 16)
    result = filter_precipitation_only(img)
    arr = np.array(result)
    assert (arr == 0).all(), "Fully transparent tile should return all zeros"


def test_filter_precipitation_only_clears_non_precip_pixels():
    """Pixels that are opaque but not a precipitation colour are zeroed out."""
    # Pure green is far from all precipitation colours
    img = _solid_rgba(8, 8, (0, 255, 0, 255))
    result = filter_precipitation_only(img)
    arr = np.array(result)
    assert (arr[:, :, 3] == 0).all(), "Non-precipitation opaque pixels should be cleared"


def test_filter_precipitation_only_boosts_alpha():
    """Matched precipitation pixels have their alpha clamped upward."""
    pr, pg, pb, _ = PRECIP_COLOURS[4]
    original_alpha = 100
    img = _solid_rgba(4, 4, (pr, pg, pb, original_alpha))
    result = filter_precipitation_only(img)
    arr = np.array(result)
    # Alpha should be boosted by 40 (clamped to 255)
    expected_alpha = min(255, original_alpha + 40)
    assert (arr[:, :, 3] == expected_alpha).all()


# ---------------------------------------------------------------------------
# draw_location_marker
# ---------------------------------------------------------------------------

def test_draw_location_marker_changes_pixels():
    """draw_location_marker modifies pixels near the centre point."""
    size = 200
    img = _blank_rgba(size, size)
    cx, cy = size // 2, size // 2

    draw_location_marker(img, cx, cy)

    arr = np.array(img)
    # At least some pixels near the centre should be non-zero
    region = arr[cy - 15:cy + 15, cx - 15:cx + 15]
    assert region.max() > 0, "Expected pixels to be drawn near the location marker centre"


def test_draw_location_marker_draws_at_specified_coords():
    """Marker pixels appear at the specified (x, y), not at the origin."""
    size = 400
    img = _blank_rgba(size, size)
    cx, cy = 300, 100

    draw_location_marker(img, cx, cy)

    arr = np.array(img)
    # Region around the origin should be empty
    origin_region = arr[0:30, 0:30]
    assert origin_region.max() == 0, "No pixels should be drawn near the origin"
    # Region around the marker should have content
    marker_region = arr[cy - 15:cy + 15, cx - 15:cx + 15]
    assert marker_region.max() > 0, "Pixels should be drawn near the specified coordinates"


# ---------------------------------------------------------------------------
# draw_range_ring
# ---------------------------------------------------------------------------

def test_draw_range_ring_draws_pixels():
    """draw_range_ring produces at least some non-transparent pixels on the image."""
    size = 400
    img = _blank_rgba(size, size)
    cx, cy = size // 2, size // 2
    kpp = 0.5  # 0.5 km/pixel -> 64km radius = 128px

    draw_range_ring(img, cx, cy, 64, kpp, "64km")

    arr = np.array(img)
    assert arr.max() > 0, "draw_range_ring should produce non-transparent pixels"


def test_draw_range_ring_pixels_near_expected_radius():
    """Ring pixels appear at approximately the expected pixel radius from centre."""
    size = 600
    img = _blank_rgba(size, size)
    cx, cy = size // 2, size // 2
    radius_km = 50
    kpp = 0.5
    expected_radius_px = int(radius_km / kpp)  # 100px

    draw_range_ring(img, cx, cy, radius_km, kpp, "50km")

    arr = np.array(img)
    # Check pixels exist in a narrow band around the expected radius
    # Sample the top of the ring (cx, cy - r_px)
    y_ring = cy - expected_radius_px
    assert 0 <= y_ring < size, (
        f"Ring top y={y_ring} is out of bounds for image size {size} - test setup is broken"
    )
    # There should be a pixel drawn within +-2px of the ring top
    band = arr[max(0, y_ring - 2):min(size, y_ring + 3), cx - 2:cx + 3]
    assert band.max() > 0, "Ring pixels should appear at the expected radius"


# ---------------------------------------------------------------------------
# crop_to_radius
# ---------------------------------------------------------------------------

def test_crop_to_radius_returns_output_size():
    """crop_to_radius always returns an image of OUTPUT_SIZE x OUTPUT_SIZE."""
    img = _solid_rgba(1280, 1280, (50, 100, 150, 255))
    kpp = 0.5
    radius_km = 128
    cropped = crop_to_radius(img, 640, 640, radius_km, kpp)
    assert cropped.size == (OUTPUT_SIZE, OUTPUT_SIZE)


def test_crop_to_radius_clamps_to_image_bounds():
    """crop_to_radius clamps the crop region to the image bounds without raising."""
    img = _solid_rgba(100, 100, (80, 80, 80, 255))
    # loc at edge, large radius - crop will be clamped
    kpp = 0.3
    cropped = crop_to_radius(img, 10, 10, 200, kpp)
    assert cropped.size == (OUTPUT_SIZE, OUTPUT_SIZE)


def test_crop_to_radius_different_radii_produce_same_output_size():
    """Output is always OUTPUT_SIZE regardless of the radius requested."""
    img = _solid_rgba(2000, 2000, (60, 120, 180, 255))
    cx, cy = 1000, 1000
    kpp = 1.0
    for radius_km in [32, 64, 128, 256]:
        cropped = crop_to_radius(img, cx, cy, radius_km, kpp)
        assert cropped.size == (OUTPUT_SIZE, OUTPUT_SIZE), (
            f"Expected {OUTPUT_SIZE}x{OUTPUT_SIZE} for radius {radius_km}km"
        )


# ---------------------------------------------------------------------------
# fetch_tile_grid (async, mocked session)
# ---------------------------------------------------------------------------

def _make_tile_png(rgba=(100, 100, 100, 255)):
    """Create a small valid PNG tile as bytes."""
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), rgba)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_session(tile_bytes=None, status=200):
    """Create a mock aiohttp session that returns tile_bytes for all GETs."""
    if tile_bytes is None:
        tile_bytes = _make_tile_png()

    resp = AsyncMock()
    resp.status = status
    resp.read = AsyncMock(return_value=tile_bytes)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    return session


@pytest.mark.asyncio
async def test_fetch_tile_grid_returns_correct_size():
    """Canvas size must be (2*GRID_HALF+1) * TILE_SIZE on each side."""
    session = _mock_session()
    canvas = await fetch_tile_grid(session, "http://test/{z}/{x}/{y}.png", 100, 50)
    expected = (2 * GRID_HALF + 1) * TILE_SIZE
    assert canvas.size == (expected, expected)


@pytest.mark.asyncio
async def test_fetch_tile_grid_fetches_correct_tile_count():
    """Should fetch (2*GRID_HALF+1)^2 tiles."""
    session = _mock_session()
    await fetch_tile_grid(session, "http://test/{z}/{x}/{y}.png", 100, 50)
    expected_count = (2 * GRID_HALF + 1) ** 2
    assert session.get.call_count == expected_count


@pytest.mark.asyncio
async def test_fetch_tile_grid_survives_http_error(capsys):
    """Failed tile fetches are skipped with a printed error, not raised."""
    resp = AsyncMock()
    resp.__aenter__ = AsyncMock(side_effect=Exception("connection reset"))
    resp.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=resp)

    canvas = await fetch_tile_grid(session, "http://test/{z}/{x}/{y}.png", 100, 50)
    # Should return a canvas (all transparent since all tiles failed)
    assert canvas.size[0] > 0
    assert "Failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_fetch_radar_grid_survives_http_error(capsys):
    """Failed radar tile fetches are skipped with a printed error, not raised."""
    resp = AsyncMock()
    resp.__aenter__ = AsyncMock(side_effect=Exception("connection reset"))
    resp.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=resp)

    canvas = await fetch_radar_grid(session, "/v2/radar/test", 100, 50)
    assert canvas.size[0] > 0
    assert "Failed radar" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_fetch_radar_grid_applies_precipitation_filter():
    """fetch_radar_grid must apply filter_precipitation_only to each tile."""
    # Use a tile with non-precipitation colour - filter should clear it
    non_precip_tile = _make_tile_png(rgba=(0, 255, 0, 255))
    session = _mock_session(tile_bytes=non_precip_tile)

    canvas = await fetch_radar_grid(session, "/v2/radar/test", 100, 50)
    arr = np.array(canvas)
    # All pixels should be transparent (green is not a precip colour)
    assert (arr[:, :, 3] == 0).all(), (
        "fetch_radar_grid must filter non-precipitation pixels"
    )
