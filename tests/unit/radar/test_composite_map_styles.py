"""Tests for map_style integration in composite.py.

TDD: written FIRST (red), then implementation makes them green.

Covers:
- fetch_map_crop defaults to VOYAGER style
- fetch_map_crop passes the style parameter through to the correct fetch_tile
- build_frame_label truncates location names > 30 chars
- build_frame_label leaves names <= 30 chars untouched
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    build_frame_label,
    fetch_map_crop,
)
from custom_components.rain_incoming.radar.map_styles import MapStyle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tile_png(colour=(128, 128, 128, 255)) -> bytes:
    img = Image.new("RGBA", (256, 256), colour)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Style parameter tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_map_crop_default_style_is_voyager():
    """Calling fetch_map_crop without a style arg must use the Voyager tile URL.

    We inspect the URLs passed to session.get and verify the Voyager CartoDB
    URL pattern appears. This proves the default routes to VOYAGER without
    needing to patch into the frozen MapStyleDefinition dataclass.
    """
    from custom_components.rain_incoming.radar.composite import _map_tile_cache

    tile_bytes = _make_tile_png()
    fetched_urls: list[str] = []

    async def _fake_get(url: str, **kwargs):
        fetched_urls.append(url)
        resp = MagicMock()
        resp.status = 200
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        resp.release = MagicMock()
        resp.read = AsyncMock(return_value=tile_bytes)
        return resp

    session = MagicMock()
    session.get = AsyncMock(side_effect=_fake_get)

    _map_tile_cache.clear()
    await fetch_map_crop(
        session=session, lat=-33.7, lon=151.2, radius_km=128,
        # No map_style argument - should default to VOYAGER
    )

    voyager_urls = [u for u in fetched_urls if "basemaps.cartocdn.com/rastertiles/voyager" in u]
    assert voyager_urls, (
        f"Expected at least one CartoDB Voyager URL in fetched URLs, "
        f"but got: {fetched_urls}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style, url_fragment", [
    (MapStyle.VOYAGER, "basemaps.cartocdn.com/rastertiles/voyager"),
    (MapStyle.OSM_STANDARD, "tile.openstreetmap.org"),
    (MapStyle.OSM_DARK, "tile.openstreetmap.org"),
    (MapStyle.ESRI_IMAGERY, "arcgisonline.com"),
])
async def test_fetch_map_crop_passes_style_through(style: MapStyle, url_fragment: str):
    """Passing each MapStyle to fetch_map_crop should route to that style's tile server.

    We inspect the URLs session.get was called with and verify the expected
    tile server URL appears.
    """
    from custom_components.rain_incoming.radar.composite import _map_tile_cache

    _map_tile_cache.clear()
    tile_bytes = _make_tile_png()
    fetched_urls: list[str] = []

    async def _fake_get(url: str, **kwargs):
        fetched_urls.append(url)
        resp = MagicMock()
        resp.status = 200
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        resp.release = MagicMock()
        resp.read = AsyncMock(return_value=tile_bytes)
        return resp

    session = MagicMock()
    session.get = AsyncMock(side_effect=_fake_get)

    _map_tile_cache.clear()
    await fetch_map_crop(
        session=session, lat=-33.7, lon=151.2, radius_km=128,
        map_style=style,
    )

    matching = [u for u in fetched_urls if url_fragment in u]
    assert matching, (
        f"fetch_map_crop with style={style} should fetch from {url_fragment!r}, "
        f"but fetched URLs were: {fetched_urls}"
    )


# ---------------------------------------------------------------------------
# Location name truncation tests
# ---------------------------------------------------------------------------

def test_build_frame_label_truncates_long_location_names():
    """Location names > 30 chars must be truncated to 29 chars + ellipsis."""
    long_name = "A" * 40  # 40 chars - clearly over the limit
    label = build_frame_label(
        timestamp=None, tz_name=None, radius_km=128, location_name=long_name,
    )
    truncated = "A" * 29 + "\u2026"
    assert truncated in label, (
        f"Expected truncated name {truncated!r} in label, got: {label!r}"
    )
    assert long_name not in label, (
        f"Full 40-char name should not appear in label, got: {label!r}"
    )

    # Boundary case: 31 chars is the FIRST length that triggers truncation.
    # Off-by-one bugs live here (e.g. > vs >= in the length check).
    name_31 = "D" * 31
    label_31 = build_frame_label(
        timestamp=None, tz_name=None, radius_km=128, location_name=name_31,
    )
    truncated_31 = "D" * 29 + "\u2026"
    assert truncated_31 in label_31, (
        f"31-char name should be truncated to 29 chars + ellipsis, got: {label_31!r}"
    )
    assert name_31 not in label_31, (
        f"Full 31-char name should not appear in label, got: {label_31!r}"
    )

    # Boundary just below: 30-char name must NOT be truncated
    name_30_boundary = "E" * 30
    label_30 = build_frame_label(
        timestamp=None, tz_name=None, radius_km=128, location_name=name_30_boundary,
    )
    assert name_30_boundary in label_30, (
        f"30-char name (on the boundary) should NOT be truncated, got: {label_30!r}"
    )


def test_build_frame_label_leaves_short_names_intact():
    """Location names <= 30 chars must not be truncated."""
    name_30 = "B" * 30  # exactly 30 chars - on the boundary
    label = build_frame_label(
        timestamp=None, tz_name=None, radius_km=128, location_name=name_30,
    )
    assert name_30 in label, (
        f"30-char name should appear unchanged in label, got: {label!r}"
    )

    name_29 = "C" * 29  # 29 chars - under the limit
    label2 = build_frame_label(
        timestamp=None, tz_name=None, radius_km=128, location_name=name_29,
    )
    assert name_29 in label2, (
        f"29-char name should appear unchanged in label, got: {label2!r}"
    )
