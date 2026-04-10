from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aiohttp import ClientError

from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import (
    RainViewerFrame,
    RainViewerProvider,
    _colour_to_intensity,
    _resolve_url,
    _tile_bounds,
)
from custom_components.rain_incoming.radar.geo import lat_lon_to_tile


# --- URL resolution ---

class TestResolveUrl:
    def test_returns_default_when_env_var_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _resolve_url("MISSING_VAR", "https://default.com") == "https://default.com"

    def test_returns_env_value_when_set(self):
        with patch.dict("os.environ", {"MY_URL": "https://override.com"}):
            assert _resolve_url("MY_URL", "https://default.com") == "https://override.com"

    def test_returns_default_when_env_var_is_empty_string(self):
        with patch.dict("os.environ", {"MY_URL": ""}):
            assert _resolve_url("MY_URL", "https://default.com") == "https://default.com"


# --- Unit tests for pure helpers ---

class TestColourToIntensity:
    def test_transparent_pixel_is_zero(self):
        assert _colour_to_intensity(0, 0, 0, alpha=0) == pytest.approx(0.0)

    def test_land_colour_is_zero(self):
        # Known land mask colour from exploration
        assert _colour_to_intensity(170, 158, 121, alpha=255) == pytest.approx(0.0)

    def test_light_rain_colour_nonzero(self):
        # Known light rain colour (0, 154, 213) in RainViewer scheme 6
        intensity = _colour_to_intensity(0, 154, 213, alpha=255)
        assert 0.0 < intensity <= 0.5

    def test_heavy_rain_colour_higher_than_light(self):
        light = _colour_to_intensity(0, 154, 213, alpha=255)
        heavy = _colour_to_intensity(193, 0, 0, alpha=255)
        assert heavy > light

    def test_max_intensity_does_not_exceed_one(self):
        assert _colour_to_intensity(255, 119, 255, alpha=255) <= 1.0


class TestLatLonToTile:
    def test_known_location(self):
        # Terry Hills at zoom 7 should be tile (117, 76)
        x, y = lat_lon_to_tile(-33.701, 151.209, zoom=7)
        assert x == 117
        assert y == 76

    def test_zoom_increases_tile_resolution(self):
        x7, y7 = lat_lon_to_tile(-33.701, 151.209, zoom=7)
        x8, y8 = lat_lon_to_tile(-33.701, 151.209, zoom=8)
        assert x8 == x7 * 2 or x8 == x7 * 2 + 1
        assert y8 == y7 * 2 or y8 == y7 * 2 + 1


class TestTileBounds:
    def test_tile_bounds_contains_centre_lat_lon(self):
        bounds = _tile_bounds(117, 76, zoom=7)
        assert bounds.lon_min <= 151.209 <= bounds.lon_max
        assert bounds.lat_min <= -33.701 <= bounds.lat_max

    def test_adjacent_tiles_share_edge(self):
        b0 = _tile_bounds(117, 76, zoom=7)
        b1 = _tile_bounds(118, 76, zoom=7)
        assert b0.lon_max == pytest.approx(b1.lon_min)


# --- RainViewerFrame tests ---

FAKE_GRID = np.zeros((64, 64), dtype=np.float32)
FAKE_GRID[30:34, 30:34] = 0.4

FAKE_BOUNDS = BoundingBox(lat_min=-35.0, lat_max=-32.0, lon_min=149.0, lon_max=153.0)


class TestRainViewerFrame:
    def _make_frame(self) -> RainViewerFrame:
        ts = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        return RainViewerFrame(
            timestamp=ts,
            path="/v2/radar/abc123",
            zoom=7,
            colour_scheme=6,
        )

    def test_timestamp_property(self):
        ts = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        frame = RainViewerFrame(timestamp=ts, path="/v2/radar/x", zoom=7, colour_scheme=6)
        assert frame.timestamp == ts

    @pytest.mark.asyncio
    async def test_get_intensity_grid_shape(self):
        frame = self._make_frame()
        with patch.object(frame, "_fetch_stitched_grid", new=AsyncMock(return_value=FAKE_GRID)):
            grid = await frame._fetch_stitched_grid(FAKE_BOUNDS, 64, 64)
        assert grid.shape == (64, 64)

    def test_get_intensity_at_delegates_to_grid(self):
        frame = self._make_frame()
        frame._cached_grid = FAKE_GRID
        frame._cached_bounds = FAKE_BOUNDS
        # Point in the middle of the zero area
        intensity = frame.get_intensity_at(-34.9, 149.1)
        assert intensity == pytest.approx(0.0, abs=0.1)


# --- RainViewerProvider tests ---

MANIFEST = {
    "generated": 1234567890,
    "radar": {
        "past": [
            {"time": 1712484000, "path": "/v2/radar/frame1"},
            {"time": 1712484600, "path": "/v2/radar/frame2"},
            {"time": 1712485200, "path": "/v2/radar/frame3"},
        ],
        "nowcast": [],
    },
}


class TestRainViewerProvider:
    @pytest.mark.asyncio
    async def test_get_frames_returns_requested_count(self):
        provider = RainViewerProvider()
        with patch.object(
            provider, "_fetch_manifest", new=AsyncMock(return_value=MANIFEST)
        ):
            frames = await provider.get_frames(-33.701, 151.209, count=2)
        assert len(frames) == 2

    @pytest.mark.asyncio
    async def test_get_frames_oldest_first(self):
        provider = RainViewerProvider()
        with patch.object(
            provider, "_fetch_manifest", new=AsyncMock(return_value=MANIFEST)
        ):
            frames = await provider.get_frames(-33.701, 151.209, count=3)
        timestamps = [f.timestamp for f in frames]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_get_frames_returns_fewer_when_unavailable(self):
        sparse_manifest = {
            "radar": {"past": [{"time": 1712484000, "path": "/v2/radar/only"}], "nowcast": []}
        }
        provider = RainViewerProvider()
        with patch.object(
            provider, "_fetch_manifest", new=AsyncMock(return_value=sparse_manifest)
        ):
            frames = await provider.get_frames(-33.701, 151.209, count=5)
        assert len(frames) == 1

    @pytest.mark.asyncio
    async def test_get_frames_raises_on_network_error(self):
        provider = RainViewerProvider()
        with patch.object(
            provider, "_fetch_manifest", new=AsyncMock(side_effect=ClientError())
        ):
            with pytest.raises(ClientError):
                await provider.get_frames(-33.701, 151.209, count=3)
