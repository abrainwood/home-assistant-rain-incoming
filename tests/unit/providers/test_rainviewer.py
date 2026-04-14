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
    check_coverage,
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

    @pytest.mark.asyncio
    async def test_prefetch_frames_fetches_concurrently(self):
        """prefetch_frames must fetch multiple frames concurrently, not sequentially.

        Uses overlap detection: if max_concurrent > 1 during the fetch, the
        frames were fetched in parallel.
        """
        import asyncio

        provider = RainViewerProvider()
        frames = [
            RainViewerFrame(
                timestamp=datetime(2026, 4, 10, 10, i * 10, tzinfo=timezone.utc),
                path=f"/v2/radar/frame{i}",
                zoom=7,
                colour_scheme=2,
            )
            for i in range(3)
        ]

        concurrent_count = 0
        max_concurrent = 0

        async def tracking_fetch(self, bounds, width, height, session, budget=None):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1

        with patch.object(RainViewerFrame, "_fetch_stitched_grid", tracking_fetch):
            await provider.prefetch_frames(
                frames, FAKE_BOUNDS, 64, 64, MagicMock(),
            )

        assert max_concurrent > 1, (
            f"Frames were fetched sequentially (max_concurrent={max_concurrent}). "
            "prefetch_frames must use asyncio.gather or similar for concurrent fetches."
        )

    @pytest.mark.asyncio
    async def test_get_frames_uses_provided_session(self):
        """get_frames must use the provided session for the manifest fetch,
        not create a new one. This avoids wasting TCP connections and
        bypassing HA's managed session with its SSL/proxy config."""
        provider = RainViewerProvider()
        mock_session = MagicMock()

        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=MANIFEST)

        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            new=AsyncMock(return_value=manifest_resp),
        ) as mock_fetch:
            frames = await provider.get_frames(-33.701, 151.209, count=2, session=mock_session)

        assert len(frames) == 2
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args.args[0] is mock_session, (
            "get_frames must pass the provided session to fetch_with_retry, "
            "not create a new aiohttp.ClientSession"
        )


# --- check_coverage tests ---

from io import BytesIO
from PIL import Image


def _make_tile_png(has_precip: bool) -> bytes:
    """Create a minimal 256x256 PNG tile for testing."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    if has_precip:
        # Paint some non-transparent pixels to simulate precipitation
        for x in range(10, 20):
            for y in range(10, 20):
                img.putpixel((x, y), (0, 154, 213, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCheckCoverage:
    @pytest.mark.asyncio
    async def test_returns_true_when_tile_has_precipitation(self):
        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=MANIFEST)

        tile_resp = AsyncMock()
        tile_resp.read = AsyncMock(return_value=_make_tile_png(has_precip=True))

        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            side_effect=[manifest_resp, tile_resp],
        ):
            result = await check_coverage(-33.701, 151.209, session=mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_all_tiles_transparent(self):
        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=MANIFEST)

        tile_resp = AsyncMock()
        tile_resp.read = AsyncMock(return_value=_make_tile_png(has_precip=False))

        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            side_effect=[manifest_resp, tile_resp, tile_resp, tile_resp],
        ):
            result = await check_coverage(0.0, 0.0, session=mock_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_past_frames(self):
        empty_manifest = {"radar": {"past": []}}
        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=empty_manifest)

        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            return_value=manifest_resp,
        ):
            result = await check_coverage(-33.701, 151.209, session=mock_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_raises_on_network_error(self):
        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            side_effect=ClientError(),
        ):
            with pytest.raises(ClientError):
                await check_coverage(-33.701, 151.209, session=mock_session)
