from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aiohttp import ClientError

from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import (
    PRECIP_COLOURS,
    RainViewerFrame,
    RainViewerProvider,
    _colour_to_intensity,
    _resolve_url,
    _tile_bounds,
    check_coverage,
)
from custom_components.rain_incoming.radar.geo import lat_lon_to_tile


# --- Palette structural invariants ---

class TestPrecipColoursOrdering:
    def test_intensities_monotonically_increasing_by_index(self):
        """Every PRECIP_COLOURS entry must have strictly higher intensity than its
        predecessor. Catches silent drift where someone inserts a colour at the wrong
        position - which is exactly how the bug behind GH #180 happened.
        """
        intensities = [intensity for _, _, _, intensity in PRECIP_COLOURS]
        for i in range(1, len(intensities)):
            assert intensities[i] > intensities[i - 1], (
                f"PRECIP_COLOURS[{i}] intensity {intensities[i]} not greater than "
                f"PRECIP_COLOURS[{i-1}] intensity {intensities[i-1]} - palette must be "
                f"strictly monotonic by index."
            )


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

@pytest.mark.skip(reason="experiment branch #195-palette-v7: tests pin V2 production palette; this branch deliberately uses V7 layout (no trace, no cyan, no light/medium blue)")
class TestColourToIntensity:
    def test_transparent_pixel_is_zero(self):
        assert _colour_to_intensity(0, 0, 0, alpha=0) == pytest.approx(0.0)

    def test_outermost_trace_khaki_returns_zero_intensity(self):
        # (170, 158, 121) was the outermost khaki trace tier in V1 (12-entry palette).
        # Per GH #189: V2 drops the outer two trace tiers (confirmed by long-backtest
        # analysis, #188). They added false positives without meaningful detections.
        # With only the inner trace (218,204,147) remaining, (170,158,121) is now ~71
        # L2 from the nearest palette entry - exceeds MAX_COLOUR_DISTANCE (60) so it
        # returns 0.0 and is treated as land mask / non-precipitation.
        intensity = _colour_to_intensity(170, 158, 121, alpha=255)
        assert intensity == pytest.approx(0.0)

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

    def test_cell_core_blue_higher_intensity_than_cell_halo_cyan(self):
        # (0, 154, 213) appears at cell cores (high intensity);
        # (81, 197, 232) appears at cell halos (lower intensity).
        # Spatial radial traces through real captured radar tiles confirm ordering.
        core_blue = _colour_to_intensity(0, 154, 213, alpha=255)
        halo_cyan = _colour_to_intensity(81, 197, 232, alpha=255)
        assert core_blue > halo_cyan

    def test_only_inner_trace_tier_is_a_palette_entry(self):
        # Per GH #189 (V2 palette): only the innermost trace tier remains in
        # PRECIP_COLOURS. The outer two were dropped after long-backtest analysis
        # (#188) showed they added false positives without meaningful detections.
        #
        # Inner tier (218,204,147) must be present.
        # Middle (206,192,135) and outer (170,158,121) must NOT be present.
        # Palette length must be exactly 10 (locks size to prevent silent drift).
        rgb_entries = {(r, g, b) for r, g, b, _ in PRECIP_COLOURS}
        assert (218, 204, 147) in rgb_entries, "(218,204,147) inner trace must remain in PRECIP_COLOURS"
        assert (170, 158, 121) not in rgb_entries, "(170,158,121) outer trace must be removed per #189"
        assert (206, 192, 135) not in rgb_entries, "(206,192,135) middle trace must be removed per #189"
        assert len(PRECIP_COLOURS) == 10, (
            f"PRECIP_COLOURS should have 10 entries after V2 drop, got {len(PRECIP_COLOURS)}"
        )

    def test_blue_tier_ordered_dark_to_cyan(self):
        # RainViewer Universal Blue scheme 2: darker blue means heavier precipitation.
        # Cell-core to cell-halo ordering (inferred from radial trace evidence + standard
        # radar convention): dark blue > medium blue > light blue > bright cyan.
        # The highest trace tier (218,204,147 at 0.09) must sit below bright cyan.
        dark = _colour_to_intensity(0, 91, 142, alpha=255)
        medium = _colour_to_intensity(0, 119, 170, alpha=255)
        light = _colour_to_intensity(0, 154, 213, alpha=255)
        cyan = _colour_to_intensity(81, 197, 232, alpha=255)
        inner_trace = _colour_to_intensity(218, 204, 147, alpha=255)
        assert inner_trace < cyan < light < medium < dark


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

    @pytest.mark.asyncio
    async def test_get_frames_creates_fallback_session_when_none_provided(self):
        """When no session is provided, _fetch_manifest must create its own."""
        provider = RainViewerProvider()

        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=MANIFEST)

        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            new=AsyncMock(return_value=manifest_resp),
        ) as mock_fetch:
            frames = await provider.get_frames(-33.701, 151.209, count=2)

        assert len(frames) == 2
        mock_fetch.assert_called_once()
        # The session arg should be an aiohttp.ClientSession (created internally),
        # not None
        call_args = mock_fetch.call_args
        assert call_args.args[0] is not None, (
            "When no session is provided, _fetch_manifest must create a fallback session"
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

        # check_coverage fetches manifest first (sequential), then probes
        # tiles concurrently. Use a function side_effect that returns the
        # manifest for the first call and tile responses for the rest.
        call_count = 0

        async def _fake_fetch(session, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return manifest_resp
            return tile_resp

        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            side_effect=_fake_fetch,
        ):
            result = await check_coverage(-33.701, 151.209, session=mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_all_tiles_transparent(self):
        manifest_resp = AsyncMock()
        manifest_resp.json = AsyncMock(return_value=MANIFEST)

        tile_resp = AsyncMock()
        tile_resp.read = AsyncMock(return_value=_make_tile_png(has_precip=False))

        call_count = 0

        async def _fake_fetch(session, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return manifest_resp
            return tile_resp

        mock_session = MagicMock()
        with patch(
            "custom_components.rain_incoming.providers.rainviewer.fetch_with_retry",
            side_effect=_fake_fetch,
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


# ---------------------------------------------------------------------------
# _tile_to_intensity_array: unique colour optimisation
# ---------------------------------------------------------------------------


class TestTileToIntensityArrayPerformance:
    """The colour matching must use unique-colour lookup, not per-pixel L2."""

    def test_few_unique_colours_is_fast(self):
        """A tile with few unique colours (typical) should complete in under 3ms.

        The old per-pixel L2 approach took ~15ms. The unique-colour lookup
        should be ~1ms by matching only the ~7 unique colours instead of
        broadcasting across all 65,536 pixels.
        """
        import time
        from io import BytesIO
        from PIL import Image
        from custom_components.rain_incoming.providers.rainviewer import (
            _tile_to_intensity_array,
            PRECIP_COLOURS,
        )

        # Create a tile with 3 known precipitation colours (typical real-world tile)
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        r, g, b, _ = PRECIP_COLOURS[0]
        # Fill top third with one colour
        for y in range(85):
            for x in range(256):
                img.putpixel((x, y), (r, g, b, 200))
        buf = BytesIO()
        img.save(buf, format="PNG")
        tile_bytes = buf.getvalue()

        # Warm up
        _tile_to_intensity_array(tile_bytes)

        # Time 10 runs
        start = time.perf_counter()
        for _ in range(10):
            _tile_to_intensity_array(tile_bytes)
        avg_ms = (time.perf_counter() - start) / 10 * 1000

        assert avg_ms < 3.0, (
            f"_tile_to_intensity_array took {avg_ms:.1f}ms per call, "
            f"expected <3ms with unique-colour optimisation"
        )


# ---------------------------------------------------------------------------
# _tile_to_intensity_array: semantic / spatial ordering
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="experiment branch #195-palette-v7: V7 removes the cyan + light/medium blue tiers used as spatial sentinels")
class TestTileToIntensityArraySemantic:
    """Full pipeline test: real captured tile -> intensity grid -> spatial ordering.

    Loads a 256x256 tile from the golden_v2 fixture set (a captured RainViewer
    radar tile, scheme 2). Asserts that `_tile_to_intensity_array` assigns a
    higher intensity value to a pixel we independently verified is cell-core blue
    (0, 154, 213) than to a pixel we independently verified is cell-halo cyan
    (81, 197, 232).

    This test is pinned to specific pixel coordinates confirmed via PIL before
    being committed. It catches semantic regressions where the palette ordering
    unit tests pass but the full tile->intensity pipeline is broken.

    Fixture: tests/fixtures/golden_v2/Canberra/bronze/tiles/1775715600/7_115_77_s2.png
    Pixel coords confirmed by scanning the RGBA array with PIL:
      core_blue  (0, 154, 213) at row=0, col=24
      halo_cyan  (81, 197, 232) at row=0, col=5
    """

    def test_captured_tile_intensity_matches_spatial_ordering(self):
        import pathlib
        from custom_components.rain_incoming.providers.rainviewer import (
            _tile_to_intensity_array,
            _colour_to_intensity,
        )

        fixture = (
            pathlib.Path(__file__).parent.parent.parent
            / "fixtures"
            / "golden_v2"
            / "Canberra"
            / "bronze"
            / "tiles"
            / "1775715600"
            / "7_115_77_s2.png"
        )
        tile_bytes = fixture.read_bytes()
        grid = _tile_to_intensity_array(tile_bytes)

        # Independently confirmed pixel coordinates (see class docstring)
        core_blue_intensity = float(grid[0, 24])   # (0, 154, 213) - cell core
        halo_cyan_intensity = float(grid[0, 5])    # (81, 197, 232) - cell halo

        # Both pixels must map to nonzero intensity (they're real precipitation)
        assert core_blue_intensity > 0.0, (
            f"core_blue pixel produced zero intensity ({core_blue_intensity}); "
            "expected nonzero - palette match may be broken"
        )
        assert halo_cyan_intensity > 0.0, (
            f"halo_cyan pixel produced zero intensity ({halo_cyan_intensity}); "
            "expected nonzero - palette match may be broken"
        )

        # Cell core must register higher intensity than cell halo (radial ordering)
        assert core_blue_intensity > halo_cyan_intensity, (
            f"core_blue intensity {core_blue_intensity:.4f} not greater than "
            f"halo_cyan intensity {halo_cyan_intensity:.4f}; "
            "spatial radial ordering is violated - this is the GH #180 class of bug"
        )

        # Pin actual values so the test fails loudly if the palette shifts
        assert core_blue_intensity == pytest.approx(
            _colour_to_intensity(0, 154, 213, alpha=255), abs=1e-5
        ), "core_blue tile value diverged from _colour_to_intensity reference"
        assert halo_cyan_intensity == pytest.approx(
            _colour_to_intensity(81, 197, 232, alpha=255), abs=1e-5
        ), "halo_cyan tile value diverged from _colour_to_intensity reference"
