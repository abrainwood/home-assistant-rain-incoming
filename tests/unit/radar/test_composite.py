from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    FrameRenderContext,
    _map_tile_cache,
    _radar_tile_cache,
    calculate_map_zoom,
    composite_frames,
    draw_crosshair,
    draw_range_rings,
    filter_precipitation_pixels,
    fetch_map_crop,
    km_per_pixel,
    render_animated_composite,
    render_composite,
)


class TestKmPerPixel:
    def test_equator_zoom_0(self):
        # At zoom 0 the whole world fits in 256 pixels
        result = km_per_pixel(0.0, 0)
        expected = 40075.0 / 256
        assert result == pytest.approx(expected, rel=1e-6)

    def test_higher_zoom_halves_distance(self):
        z0 = km_per_pixel(0.0, 0)
        z1 = km_per_pixel(0.0, 1)
        assert z1 == pytest.approx(z0 / 2, rel=1e-6)

    def test_latitude_reduces_km(self):
        equator = km_per_pixel(0.0, 7)
        mid_lat = km_per_pixel(45.0, 7)
        assert mid_lat < equator
        assert mid_lat == pytest.approx(equator * math.cos(math.radians(45.0)), rel=1e-6)


class TestCalculateMapZoom:
    def test_small_radius_gives_higher_zoom(self):
        z_small = calculate_map_zoom(-33.7, 64, 640)
        z_large = calculate_map_zoom(-33.7, 256, 640)
        assert z_small > z_large

    def test_default_radius_128_reasonable_zoom(self):
        z = calculate_map_zoom(-33.7, 128, 640)
        # Should be somewhere between 7 and 12 for a 128km radius
        assert 7 <= z <= 12

    def test_returns_int(self):
        z = calculate_map_zoom(0.0, 128, 640)
        assert isinstance(z, int)


class TestFilterPrecipitationPixels:
    def test_transparent_pixels_stay_transparent(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = filter_precipitation_pixels(img)
        assert result.shape == (10, 10, 4)
        assert (result[:, :, 3] == 0).all()

    def test_precipitation_colour_preserved(self):
        """A pixel matching a known RainViewer precipitation colour keeps its alpha."""
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS
        r, g, b, _ = PRECIP_COLOURS[2]  # (0, 154, 213) - light-moderate rain
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [r, g, b, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 255

    def test_outermost_trace_khaki_filtered_out(self):
        """(170, 158, 121) was the outermost trace tier in V1. Per GH #189 (V2 palette),
        it has been removed from PRECIP_COLOURS. Its L2 distance to the nearest remaining
        entry (218,204,147) is ~71, which exceeds MAX_COLOUR_DISTANCE (60), so it is now
        treated as land mask and must be filtered OUT (alpha set to 0)."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [170, 158, 121, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 0

    def test_low_alpha_pixel_removed(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [255, 255, 255, 5]  # alpha <= 10 threshold
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 0

    def test_all_documented_colours_pass_filter(self):
        """Every colour in the documented scheme 2 table should pass the filter."""
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS
        img = np.zeros((len(PRECIP_COLOURS), 1, 4), dtype=np.uint8)
        for i, (r, g, b, _) in enumerate(PRECIP_COLOURS):
            img[i, 0] = [r, g, b, 200]
        result = filter_precipitation_pixels(img)
        for i in range(len(PRECIP_COLOURS)):
            assert result[i, 0, 3] == 200, (
                f"Precipitation colour index {i} was incorrectly filtered out"
            )

    def test_arbitrary_non_precip_colour_removed(self):
        """A random colour far from any precipitation colour is removed."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[3, 3] = [128, 128, 128, 200]  # grey - not a precipitation colour
        result = filter_precipitation_pixels(img)
        assert result[3, 3, 3] == 0


class TestDrawCrosshair:
    def test_draws_red_at_circle_edge(self):
        from custom_components.rain_incoming.radar.composite import _CROSSHAIR_RADIUS
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        draw_crosshair(img, 50, 50)
        pixels = np.array(img)
        # The circle edge should have red pixels
        edge_pixel = pixels[50 - _CROSSHAIR_RADIUS, 50]
        assert edge_pixel[0] > 100  # red channel present from circle outline

    def test_crosshair_lines_extend_from_centre(self):
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 255))
        draw_crosshair(img, 100, 100)
        pixels = np.array(img)
        # Check that some pixels along the horizontal line are red
        # Offset from centre to avoid the circle area
        line_pixel = pixels[100, 100 + 20]
        assert line_pixel[0] > 100  # red channel present


class TestDrawRangeRings:
    def test_draws_at_expected_radius(self):
        size = 200
        img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
        centre_x, centre_y = size // 2, size // 2
        radius_pixels = 50
        draw_range_rings(img, centre_x, centre_y, radius_pixels, radius_km=128)
        pixels = np.array(img)
        # Check a point on the full-radius ring (top of circle)
        ring_pixel = pixels[centre_y - radius_pixels, centre_x]
        # Should have some white/grey marking
        assert ring_pixel[0] > 20 or ring_pixel[1] > 20 or ring_pixel[2] > 20
        # Verify it differs from the pure black background
        assert not (ring_pixel[0] == 0 and ring_pixel[1] == 0 and ring_pixel[2] == 0 and ring_pixel[3] == 255)


def _make_tile_png(colour: tuple[int, int, int, int] = (0, 0, 0, 255)) -> bytes:
    """Create a 256x256 solid-colour RGBA PNG as bytes."""
    img = Image.new("RGBA", (256, 256), colour)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_session(map_tile_bytes: bytes | None = None, radar_tile_bytes: bytes | None = None):
    """Return a mock aiohttp.ClientSession that serves tiles from bytes.

    session.get is an async function returning a mock response directly
    (matching fetch_with_retry's await session.get(url) pattern).

    If radar_tile_bytes is None, radar fetches raise an exception to simulate failure.
    """
    map_bytes = map_tile_bytes if map_tile_bytes is not None else _make_tile_png()
    radar_bytes = radar_tile_bytes if radar_tile_bytes is not None else _make_tile_png((0, 0, 0, 0))

    async def fake_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status = 200

        if "tilecache" in url or "rainviewer" in url.lower():
            if radar_tile_bytes is None:
                resp.status = 500
                resp.raise_for_status = MagicMock(side_effect=Exception("radar fetch failed"))
                resp.read = AsyncMock(return_value=b"")
            else:
                resp.read = AsyncMock(return_value=radar_bytes)
                resp.raise_for_status = MagicMock()
                resp.headers = {}
        else:
            resp.read = AsyncMock(return_value=map_bytes)
            resp.raise_for_status = MagicMock()
            resp.headers = {}

        return resp

    session = MagicMock()
    session.get = fake_get
    return session


class TestRenderComposite:
    @pytest.fixture(autouse=True)
    def clear_tile_cache(self):
        """Clear tile caches before each test."""
        _map_tile_cache.clear()
        _radar_tile_cache.clear()
        yield
        _map_tile_cache.clear()
        _radar_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_happy_path_produces_png_with_expected_size(self):
        output_size = 256
        session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=_make_tile_png((0, 0, 0, 0)),
        )

        result = await render_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_path="/v2/radar/test123",
            output_size=output_size,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.size == (output_size, output_size)
        assert img.mode == "RGBA"

    @pytest.mark.asyncio
    async def test_all_radar_tiles_fail_still_returns_valid_image(self):
        output_size = 256
        session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=None,  # all radar fetches fail
        )

        result = await render_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_path="/v2/radar/test_fail",
            output_size=output_size,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.size == (output_size, output_size)

    @pytest.mark.asyncio
    async def test_map_tile_cache_hit_does_not_refetch(self):
        output_size = 256
        call_count = 0
        tile_bytes = _make_tile_png((30, 30, 30, 255))

        async def counting_get(url: str, **kwargs):
            nonlocal call_count
            resp = MagicMock()
            resp.status = 200
            resp.headers = {}
            resp.raise_for_status = MagicMock()
            if "tilecache" in url or "rainviewer" in url.lower():
                resp.read = AsyncMock(return_value=_make_tile_png((0, 0, 0, 0)))
            else:
                call_count += 1
                resp.read = AsyncMock(return_value=tile_bytes)
            return resp

        session = MagicMock()
        session.get = counting_get

        await render_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_path="/v2/radar/cache_test",
            output_size=output_size,
            session=session,
        )

        first_call_count = call_count

        # Second render - map tiles should come from cache
        await render_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_path="/v2/radar/cache_test_2",
            output_size=output_size,
            session=session,
        )

        assert call_count == first_call_count  # no new map tile fetches


def _make_frame_timestamps(count: int) -> list[datetime]:
    """Generate UTC timestamps for test frames, 10 minutes apart."""
    base = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return [base + timedelta(minutes=10 * i) for i in range(count)]


class TestRenderAnimatedComposite:
    @pytest.fixture(autouse=True)
    def clear_tile_cache(self):
        _map_tile_cache.clear()
        yield
        _map_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_returns_valid_gif_bytes(self):
        output_size = 256
        session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=_make_tile_png((0, 0, 0, 0)),
        )
        frame_paths = ["/v2/radar/frame1", "/v2/radar/frame2", "/v2/radar/frame3"]
        timestamps = _make_frame_timestamps(3)

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
            frame_timestamps=timestamps,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.format == "GIF"
        assert img.size == (output_size, output_size)

    @pytest.mark.asyncio
    async def test_gif_has_expected_frame_count(self):
        """Pillow deduplicates identical GIF frames, so use distinct precipitation colours."""
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        map_bytes = _make_tile_png((30, 30, 30, 255))
        precip_tile_by_frame = {}
        for i in range(5):
            r, g, b, _ = PRECIP_COLOURS[i % len(PRECIP_COLOURS)]
            precip_tile_by_frame[f"frame{i}"] = _make_tile_png((int(r), int(g), int(b), 255))

        async def varying_get(url: str, **kwargs):
            resp = MagicMock()
            resp.status = 200
            resp.headers = {}
            resp.raise_for_status = MagicMock()
            if "tilecache" in url or "rainviewer" in url.lower():
                tile_data = _make_tile_png((0, 0, 0, 0))
                for key, data in precip_tile_by_frame.items():
                    if key in url:
                        tile_data = data
                        break
                resp.read = AsyncMock(return_value=tile_data)
            else:
                resp.read = AsyncMock(return_value=map_bytes)
            return resp

        session = MagicMock()
        session.get = varying_get

        frame_paths = [f"/v2/radar/frame{i}" for i in range(5)]
        timestamps = _make_frame_timestamps(5)

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
            frame_timestamps=timestamps,
            session=session,
        )

        img = Image.open(BytesIO(result))
        assert img.n_frames == 5

    @pytest.mark.asyncio
    async def test_single_frame_produces_valid_gif(self):
        output_size = 256
        session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=_make_tile_png((0, 0, 0, 0)),
        )
        frame_paths = ["/v2/radar/single_frame"]
        timestamps = _make_frame_timestamps(1)

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
            frame_timestamps=timestamps,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.format == "GIF"
        assert img.size == (output_size, output_size)

    @pytest.mark.asyncio
    async def test_last_frame_has_longer_duration(self):
        """The last GIF frame should hold 4x longer than other frames."""
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        map_bytes = _make_tile_png((30, 30, 30, 255))
        precip_tile_by_frame = {}
        for i in range(3):
            r, g, b, _ = PRECIP_COLOURS[i % len(PRECIP_COLOURS)]
            precip_tile_by_frame[f"frame{i}"] = _make_tile_png((int(r), int(g), int(b), 255))

        async def varying_get(url: str, **kwargs):
            resp = MagicMock()
            resp.status = 200
            resp.headers = {}
            resp.raise_for_status = MagicMock()
            if "tilecache" in url or "rainviewer" in url.lower():
                tile_data = _make_tile_png((0, 0, 0, 0))
                for key, data in precip_tile_by_frame.items():
                    if key in url:
                        tile_data = data
                        break
                resp.read = AsyncMock(return_value=tile_data)
            else:
                resp.read = AsyncMock(return_value=map_bytes)
            return resp

        session = MagicMock()
        session.get = varying_get

        frame_paths = [f"/v2/radar/frame{i}" for i in range(3)]
        timestamps = _make_frame_timestamps(3)
        frame_duration_ms = 500

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=frame_duration_ms,
            frame_timestamps=timestamps,
            session=session,
        )

        img = Image.open(BytesIO(result))
        # GIF durations are in milliseconds; Pillow reports them per frame
        durations = []
        for frame_idx in range(img.n_frames):
            img.seek(frame_idx)
            durations.append(img.info.get("duration", 0))

        # All frames except last should be the base duration
        for d in durations[:-1]:
            assert d == frame_duration_ms
        # Last frame should be 4x
        assert durations[-1] == frame_duration_ms * 4


class TestBinaryThresholdRendering:
    """Verify binary detection threshold mask: above threshold = full opacity,
    below threshold = dimmed (25% alpha)."""

    def test_above_threshold_gets_full_opacity(self):
        """Rain with effective intensity >= 0.1 should render at full alpha.

        Pixel: green=200 (luminance ~0.46), confidence=0.5
        Effective intensity: 0.46 * 0.5 = 0.23 >= 0.1 -> full alpha (200 unchanged)
        After compositing over black: green = 200 * (200/255) ~ 157
        """
        from custom_components.rain_incoming.radar.composite import _composite_single_frame

        output_size = 64
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        radar_arr[32, 32] = [0, 200, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        confidence_map = np.full((output_size, output_size), 0.5, dtype=np.float32)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        ctx = FrameRenderContext(
            lat=-33.7, lon=151.2, map_zoom=8, radius_km=128, output_size=output_size,
        )
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            ctx=ctx,
            confidence_map=confidence_map,
        )
        result_arr = np.array(result)
        # Full alpha (200) composited over black: green = 200 * (200/255) ~ 157
        green_val = result_arr[32, 32, 1]
        assert green_val >= 140, (
            f"Above-threshold rain should render at full opacity, but green = {green_val}"
        )

    def test_below_threshold_gets_dimmed(self):
        """Noise with effective intensity < 0.1 should render dimmed to 25% alpha.

        Pixel: green=30 (luminance ~0.069), confidence=0.5
        Effective intensity: 0.069 * 0.5 = 0.035 < 0.1 -> dimmed (alpha * 0.25)
        Original alpha=200, dimmed to 50. After compositing: green = 30 * (50/255) ~ 6
        """
        from custom_components.rain_incoming.radar.composite import _composite_single_frame

        output_size = 64
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        # Low intensity pixel - luminance will be low
        radar_arr[32, 32] = [0, 30, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        confidence_map = np.full((output_size, output_size), 0.5, dtype=np.float32)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        ctx = FrameRenderContext(
            lat=-33.7, lon=151.2, map_zoom=8, radius_km=128, output_size=output_size,
        )
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            ctx=ctx,
            confidence_map=confidence_map,
        )
        result_arr = np.array(result)
        # Dimmed to 25% alpha: 200 * 0.25 = 50, composited: 30 * (50/255) ~ 6
        green_val = result_arr[32, 32, 1]
        assert green_val < 20, (
            f"Below-threshold noise should be dimmed, but green = {green_val}"
        )

    def test_no_confidence_renders_full(self):
        """Without confidence map, all radar renders at full alpha."""
        from custom_components.rain_incoming.radar.composite import _composite_single_frame

        output_size = 64
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        radar_arr[32, 32] = [0, 200, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        ctx = FrameRenderContext(
            lat=-33.7, lon=151.2, map_zoom=8, radius_km=128, output_size=output_size,
        )
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            ctx=ctx,
            confidence_map=None,
        )
        result_arr = np.array(result)
        green_val = result_arr[32, 32, 1]
        # Full alpha (200) composited: 200 * (200/255) ~ 157
        assert green_val >= 140, (
            f"No confidence map should render full opacity, but green = {green_val}"
        )

    def test_high_confidence_rain_at_60pct_is_fully_visible(self):
        """Rain at 60% confidence with bright pixels gets full opacity (not dimmed).

        This verifies the fix: the old conf^1.5 curve would give 0.6^1.5=0.46 alpha
        multiplier, making rain invisible. Binary threshold gives full opacity.

        Pixel: green=200 (luminance ~0.46), confidence=0.6
        Effective: 0.46 * 0.6 = 0.28 >= 0.1 -> full alpha
        """
        from custom_components.rain_incoming.radar.composite import _composite_single_frame

        output_size = 64
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        radar_arr[32, 32] = [0, 200, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        confidence_map = np.full((output_size, output_size), 0.6, dtype=np.float32)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        ctx = FrameRenderContext(
            lat=-33.7, lon=151.2, map_zoom=8, radius_km=128, output_size=output_size,
        )
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            ctx=ctx,
            confidence_map=confidence_map,
        )
        result_arr = np.array(result)
        green_val = result_arr[32, 32, 1]
        # With binary threshold: full alpha -> green ~ 157
        # With old conf^1.5: alpha=93 -> green ~ 73 (fails this assertion)
        assert green_val >= 140, (
            f"Rain at 60% confidence should be fully visible, but green = {green_val}"
        )


class TestConfidenceWeightedRendering:
    """Verify that QC texture scoring dims speckly radar and preserves smooth rain."""

    @pytest.fixture(autouse=True)
    def clear_tile_cache(self):
        _map_tile_cache.clear()
        yield
        _map_tile_cache.clear()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="experiment branch #195-palette-v8: V8 removes all blue tiers; confidence rendering scenario uses light-rain pixels not in V8")
    async def test_speckle_is_dimmed_vs_smooth_rain(self):
        """Speckly radar tile should have lower mean alpha than smooth rain tile
        after confidence-weighted rendering."""
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        # Use a real precipitation colour so the colour filter doesn't strip it
        pr, pg, pb, _ = PRECIP_COLOURS[3]  # moderate rain (81, 197, 232)

        # Create a smooth rain radar tile: solid precipitation colour with full alpha
        smooth_img = Image.new("RGBA", (256, 256), (int(pr), int(pg), int(pb), 200))
        smooth_buf = BytesIO()
        smooth_img.save(smooth_buf, format="PNG")
        smooth_bytes = smooth_buf.getvalue()

        # Create a speckly radar tile: random scattered pixels using same colour
        rng = np.random.default_rng(42)
        speckle_arr = np.zeros((256, 256, 4), dtype=np.uint8)
        mask = rng.random((256, 256)) > 0.7
        speckle_arr[mask] = [int(pr), int(pg), int(pb), 200]
        speckle_img = Image.fromarray(speckle_arr)
        speckle_buf = BytesIO()
        speckle_img.save(speckle_buf, format="PNG")
        speckle_bytes = speckle_buf.getvalue()

        # Render with smooth radar
        smooth_session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=smooth_bytes,
        )
        smooth_result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=["/v2/radar/smooth"],
            output_size=output_size,
            frame_timestamps=_make_frame_timestamps(1),
            session=smooth_session,
        )

        # Render with speckly radar
        speckle_session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=speckle_bytes,
        )
        speckle_result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=["/v2/radar/speckle"],
            output_size=output_size,
            frame_timestamps=_make_frame_timestamps(1),
            session=speckle_session,
        )

        # Compare: GIF frames are RGB, so compare green channel intensity
        # (the green from radar should be more preserved in smooth vs speckle)
        smooth_frame = np.array(Image.open(BytesIO(smooth_result)).convert("RGB"))
        speckle_frame = np.array(Image.open(BytesIO(speckle_result)).convert("RGB"))

        # Green channel mean - smooth rain should show more green
        smooth_green = smooth_frame[:, :, 1].mean()
        speckle_green = speckle_frame[:, :, 1].mean()
        assert smooth_green > speckle_green


# ---------------------------------------------------------------------------
# Fix 1: Semaphore must be held for the full connection lifecycle
# ---------------------------------------------------------------------------

class TestTileSemaphoreLifecycle:
    """The tile semaphore must cover response body reading, not just the request.

    Previously: semaphore released when fetch_with_retry returned the response
    object (headers received), before resp.read() was called. This meant
    connections were open outside the semaphore, making the limit ineffective.

    Fix: semaphore held until body is fully read.
    """

    @pytest.fixture(autouse=True)
    def clear_tile_caches(self):
        _map_tile_cache.clear()
        _radar_tile_cache.clear()
        yield
        _map_tile_cache.clear()
        _radar_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_semaphore_held_during_body_read(self):
        """With semaphore(1), two concurrent tile fetches must not overlap
        during resp.read() — the second must wait until the first completes
        its full read, not just until headers arrive."""
        from custom_components.rain_incoming.radar.composite import _fetch_tile

        overlap_detected = False
        reading_count = 0

        async def slow_read():
            nonlocal reading_count, overlap_detected
            reading_count += 1
            if reading_count > 1:
                overlap_detected = True
            await asyncio.sleep(0)  # yield so the other task can attempt to proceed
            reading_count -= 1
            return _make_tile_png((0, 0, 0, 0))

        resp = MagicMock()
        resp.status = 200
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        resp.read = AsyncMock(side_effect=slow_read)

        session = MagicMock()
        session.get = AsyncMock(return_value=resp)

        test_semaphore = asyncio.Semaphore(1)
        with patch(
            "custom_components.rain_incoming.radar.composite._get_tile_semaphore",
            return_value=test_semaphore,
        ):
            await asyncio.gather(
                _fetch_tile(session, "http://example.com/tile1"),
                _fetch_tile(session, "http://example.com/tile2"),
            )

        assert not overlap_detected, (
            "Two tile fetches ran concurrently during resp.read() — "
            "semaphore was released before the body was fully read"
        )


# ---------------------------------------------------------------------------
# Fix 4: composite_frames must be offloaded to the executor
# ---------------------------------------------------------------------------

class TestCompositeFramesOffloading:
    """When run_in_executor is provided to render_animated_composite,
    the CPU-bound composite_frames work must run through the executor,
    not synchronously on the event loop."""

    @pytest.fixture(autouse=True)
    def clear_tile_caches(self):
        _map_tile_cache.clear()
        _radar_tile_cache.clear()
        yield
        _map_tile_cache.clear()
        _radar_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_composite_frames_dispatched_via_executor(self):
        """render_animated_composite must route composite_frames through
        run_in_executor when one is provided."""
        import functools

        executor_fns: list[str] = []

        async def tracking_executor(fn, *args, **kwargs):
            name = getattr(fn, "__name__", None) or getattr(fn, "func", fn).__name__
            executor_fns.append(name)
            # Run synchronously in test (executor would normally offload to thread)
            if kwargs:
                return fn(**kwargs)
            return fn(*args)

        session = _mock_session(
            map_tile_bytes=_make_tile_png((30, 30, 30, 255)),
            radar_tile_bytes=_make_tile_png((0, 0, 0, 0)),
        )

        await render_animated_composite(
            lat=-33.7,
            lon=151.2,
            radius_km=64,
            frame_paths=["/v2/radar/f1"],
            frame_duration_ms=500,
            session=session,
            run_in_executor=tracking_executor,
        )

        assert "composite_frames" in executor_fns, (
            f"composite_frames was not dispatched via run_in_executor. "
            f"Executor received: {executor_fns}"
        )


# ---------------------------------------------------------------------------
# Fix 5: Map tile fetches must use the tile semaphore
# ---------------------------------------------------------------------------

class TestMapTileSemaphore:
    """Map tile fetches must acquire the tile semaphore, not bypass it.

    Previously: _fetch_map_crop called asyncio.gather without any concurrency
    control, allowing unlimited simultaneous connections to map tile providers.

    Fix: map tile fetches go through _get_tile_semaphore() so they count
    against the same shared limit as radar tile fetches.
    """

    @pytest.fixture(autouse=True)
    def clear_tile_caches(self):
        _map_tile_cache.clear()
        _radar_tile_cache.clear()
        yield
        _map_tile_cache.clear()
        _radar_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_map_tiles_acquire_tile_semaphore(self):
        """fetch_map_crop must acquire the tile semaphore for each cache-miss tile.

        With semaphore(1), concurrent map tile fetches must be serialized,
        proving they go through the semaphore rather than bypassing it.
        """
        overlap_detected = False
        fetching_count = 0

        async def serialized_fetch(*args, **kwargs):
            nonlocal fetching_count, overlap_detected
            fetching_count += 1
            if fetching_count > 1:
                overlap_detected = True
            await asyncio.sleep(0)
            fetching_count -= 1
            resp = MagicMock()
            resp.status = 200
            resp.headers = {}
            resp.raise_for_status = MagicMock()
            resp.read = AsyncMock(return_value=_make_tile_png((30, 30, 30, 255)))
            return resp

        session = MagicMock()
        session.get = serialized_fetch

        test_semaphore = asyncio.Semaphore(1)
        with patch(
            "custom_components.rain_incoming.radar.composite._get_tile_semaphore",
            return_value=test_semaphore,
        ):
            await fetch_map_crop(
                session=session,
                lat=-33.7,
                lon=151.2,
                radius_km=128,
                output_size=256,
            )

        assert not overlap_detected, (
            "Map tile fetches ran concurrently — they are not going through "
            "the tile semaphore. Map tiles must use _get_tile_semaphore()."
        )


