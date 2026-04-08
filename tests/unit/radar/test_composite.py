from __future__ import annotations

import math
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from custom_components.incoming_rain.radar.composite import (
    _map_tile_cache,
    calculate_map_zoom,
    draw_crosshair,
    draw_range_rings,
    filter_precipitation_pixels,
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
        # A fully transparent image should remain transparent
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = filter_precipitation_pixels(img)
        assert result.shape == (10, 10, 4)
        assert (result[:, :, 3] == 0).all()

    def test_precipitation_colour_preserved(self):
        # Create an image with a known precipitation colour
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        # Use the first precip colour: (0, 91, 142) with full alpha
        img[5, 5] = [0, 91, 142, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 255  # alpha preserved

    def test_non_precip_colour_removed(self):
        # Create a pixel with a non-precipitation colour (land mask)
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        # Pure white with full alpha - not a precip colour
        img[5, 5] = [255, 255, 255, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 0  # should be made transparent


class TestDrawCrosshair:
    def test_draws_red_at_circle_edge(self):
        from custom_components.incoming_rain.radar.composite import _CROSSHAIR_RADIUS
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
        draw_range_rings(img, centre_x, centre_y, radius_pixels)
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


def _make_context_manager(resp):
    """Wrap a mock response in an async context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_session(map_tile_bytes: bytes | None = None, radar_tile_bytes: bytes | None = None):
    """Return a mock aiohttp.ClientSession that serves tiles from bytes.

    If radar_tile_bytes is None, radar fetches raise an exception to simulate failure.
    """
    map_bytes = map_tile_bytes if map_tile_bytes is not None else _make_tile_png()
    radar_bytes = radar_tile_bytes if radar_tile_bytes is not None else _make_tile_png((0, 0, 0, 0))

    def fake_get(url: str):
        resp = AsyncMock()

        if "tilecache" in url or "rainviewer" in url.lower():
            if radar_tile_bytes is None:
                resp.raise_for_status = MagicMock(side_effect=Exception("radar fetch failed"))
                resp.read = AsyncMock(return_value=b"")
            else:
                resp.read = AsyncMock(return_value=radar_bytes)
                resp.raise_for_status = MagicMock()
        else:
            resp.read = AsyncMock(return_value=map_bytes)
            resp.raise_for_status = MagicMock()

        return _make_context_manager(resp)

    session = MagicMock()
    session.get = fake_get
    return session


class TestRenderComposite:
    @pytest.fixture(autouse=True)
    def clear_tile_cache(self):
        """Clear the map tile cache before each test."""
        _map_tile_cache.clear()
        yield
        _map_tile_cache.clear()

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

        def counting_get(url: str):
            nonlocal call_count
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            if "tilecache" in url or "rainviewer" in url.lower():
                resp.read = AsyncMock(return_value=_make_tile_png((0, 0, 0, 0)))
            else:
                call_count += 1
                resp.read = AsyncMock(return_value=tile_bytes)
            return _make_context_manager(resp)

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

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.format == "GIF"
        assert img.size == (output_size, output_size)

    @pytest.mark.asyncio
    async def test_gif_has_expected_frame_count(self):
        """Pillow deduplicates identical GIF frames, so use distinct precipitation colours."""
        from custom_components.incoming_rain.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        map_bytes = _make_tile_png((30, 30, 30, 255))
        # Build distinct radar tiles keyed by frame path
        precip_tile_by_frame = {}
        for i in range(5):
            r, g, b, _ = PRECIP_COLOURS[i % len(PRECIP_COLOURS)]
            precip_tile_by_frame[f"frame{i}"] = _make_tile_png((int(r), int(g), int(b), 255))

        def varying_get(url: str):
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            if "tilecache" in url or "rainviewer" in url.lower():
                # Match the frame path from the URL to return the right tile
                tile_data = _make_tile_png((0, 0, 0, 0))  # fallback
                for key, data in precip_tile_by_frame.items():
                    if key in url:
                        tile_data = data
                        break
                resp.read = AsyncMock(return_value=tile_data)
            else:
                resp.read = AsyncMock(return_value=map_bytes)
            return _make_context_manager(resp)

        session = MagicMock()
        session.get = varying_get

        frame_paths = [f"/v2/radar/frame{i}" for i in range(5)]

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
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

        result = await render_animated_composite(
            lat=-33.7, lon=151.2, radius_km=128,
            frame_paths=frame_paths,
            output_size=output_size,
            frame_duration_ms=500,
            session=session,
        )

        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img.format == "GIF"
        assert img.size == (output_size, output_size)
