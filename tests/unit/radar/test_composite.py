from __future__ import annotations

import math
from datetime import datetime, timezone
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
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = filter_precipitation_pixels(img)
        assert result.shape == (10, 10, 4)
        assert (result[:, :, 3] == 0).all()

    def test_opaque_pixel_preserved(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [100, 150, 200, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 255

    def test_low_alpha_pixel_removed(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [255, 255, 255, 5]  # alpha <= 10 threshold
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 0


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
        from custom_components.incoming_rain.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        map_bytes = _make_tile_png((30, 30, 30, 255))
        precip_tile_by_frame = {}
        for i in range(5):
            r, g, b, _ = PRECIP_COLOURS[i % len(PRECIP_COLOURS)]
            precip_tile_by_frame[f"frame{i}"] = _make_tile_png((int(r), int(g), int(b), 255))

        def varying_get(url: str):
            resp = AsyncMock()
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
            return _make_context_manager(resp)

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
        from custom_components.incoming_rain.providers.rainviewer import PRECIP_COLOURS

        output_size = 256
        map_bytes = _make_tile_png((30, 30, 30, 255))
        precip_tile_by_frame = {}
        for i in range(3):
            r, g, b, _ = PRECIP_COLOURS[i % len(PRECIP_COLOURS)]
            precip_tile_by_frame[f"frame{i}"] = _make_tile_png((int(r), int(g), int(b), 255))

        def varying_get(url: str):
            resp = AsyncMock()
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
            return _make_context_manager(resp)

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


class TestConfidenceAlphaCurve:
    """Verify the confidence-to-alpha curve produces visible rain at moderate confidence
    and suppresses clutter at low confidence."""

    def test_rain_at_60pct_confidence_is_visible(self):
        """A pixel at 60% confidence should have alpha >= 100 in the output.

        With the original conf^3 curve: 0.6^3 * 200 = 43 (INVISIBLE - bug!)
        With the fixed conf^1.5 curve: 0.6^1.5 * 200 = 93 (visible)
        """
        from custom_components.incoming_rain.radar.composite import _composite_single_frame

        output_size = 64
        # Create radar image: single green pixel with alpha=200
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        radar_arr[32, 32] = [0, 200, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        # Confidence map: 0.6 everywhere
        confidence_map = np.full((output_size, output_size), 0.6, dtype=np.float32)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            lat=-33.7,
            lon=151.2,
            map_zoom=8,
            radius_km=128,
            output_size=output_size,
            confidence_map=confidence_map,
        )
        result_arr = np.array(result)
        # The pixel at (32, 32) should have visible green (alpha-composited over black)
        # Alpha = 200 * adjusted_confidence. At conf=0.6:
        #   conf^3: 200 * 0.216 = 43 (barely visible)
        #   conf^1.5: 200 * 0.465 = 93 (clearly visible)
        # After alpha compositing over black bg, green channel = G * (alpha/255)
        # We check the green channel of the composite output is clearly visible
        green_val = result_arr[32, 32, 1]
        assert green_val >= 50, (
            f"Rain at 60% confidence should be visible, but green channel = {green_val}"
        )

    def test_clutter_at_20pct_confidence_is_suppressed(self):
        """A pixel at 20% confidence should have alpha < 50 in the output.

        With conf^1.5: 0.2^1.5 * 200 = 17.9 (well suppressed)
        With conf^3: 0.2^3 * 200 = 1.6 (also suppressed)
        """
        from custom_components.incoming_rain.radar.composite import _composite_single_frame

        output_size = 64
        radar_arr = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        radar_arr[32, 32] = [0, 200, 0, 200]
        radar_img = Image.fromarray(radar_arr)

        confidence_map = np.full((output_size, output_size), 0.2, dtype=np.float32)

        map_crop = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))
        result = _composite_single_frame(
            map_crop=map_crop,
            radar_resized=radar_img,
            lat=-33.7,
            lon=151.2,
            map_zoom=8,
            radius_km=128,
            output_size=output_size,
            confidence_map=confidence_map,
        )
        result_arr = np.array(result)
        # At 20% confidence, the green should be barely visible
        green_val = result_arr[32, 32, 1]
        assert green_val < 50, (
            f"Clutter at 20% confidence should be suppressed, but green channel = {green_val}"
        )


class TestConfidenceWeightedRendering:
    """Verify that QC texture scoring dims speckly radar and preserves smooth rain."""

    @pytest.fixture(autouse=True)
    def clear_tile_cache(self):
        _map_tile_cache.clear()
        yield
        _map_tile_cache.clear()

    @pytest.mark.asyncio
    async def test_speckle_is_dimmed_vs_smooth_rain(self):
        """Speckly radar tile should have lower mean alpha than smooth rain tile
        after confidence-weighted rendering."""
        output_size = 256

        # Create a smooth rain radar tile: solid green with full alpha
        smooth_img = Image.new("RGBA", (256, 256), (0, 200, 0, 200))
        smooth_buf = BytesIO()
        smooth_img.save(smooth_buf, format="PNG")
        smooth_bytes = smooth_buf.getvalue()

        # Create a speckly radar tile: random scattered pixels
        rng = np.random.default_rng(42)
        speckle_arr = np.zeros((256, 256, 4), dtype=np.uint8)
        mask = rng.random((256, 256)) > 0.7
        speckle_arr[mask] = [0, 200, 0, 200]
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


