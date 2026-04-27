from __future__ import annotations

import logging
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.const import (
    SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD,
)
from custom_components.rain_incoming.providers.himawari import (
    IRTile,
    cloud_fraction_in_window,
    fetch_himawari_ir_tile,
)


def _make_ir_tile_png() -> bytes:
    """Create a minimal 256x256 RGBA PNG simulating a Himawari IR tile."""
    img = Image.new("RGBA", (256, 256), (80, 80, 80, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestFetchHimawariIrTile:
    @pytest.mark.asyncio
    async def test_returns_ir_tile_for_known_location(self):
        """Happy path: fetch returns IRTile with correct shape, dtype, zoom,
        and the GIBS Himawari layer URL is requested."""
        tile_bytes = _make_ir_tile_png()

        captured_urls: list[str] = []

        async def _fake_fetch(session, url, **kwargs):
            captured_urls.append(url)
            resp = AsyncMock()
            resp.read = AsyncMock(return_value=tile_bytes)
            return resp

        mock_session = MagicMock()

        with patch(
            "custom_components.rain_incoming.providers.himawari.fetch_with_retry",
            side_effect=_fake_fetch,
        ):
            # Sydney
            result = await fetch_himawari_ir_tile(-33.87, 151.21, mock_session)

        assert result is not None
        assert isinstance(result, IRTile)
        assert isinstance(result.pixels, np.ndarray)
        assert result.pixels.shape == (256, 256, 4)
        assert result.pixels.dtype == np.uint8
        assert isinstance(result.tile_x, int)
        assert isinstance(result.tile_y, int)
        assert result.zoom == 6

        assert len(captured_urls) == 1
        url = captured_urls[0]
        assert "Himawari_AHI_Band13_Clean_Infrared" in url
        assert "GoogleMapsCompatible_Level6" in url
        assert "/6/" in url  # zoom 6

    @pytest.mark.asyncio
    async def test_returns_none_and_warns_on_http_error(self, caplog):
        """Sad path: HTTP error from fetch_with_retry returns None and logs WARNING."""
        http_error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404,
            message="Not Found",
        )

        mock_session = MagicMock()
        with caplog.at_level(logging.WARNING, logger="custom_components.rain_incoming.providers.himawari"):
            with patch(
                "custom_components.rain_incoming.providers.himawari.fetch_with_retry",
                side_effect=http_error,
            ):
                result = await fetch_himawari_ir_tile(-33.87, 151.21, mock_session)

        assert result is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one WARNING log on HTTP error"
        combined = " ".join(r.getMessage() for r in warnings)
        assert "404" in combined, f"WARNING should include HTTP status; got: {combined!r}"
        assert "Himawari" in combined, f"WARNING should include layer/URL hint; got: {combined!r}"

    @pytest.mark.asyncio
    async def test_returns_none_and_warns_on_malformed_response(self, caplog):
        """Sad path: non-PNG response body causes Image.open to raise; we
        return None and log WARNING rather than letting it propagate."""
        bad_resp = AsyncMock()
        bad_resp.read = AsyncMock(return_value=b"not a png")

        async def _fake_fetch(session, url, **kwargs):
            return bad_resp

        mock_session = MagicMock()
        with caplog.at_level(logging.WARNING, logger="custom_components.rain_incoming.providers.himawari"):
            with patch(
                "custom_components.rain_incoming.providers.himawari.fetch_with_retry",
                side_effect=_fake_fetch,
            ):
                result = await fetch_himawari_ir_tile(-33.87, 151.21, mock_session)

        assert result is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one WARNING log on malformed response"


# Sydney coords -> tile (58, 38) at zoom 6 per _lat_lon_to_tile. Reused
# across cloud-fraction tests so the (lat, lon) the function uses to locate
# the window aligns with the synthetic tile we hand it.
_SYDNEY_LAT = -33.87
_SYDNEY_LON = 151.21
_SYDNEY_TILE_X = 58
_SYDNEY_TILE_Y = 38


def _make_uniform_tile(rgb: tuple[int, int, int]) -> IRTile:
    """Build a synthetic IRTile filled with a uniform RGB colour."""
    pixels = np.zeros((256, 256, 4), dtype=np.uint8)
    pixels[:, :, 0] = rgb[0]
    pixels[:, :, 1] = rgb[1]
    pixels[:, :, 2] = rgb[2]
    pixels[:, :, 3] = 255
    return IRTile(
        pixels=pixels,
        tile_x=_SYDNEY_TILE_X,
        tile_y=_SYDNEY_TILE_Y,
        zoom=6,
    )


class TestCloudFractionInWindow:
    """`cloud_fraction_in_window` is a pure function over IRTile.pixels.

    Luminance = mean of RGB channels. Pixels whose luminance is >= threshold
    count as cloudy. Result is the cloudy fraction within a circular/square
    window of `radius_km` around (lat, lon)."""

    def test_all_bright_pixels_returns_one(self):
        bright = SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD + 30
        tile = _make_uniform_tile((bright, bright, bright))

        result = cloud_fraction_in_window(
            tile,
            _SYDNEY_LAT,
            _SYDNEY_LON,
            radius_km=50,
            threshold=SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD,
        )

        assert result == pytest.approx(1.0)

    def test_all_dark_pixels_returns_zero(self):
        dark = SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD - 30
        tile = _make_uniform_tile((dark, dark, dark))

        result = cloud_fraction_in_window(
            tile,
            _SYDNEY_LAT,
            _SYDNEY_LON,
            radius_km=50,
            threshold=SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD,
        )

        assert result == pytest.approx(0.0)

    def test_pixels_at_threshold_count_as_cloudy(self):
        """Boundary case: luminance == threshold must be treated as cloudy
        (>= comparison, not strict >). A tile uniformly filled at exactly
        threshold should yield fraction = 1.0."""
        thr = SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD
        tile = _make_uniform_tile((thr, thr, thr))

        result = cloud_fraction_in_window(
            tile,
            _SYDNEY_LAT,
            _SYDNEY_LON,
            radius_km=50,
            threshold=thr,
        )

        assert result == pytest.approx(1.0)

    def test_half_bright_half_dark_returns_about_half(self):
        """Mixed tile: top half rows bright (cloudy), bottom half dark.

        Use a large radius so the window covers the full tile, making the
        cloudy fraction exactly 0.5 regardless of where Sydney lands within
        the tile."""
        bright = SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD + 30
        dark = SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD - 30

        pixels = np.zeros((256, 256, 4), dtype=np.uint8)
        pixels[:128, :, 0:3] = bright
        pixels[128:, :, 0:3] = dark
        pixels[:, :, 3] = 255

        tile = IRTile(
            pixels=pixels,
            tile_x=_SYDNEY_TILE_X,
            tile_y=_SYDNEY_TILE_Y,
            zoom=6,
        )

        # Radius huge enough that the window covers the full 256x256 tile.
        # At zoom 6 / lat ~-34, km_per_pixel ~2.16km, so 256 px is ~553 km.
        result = cloud_fraction_in_window(
            tile,
            _SYDNEY_LAT,
            _SYDNEY_LON,
            radius_km=10_000,
            threshold=SATELLITE_CLOUD_BRIGHTNESS_THRESHOLD,
        )

        assert result == pytest.approx(0.5, abs=0.01)
