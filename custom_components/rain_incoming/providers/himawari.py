"""NASA GIBS Himawari Band 13 Clean Infrared tile provider.

Used as a cloud-presence confidence signal: if there are no clouds near a
location, any radar return is noise. Tiles are 256x256 RGBA PNG at zoom 6
(the max available for this layer).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image

from ..const import GIBS_HIMAWARI_URL, GIBS_HIMAWARI_ZOOM
from ..http_retry import fetch_with_retry

_LOGGER = logging.getLogger(__name__)


@dataclass
class IRTile:
    pixels: np.ndarray  # (256, 256, 4) RGBA uint8
    tile_x: int
    tile_y: int
    zoom: int


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def cloud_fraction_in_window(
    tile: IRTile,
    lat: float,
    lon: float,
    radius_km: float,
    threshold: int,
) -> float:
    """Fraction of pixels within radius_km of (lat, lon) whose luminance >= threshold.

    Luminance is the mean of the RGB channels. Window bounds are clamped to
    the 256x256 tile, so radii larger than the tile span are safe.
    """
    n = 2 ** tile.zoom
    fx = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    fy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    px = int((fx - tile.tile_x) * 256)
    py = int((fy - tile.tile_y) * 256)

    km_per_px = (40075.0 * math.cos(lat_rad)) / (n * 256)
    half = max(1, int(radius_km / km_per_px))

    x0 = max(0, px - half)
    x1 = min(256, px + half + 1)
    y0 = max(0, py - half)
    y1 = min(256, py + half + 1)

    window = tile.pixels[y0:y1, x0:x1]
    luminance = window[:, :, :3].astype(np.float32).mean(axis=2)
    return float((luminance >= threshold).mean())


async def fetch_himawari_ir_tile(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> IRTile | None:
    zoom = GIBS_HIMAWARI_ZOOM
    x, y = _lat_lon_to_tile(lat, lon, zoom)
    url = GIBS_HIMAWARI_URL.format(z=zoom, y=y, x=x)
    try:
        resp = await fetch_with_retry(session, url)
        data = await resp.read()
        img = Image.open(BytesIO(data)).convert("RGBA")
        pixels = np.array(img, dtype=np.uint8)
        return IRTile(pixels=pixels, tile_x=x, tile_y=y, zoom=zoom)
    except aiohttp.ClientResponseError as e:
        _LOGGER.warning(
            "Himawari IR tile fetch failed: HTTP %d for %s", e.status, url
        )
        return None
    except Exception as e:
        _LOGGER.warning(
            "Himawari IR tile decode failed: %s: %s", type(e).__name__, e
        )
        return None
