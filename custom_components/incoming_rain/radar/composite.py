from __future__ import annotations

import logging
import math
import os
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image, ImageDraw

from ..const import RAINVIEWER_COLOUR_SCHEME, RAINVIEWER_ZOOM
from ..providers.rainviewer import (
    TILE_BASE_URL,
    TILE_SIZE,
    PRECIP_COLOURS,
    MAX_COLOUR_DISTANCE,
)
from .geo import lat_lon_to_tile

_LOGGER = logging.getLogger(__name__)

_MAP_TILE_BASE = os.environ.get("MAP_TILE_URL", "https://basemaps.cartocdn.com/dark_all")
_RAINVIEWER_TILE_BASE = TILE_BASE_URL

_CROSSHAIR_RADIUS = 8
_CROSSHAIR_LINE_LENGTH = 16
_CROSSHAIR_LINE_GAP = 12

# Module-level cache for static map tiles: (zoom, x, y) -> RGBA Image
_map_tile_cache: dict[tuple[int, int, int], Image.Image] = {}


def km_per_pixel(lat: float, zoom: int) -> float:
    return (40075.0 / (256 * 2**zoom)) * math.cos(math.radians(lat))


def calculate_map_zoom(lat: float, radius_km: int, output_size: int) -> int:
    best_zoom = 1
    best_diff = float("inf")
    for z in range(1, 19):
        kpp = km_per_pixel(lat, z)
        if kpp <= 0:
            continue
        radius_pixels = radius_km / kpp
        diameter_pixels = 2 * radius_pixels
        diff = abs(diameter_pixels - output_size)
        if diff < best_diff:
            best_diff = diff
            best_zoom = z
    return best_zoom


def _lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Return global pixel coordinates for lat/lon at given zoom."""
    n = 2**zoom
    px = (lon + 180.0) / 360.0 * n * 256
    lat_r = math.radians(lat)
    py = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * 256
    return px, py


def filter_precipitation_pixels(rgba_array: np.ndarray) -> np.ndarray:
    """Keep only pixels matching known precipitation colours, zero out the rest."""
    result = rgba_array.copy()
    alpha_mask = rgba_array[:, :, 3] >= 10

    if not alpha_mask.any():
        return result

    colours = np.array(
        [[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32
    )
    rgb = rgba_array[:, :, :3].astype(np.float32)
    diff = rgb[:, :, np.newaxis, :] - colours[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    best_dist = distances.min(axis=-1)

    non_precip = ~alpha_mask | (best_dist > MAX_COLOUR_DISTANCE)
    result[non_precip, 3] = 0

    return result


def draw_crosshair(img: Image.Image, cx: int, cy: int) -> None:
    """Draw a red crosshair with white outline at (cx, cy)."""
    draw = ImageDraw.Draw(img)
    r = _CROSSHAIR_RADIUS
    gap = _CROSSHAIR_LINE_GAP
    length = _CROSSHAIR_LINE_LENGTH

    outline_color = (255, 255, 255, 200)
    fill_color = (255, 0, 0, 255)

    # White outline circle
    draw.ellipse(
        [cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
        outline=outline_color, width=1,
    )
    # Red circle
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=fill_color, width=2,
    )

    # Crosshair lines with white outline then red fill
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        x0 = cx + dx * gap
        y0 = cy + dy * gap
        x1 = cx + dx * (gap + length)
        y1 = cy + dy * (gap + length)
        draw.line([(x0, y0), (x1, y1)], fill=outline_color, width=3)
        draw.line([(x0, y0), (x1, y1)], fill=fill_color, width=1)


def draw_range_rings(img: Image.Image, cx: int, cy: int, full_radius_px: int) -> None:
    """Draw subtle range rings at half-radius and full-radius."""
    draw = ImageDraw.Draw(img)
    ring_color = (255, 255, 255, 60)

    for factor in (0.5, 1.0):
        r = int(full_radius_px * factor)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=ring_color, width=1,
        )


async def _fetch_tile(session: aiohttp.ClientSession, url: str) -> Image.Image:
    async with session.get(url) as resp:
        resp.raise_for_status()
        data = await resp.read()
    return Image.open(BytesIO(data)).convert("RGBA")


async def _fetch_map_tile(
    session: aiohttp.ClientSession, zoom: int, tx: int, ty: int,
) -> Image.Image:
    """Fetch a map tile, using the module-level cache for hits."""
    key = (zoom, tx, ty)
    cached = _map_tile_cache.get(key)
    if cached is not None:
        return cached

    url = f"{_MAP_TILE_BASE}/{zoom}/{tx}/{ty}.png"
    tile = await _fetch_tile(session, url)
    _map_tile_cache[key] = tile
    return tile


def _composite_single_frame(
    map_crop: Image.Image,
    radar_resized: Image.Image,
    lat: float,
    map_zoom: int,
    radius_km: int,
    output_size: int,
) -> Image.Image:
    """CPU-bound rendering: composite map + radar, draw overlays. Returns RGBA Image."""
    composite = Image.alpha_composite(map_crop.convert("RGBA"), radar_resized)

    kpp = km_per_pixel(lat, map_zoom)
    radius_px = int(radius_km / kpp) if kpp > 0 else output_size // 2

    cx, cy = output_size // 2, output_size // 2
    draw_range_rings(composite, cx, cy, radius_px)
    draw_crosshair(composite, cx, cy)

    return composite


def _render_sync(
    map_crop: Image.Image,
    radar_resized: Image.Image,
    lat: float,
    map_zoom: int,
    radius_km: int,
    output_size: int,
) -> bytes:
    """CPU-bound rendering: composite, draw overlays, export PNG."""
    composite = _composite_single_frame(
        map_crop, radar_resized, lat, map_zoom, radius_km, output_size,
    )
    buf = BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def _render_gif_sync(
    frames: list[Image.Image],
    frame_duration_ms: int,
) -> bytes:
    """CPU-bound: assemble RGBA frames into an animated GIF. Returns GIF bytes."""
    rgb_frames = [f.convert("RGB") for f in frames]
    buf = BytesIO()
    if len(rgb_frames) == 1:
        rgb_frames[0].save(buf, format="GIF")
    else:
        rgb_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=rgb_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
        )
    return buf.getvalue()


class _ViewportParams:
    """Pre-computed viewport geometry shared across frames."""

    __slots__ = (
        "map_zoom", "centre_px", "centre_py", "half_size",
        "tile_x_min", "tile_x_max", "tile_y_min", "tile_y_max",
        "radar_vp_left", "radar_vp_right", "radar_vp_top", "radar_vp_bottom",
        "radar_tx_min", "radar_tx_max", "radar_ty_min", "radar_ty_max",
    )

    def __init__(self, lat: float, lon: float, radius_km: int, output_size: int) -> None:
        self.map_zoom = calculate_map_zoom(lat, radius_km, output_size)
        self.centre_px, self.centre_py = _lat_lon_to_pixel(lat, lon, self.map_zoom)
        self.half_size = output_size / 2

        self.tile_x_min = int((self.centre_px - self.half_size) // 256)
        self.tile_x_max = int((self.centre_px + self.half_size) // 256)
        self.tile_y_min = int((self.centre_py - self.half_size) // 256)
        self.tile_y_max = int((self.centre_py + self.half_size) // 256)

        radar_zoom = RAINVIEWER_ZOOM
        map_n = 2 ** self.map_zoom
        radar_n = 2 ** radar_zoom
        scale = radar_n / map_n

        vp_left = self.centre_px - self.half_size
        vp_right = self.centre_px + self.half_size
        vp_top = self.centre_py - self.half_size
        vp_bottom = self.centre_py + self.half_size

        self.radar_vp_left = vp_left * scale
        self.radar_vp_right = vp_right * scale
        self.radar_vp_top = vp_top * scale
        self.radar_vp_bottom = vp_bottom * scale

        self.radar_tx_min = int(self.radar_vp_left // 256)
        self.radar_tx_max = int(self.radar_vp_right // 256)
        self.radar_ty_min = int(self.radar_vp_top // 256)
        self.radar_ty_max = int(self.radar_vp_bottom // 256)


async def _fetch_map_crop(
    session: aiohttp.ClientSession,
    vp: _ViewportParams,
    output_size: int,
) -> Image.Image:
    """Fetch and stitch map tiles (in parallel), then crop to viewport."""
    import asyncio

    coords = [
        (tx, ty)
        for ty in range(vp.tile_y_min, vp.tile_y_max + 1)
        for tx in range(vp.tile_x_min, vp.tile_x_max + 1)
    ]

    async def _fetch_one(tx: int, ty: int):
        try:
            return tx, ty, await _fetch_map_tile(session, vp.map_zoom, tx, ty)
        except Exception:
            _LOGGER.debug("Failed to fetch map tile z=%d x=%d y=%d", vp.map_zoom, tx, ty)
            return tx, ty, None

    results = await asyncio.gather(*[_fetch_one(tx, ty) for tx, ty in coords])

    canvas_w = (vp.tile_x_max - vp.tile_x_min + 1) * 256
    canvas_h = (vp.tile_y_max - vp.tile_y_min + 1) * 256
    map_canvas = Image.new("RGBA", (canvas_w, canvas_h))

    for tx, ty, tile in results:
        if tile is not None:
            map_canvas.paste(tile, ((tx - vp.tile_x_min) * 256, (ty - vp.tile_y_min) * 256))

    crop_x = int(vp.centre_px - vp.half_size - vp.tile_x_min * 256)
    crop_y = int(vp.centre_py - vp.half_size - vp.tile_y_min * 256)
    return map_canvas.crop((crop_x, crop_y, crop_x + output_size, crop_y + output_size))


async def _fetch_radar_overlay(
    session: aiohttp.ClientSession,
    vp: _ViewportParams,
    frame_path: str,
    output_size: int,
) -> Image.Image:
    """Fetch radar tiles for one frame (in parallel), filter, crop, and resize to viewport."""
    import asyncio
    radar_zoom = RAINVIEWER_ZOOM

    coords = [
        (tx, ty)
        for ty in range(vp.radar_ty_min, vp.radar_ty_max + 1)
        for tx in range(vp.radar_tx_min, vp.radar_tx_max + 1)
    ]

    async def _fetch_one(tx: int, ty: int):
        url = (
            f"{_RAINVIEWER_TILE_BASE}{frame_path}"
            f"/{TILE_SIZE}/{radar_zoom}/{tx}/{ty}/{RAINVIEWER_COLOUR_SCHEME}/0.png"
        )
        try:
            return tx, ty, await _fetch_tile(session, url)
        except Exception:
            _LOGGER.debug("Failed to fetch radar tile z=%d x=%d y=%d", radar_zoom, tx, ty)
            return tx, ty, None

    results = await asyncio.gather(*[_fetch_one(tx, ty) for tx, ty in coords])

    radar_canvas_w = (vp.radar_tx_max - vp.radar_tx_min + 1) * 256
    radar_canvas_h = (vp.radar_ty_max - vp.radar_ty_min + 1) * 256
    radar_canvas = Image.new("RGBA", (radar_canvas_w, radar_canvas_h), (0, 0, 0, 0))

    for tx, ty, tile in results:
        if tile is not None:
            radar_canvas.paste(tile, ((tx - vp.radar_tx_min) * 256, (ty - vp.radar_ty_min) * 256))

    radar_arr = np.array(radar_canvas)
    radar_arr = filter_precipitation_pixels(radar_arr)
    radar_filtered = Image.fromarray(radar_arr)

    radar_crop_x = int(vp.radar_vp_left - vp.radar_tx_min * 256)
    radar_crop_y = int(vp.radar_vp_top - vp.radar_ty_min * 256)
    radar_crop_w = int(vp.radar_vp_right - vp.radar_vp_left)
    radar_crop_h = int(vp.radar_vp_bottom - vp.radar_vp_top)

    radar_crop = radar_filtered.crop((
        radar_crop_x, radar_crop_y,
        radar_crop_x + radar_crop_w, radar_crop_y + radar_crop_h,
    ))

    return radar_crop.resize((output_size, output_size), Image.BILINEAR)


async def render_composite(
    lat: float,
    lon: float,
    radius_km: int,
    frame_path: str,
    output_size: int = 640,
    *,
    session: aiohttp.ClientSession,
    run_in_executor: object = None,
) -> bytes:
    """Render a radar composite image as PNG bytes.

    Parameters
    ----------
    session: an aiohttp.ClientSession for fetching tiles.
    run_in_executor: optional callable (e.g. hass.async_add_executor_job)
        to offload CPU-bound rendering. If None, rendering runs inline.
    """
    vp = _ViewportParams(lat, lon, radius_km, output_size)
    map_crop = await _fetch_map_crop(session, vp, output_size)
    radar_resized = await _fetch_radar_overlay(session, vp, frame_path, output_size)

    if run_in_executor is not None:
        return await run_in_executor(
            _render_sync, map_crop, radar_resized, lat, vp.map_zoom, radius_km, output_size,
        )

    return _render_sync(map_crop, radar_resized, lat, vp.map_zoom, radius_km, output_size)


async def render_animated_composite(
    lat: float,
    lon: float,
    radius_km: int,
    frame_paths: list[str],
    output_size: int = 640,
    frame_duration_ms: int = 500,
    *,
    session: aiohttp.ClientSession,
    run_in_executor: object = None,
) -> bytes:
    """Render an animated GIF compositing multiple radar frames over a static map.

    Parameters
    ----------
    session: an aiohttp.ClientSession for fetching tiles.
    run_in_executor: optional callable (e.g. hass.async_add_executor_job)
        to offload CPU-bound GIF assembly. If None, rendering runs inline.
    """
    vp = _ViewportParams(lat, lon, radius_km, output_size)

    # Fetch map tiles once - they're the same for every frame
    map_crop = await _fetch_map_crop(session, vp, output_size)

    # Fetch all radar overlays in parallel
    import asyncio
    radar_overlays = await asyncio.gather(*[
        _fetch_radar_overlay(session, vp, fp, output_size) for fp in frame_paths
    ])

    # Composite each frame (CPU-bound but fast - no I/O)
    frames: list[Image.Image] = []
    for radar_resized in radar_overlays:
        frame_img = _composite_single_frame(
            map_crop, radar_resized, lat, vp.map_zoom, radius_km, output_size,
        )
        frames.append(frame_img)

    if not frames:
        # Shouldn't happen, but return a blank GIF if no frames
        blank = Image.new("RGB", (output_size, output_size), (0, 0, 0))
        buf = BytesIO()
        blank.save(buf, format="GIF")
        return buf.getvalue()

    if run_in_executor is not None:
        return await run_in_executor(_render_gif_sync, frames, frame_duration_ms)

    return _render_gif_sync(frames, frame_duration_ms)
