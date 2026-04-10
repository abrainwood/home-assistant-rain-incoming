from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..const import RAINVIEWER_ZOOM
from ..http_retry import rate_limited_fetch
from ..providers.rainviewer import (
    PRECIP_COLOURS,
    TILE_BASE_URL,
    TILE_SIZE,
)
from .geo import lat_lon_to_tile

_LOGGER = logging.getLogger(__name__)

# RainViewer Universal Blue colour scheme (scheme 2, the only available scheme)
# Reference: https://www.rainviewer.com/api/color-schemes.html
# Build a numpy array of known precipitation RGB values for vectorized filtering.
_PRECIP_RGB = np.array(
    [[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32
)
# Maximum L2 colour distance to accept a pixel as precipitation.
_FILTER_MAX_COLOUR_DISTANCE = 30.0

_MAP_TILE_BASE = os.environ.get(
    "MAP_TILE_URL", "https://basemaps.cartocdn.com/rastertiles/voyager"
)
_RENDER_COLOUR_SCHEME = 2
_RAINVIEWER_TILE_BASE = TILE_BASE_URL

_CROSSHAIR_RADIUS = 8
_CROSSHAIR_LINE_LENGTH = 16
_CROSSHAIR_LINE_GAP = 12

# Module-level cache for static map tiles: (zoom, x, y) -> RGBA Image
_map_tile_cache: dict[tuple[int, int, int], Image.Image] = {}

# Module-level cache for radar tiles: (frame_path, zoom, x, y, scheme) -> RGBA Image
# The cache key includes frame_path, so stale data can never be served.
# We cap size at 500 entries and evict the oldest half when exceeded.
_radar_tile_cache: dict[tuple[str, int, int, int, int], Image.Image] = {}
_RADAR_CACHE_MAX = 500

# Semaphore to limit concurrent tile fetches (shared across map + radar)
_tile_semaphore: asyncio.Semaphore | None = None


def _get_tile_semaphore() -> asyncio.Semaphore:
    """Lazily create the tile semaphore (must be called inside a running event loop)."""
    global _tile_semaphore
    if _tile_semaphore is None:
        _tile_semaphore = asyncio.Semaphore(10)
    return _tile_semaphore


def _evict_radar_cache_if_full() -> None:
    """Evict oldest half of radar tile cache when it exceeds the size limit."""
    if len(_radar_tile_cache) > _RADAR_CACHE_MAX:
        keys = list(_radar_tile_cache.keys())
        for k in keys[: len(keys) // 2]:
            del _radar_tile_cache[k]


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
    """Keep only pixels matching the documented RainViewer precipitation colours.

    RainViewer tiles contain two kinds of non-transparent pixels:
    - Precipitation colours (blues, cyans, yellows, reds, magentas) from the
      documented Universal Blue scheme 2 colour table.
    - Land-mask / ground-clutter colours (khaki/beige, e.g. 170,158,121) that
      are NOT in the documented scheme.

    We match each pixel's RGB against the known precipitation colours using L2
    distance. Pixels that don't match any known colour (distance > threshold)
    are made fully transparent.

    Uses numpy vectorized operations - no Python loops over pixels.
    """
    result = rgba_array.copy()

    # Transparent pixels stay transparent - skip them from colour matching
    alpha_mask = rgba_array[:, :, 3] > 10

    if not alpha_mask.any():
        # No opaque pixels to colour-match, but still zero out low-alpha pixels
        result[:, :, 3] = 0
        return result

    # Extract RGB as float for distance calculation
    rgb = rgba_array[:, :, :3].astype(np.float32)

    # Compute L2 distance from each pixel to each known precipitation colour
    # rgb: (H, W, 3), _PRECIP_RGB: (N, 3) -> diff: (H, W, N, 3)
    diff = rgb[:, :, np.newaxis, :] - _PRECIP_RGB[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W, N)
    best_dist = distances.min(axis=-1)  # (H, W)

    # Pixels that are non-transparent but don't match any precipitation colour
    non_precip = alpha_mask & (best_dist > _FILTER_MAX_COLOUR_DISTANCE)
    result[non_precip, 3] = 0

    # Also zero out truly transparent pixels
    result[~alpha_mask, 3] = 0

    return result


def draw_crosshair(img: Image.Image, cx: int, cy: int) -> None:
    """Draw a red crosshair with white outline at (cx, cy)."""
    draw = ImageDraw.Draw(img)
    r = _CROSSHAIR_RADIUS
    gap = _CROSSHAIR_LINE_GAP
    length = _CROSSHAIR_LINE_LENGTH

    outline_color = (255, 255, 255, 160)
    fill_color = (200, 50, 50, 180)

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


def draw_range_rings(
    img: Image.Image, cx: int, cy: int, full_radius_px: int, radius_km: int,
) -> None:
    """Draw subtle range rings at half-radius and full-radius with distance labels."""
    draw = ImageDraw.Draw(img)
    ring_color = (255, 255, 255, 80)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for factor, km in ((0.5, radius_km // 2), (1.0, radius_km)):
        r = int(full_radius_px * factor)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=ring_color, width=1,
        )
        label = f"{km}km"
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx = cx + 4
        ly = cy - r - th - 2
        # White outline for readability
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((lx + dx, ly + dy), label, fill=(255, 255, 255, 255), font=font)
        draw.text((lx, ly), label, fill=(0, 0, 0, 200), font=font)


async def _fetch_tile(session: aiohttp.ClientSession, url: str) -> Image.Image:
    resp = await rate_limited_fetch(session, url, semaphore=_get_tile_semaphore())
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
    # Boost saturation for richer land/water colours
    from PIL import ImageEnhance
    tile = ImageEnhance.Color(tile).enhance(1.8)
    _map_tile_cache[key] = tile
    return tile


def _draw_timestamp(img: Image.Image, timestamp: datetime, tz_name: str | None = None) -> None:
    """Draw a timestamp with date in the bottom-left corner."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    tz_label = timestamp.strftime("%Z") or tz_name or "UTC"
    label = timestamp.strftime(f"%d %b %Y  %H:%M {tz_label}")
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 8
    lx = padding
    ly = img.height - th - padding

    # White outline for readability on any background
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((lx + dx, ly + dy), label, fill=(255, 255, 255, 220), font=font)
    draw.text((lx, ly), label, fill=(0, 0, 0, 230), font=font)


def _composite_single_frame(
    map_crop: Image.Image,
    radar_resized: Image.Image,
    lat: float,
    lon: float,
    map_zoom: int,
    radius_km: int,
    output_size: int,
    timestamp: datetime | None = None,
    tz_name: str | None = None,
    confidence_map: np.ndarray | None = None,
) -> Image.Image:
    """CPU-bound rendering: composite map + radar, draw overlays. Returns RGBA Image.

    When confidence_map is provided (float32, 0-1, same size as radar_resized),
    it's applied as an alpha multiplier to reduce clutter opacity.
    """
    if confidence_map is not None:
        radar_arr = np.array(radar_resized)
        # Resize confidence map if dimensions differ
        if confidence_map.shape != (radar_arr.shape[0], radar_arr.shape[1]):
            from PIL import Image as PILImage
            conf_img = PILImage.fromarray((confidence_map * 255).astype(np.uint8))
            conf_img = conf_img.resize(
                (radar_arr.shape[1], radar_arr.shape[0]), PILImage.BILINEAR
            )
            confidence_map = np.array(conf_img).astype(np.float32) / 255.0
        # Binary detection mask: mirrors the detector's own threshold logic.
        # Compute effective intensity = luminance * QC_confidence. Pixels at
        # or above the detection threshold render at full opacity; below it
        # they're dimmed to 25% (still faintly visible as context, but clearly
        # distinguished from "real" rain).
        _INTENSITY_THRESHOLD = 0.1
        _DIM_FACTOR = 0.25
        r, g, b = (
            radar_arr[:, :, 0].astype(np.float32),
            radar_arr[:, :, 1].astype(np.float32),
            radar_arr[:, :, 2].astype(np.float32),
        )
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        effective = luminance * confidence_map
        alpha_multiplier = np.where(
            effective >= _INTENSITY_THRESHOLD, 1.0, _DIM_FACTOR
        )
        radar_arr[:, :, 3] = (
            radar_arr[:, :, 3].astype(np.float32) * alpha_multiplier
        ).astype(np.uint8)
        radar_resized = Image.fromarray(radar_arr)

    composite = Image.alpha_composite(map_crop.convert("RGBA"), radar_resized)

    kpp = km_per_pixel(lat, map_zoom)
    radius_px = int(radius_km / kpp) if kpp > 0 else output_size // 2

    cx, cy = output_size // 2, output_size // 2
    draw_range_rings(composite, cx, cy, radius_px, radius_km)
    draw_crosshair(composite, cx, cy)

    if timestamp is not None:
        _draw_timestamp(composite, timestamp, tz_name)

    return composite


def _render_sync(
    map_crop: Image.Image,
    radar_resized: Image.Image,
    lat: float,
    lon: float,
    map_zoom: int,
    radius_km: int,
    output_size: int,
) -> bytes:
    """CPU-bound rendering: composite, draw overlays, export PNG."""
    composite = _composite_single_frame(
        map_crop, radar_resized, lat, lon, map_zoom, radius_km, output_size,
    )
    buf = BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def _render_gif_sync(
    frames: list[Image.Image],
    frame_duration_ms: int,
) -> bytes:
    """CPU-bound: assemble RGBA frames into an animated GIF. Returns GIF bytes."""
    # Quantize all frames to the FIRST frame's palette so colors (crosshair,
    # range rings, labels) stay consistent across the animation.
    first_quantized = frames[0].convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = first_quantized.getpalette()

    quantized_frames = []
    for f in frames:
        rgb = f.convert("RGB")
        q = rgb.quantize(palette=first_quantized, dither=Image.Dither.FLOYDSTEINBERG)
        quantized_frames.append(q)

    buf = BytesIO()
    if len(quantized_frames) == 1:
        quantized_frames[0].save(buf, format="GIF")
    else:
        durations = [frame_duration_ms] * len(quantized_frames)
        durations[-1] = frame_duration_ms * 4  # hold on last frame
        quantized_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=quantized_frames[1:],
            duration=durations,
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
    coords = [
        (tx, ty)
        for ty in range(vp.tile_y_min, vp.tile_y_max + 1)
        for tx in range(vp.tile_x_min, vp.tile_x_max + 1)
    ]

    async def _fetch_one(tx: int, ty: int):
        try:
            return tx, ty, await _fetch_map_tile(session, vp.map_zoom, tx, ty)
        except aiohttp.ClientResponseError as e:
            _LOGGER.warning(
                "Map tile fetch failed: HTTP %d for z=%d x=%d y=%d",
                e.status, vp.map_zoom, tx, ty,
            )
            return tx, ty, None
        except Exception as e:
            _LOGGER.warning(
                "Map tile fetch failed: %s: %s (z=%d x=%d y=%d)",
                type(e).__name__, e, vp.map_zoom, tx, ty,
            )
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
    radar_zoom = RAINVIEWER_ZOOM

    coords = [
        (tx, ty)
        for ty in range(vp.radar_ty_min, vp.radar_ty_max + 1)
        for tx in range(vp.radar_tx_min, vp.radar_tx_max + 1)
    ]

    # Evict old entries if cache is too large
    _evict_radar_cache_if_full()

    async def _fetch_one(tx: int, ty: int):
        cache_key = (frame_path, radar_zoom, tx, ty, _RENDER_COLOUR_SCHEME)
        cached = _radar_tile_cache.get(cache_key)
        if cached is not None:
            return tx, ty, cached

        url = (
            f"{_RAINVIEWER_TILE_BASE}{frame_path}"
            f"/{TILE_SIZE}/{radar_zoom}/{tx}/{ty}/{_RENDER_COLOUR_SCHEME}/0.png"
        )
        try:
            tile = await _fetch_tile(session, url)
            _radar_tile_cache[cache_key] = tile
            return tx, ty, tile
        except aiohttp.ClientResponseError as e:
            _LOGGER.warning(
                "Radar tile fetch failed: HTTP %d for z=%d x=%d y=%d (%s)",
                e.status, radar_zoom, tx, ty, frame_path,
            )
            return tx, ty, None
        except Exception as e:
            _LOGGER.warning(
                "Radar tile fetch failed: %s: %s (z=%d x=%d y=%d)",
                type(e).__name__, e, radar_zoom, tx, ty,
            )
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
            _render_sync, map_crop, radar_resized, lat, lon, vp.map_zoom, radius_km, output_size,
        )

    return _render_sync(map_crop, radar_resized, lat, lon, vp.map_zoom, radius_km, output_size)


async def render_animated_composite(
    lat: float,
    lon: float,
    radius_km: int,
    frame_paths: list[str],
    output_size: int = 640,
    frame_duration_ms: int = 500,
    *,
    frame_timestamps: list[datetime] | None = None,
    tz_name: str | None = None,
    confidence_maps: list[np.ndarray] | None = None,
    session: aiohttp.ClientSession,
    run_in_executor: object = None,
) -> bytes:
    """Render an animated GIF compositing multiple radar frames over a static map.

    Parameters
    ----------
    frame_timestamps: optional list of datetimes matching frame_paths, used for
        overlaying a timestamp on each frame.
    confidence_maps: optional per-frame confidence arrays from the QC pipeline.
        Each array is float32 (H, W), values 0-1. Applied as an alpha multiplier
        to dim low-confidence pixels.
    session: an aiohttp.ClientSession for fetching tiles.
    run_in_executor: optional callable (e.g. hass.async_add_executor_job)
        to offload CPU-bound GIF assembly. If None, rendering runs inline.
    """
    vp = _ViewportParams(lat, lon, radius_km, output_size)

    # Fetch map tiles once - they're the same for every frame
    map_crop = await _fetch_map_crop(session, vp, output_size)

    # Fetch all radar overlays in parallel
    radar_overlays = await asyncio.gather(*[
        _fetch_radar_overlay(session, vp, fp, output_size) for fp in frame_paths
    ])

    timestamps = frame_timestamps or [None] * len(frame_paths)

    # Composite each frame (CPU-bound but fast - no I/O)
    frames: list[Image.Image] = []
    for i, radar_resized in enumerate(radar_overlays):
        ts = timestamps[i] if i < len(timestamps) else None
        conf_map = confidence_maps[i] if confidence_maps and i < len(confidence_maps) else None
        frame_img = _composite_single_frame(
            map_crop, radar_resized, lat, lon, vp.map_zoom, radius_km, output_size,
            timestamp=ts, tz_name=tz_name,
            confidence_map=conf_map,
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
