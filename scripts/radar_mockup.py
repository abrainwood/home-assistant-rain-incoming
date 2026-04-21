#!/usr/bin/env python3
"""
Generate radar composite mockups at different radii.

Fetches real RainViewer + CartoDB tiles, crops to 64/128/256km radius,
and saves separate mockup files.
"""
from __future__ import annotations

import asyncio
import math
from io import BytesIO

import aiohttp
import aiohttp.resolver
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from custom_components.rain_incoming.radar.geo import lat_lon_to_tile
from custom_components.rain_incoming.providers.rainviewer import (
    MAX_COLOUR_DISTANCE,
    PRECIP_COLOURS,
    TILE_SIZE as _TILE_SIZE,
    _tile_bounds,
)
from custom_components.rain_incoming.radar.composite import km_per_pixel
from custom_components.rain_incoming.const import (
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_COLOUR_SCHEME,
    RAINVIEWER_TILE_SIZE,
    RAINVIEWER_ZOOM,
)

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_BASE_URL = "https://tilecache.rainviewer.com"
MAP_TILE_URL = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
ZOOM = RAINVIEWER_ZOOM
COLOUR_SCHEME = RAINVIEWER_COLOUR_SCHEME
TILE_SIZE = RAINVIEWER_TILE_SIZE
LAT, LON = -33.701, 151.209
GRID_HALF = RAINVIEWER_ANALYSIS_GRID
OUTPUT_SIZE = 640

# RGB-only view of the production precipitation colour table (strip the intensity field)
_PRECIP_COLOURS = [(r, g, b) for r, g, b, _ in PRECIP_COLOURS]
_MAX_COLOUR_DIST = MAX_COLOUR_DISTANCE


def filter_precipitation_only(tile):
    arr = np.array(tile, dtype=np.float32)
    result = np.zeros_like(arr, dtype=np.uint8)
    alpha_mask = arr[:, :, 3] >= 10
    if not alpha_mask.any():
        return Image.fromarray(result, "RGBA")

    colours = np.array(_PRECIP_COLOURS, dtype=np.float32)
    rgb = arr[:, :, :3]
    diff = rgb[:, :, np.newaxis, :] - colours[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    best_dist = distances.min(axis=-1)
    is_precip = alpha_mask & (best_dist <= _MAX_COLOUR_DIST)

    orig = np.array(tile, dtype=np.uint8)
    result[is_precip] = orig[is_precip]
    result[is_precip, 3] = np.clip(orig[is_precip, 3].astype(int) + 40, 0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGBA")


async def fetch_tile_grid(session, url_template, center_tx, center_ty, **fmt_kwargs):
    size = (2 * GRID_HALF + 1) * TILE_SIZE
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for dy in range(-GRID_HALF, GRID_HALF + 1):
        for dx in range(-GRID_HALF, GRID_HALF + 1):
            tx, ty = center_tx + dx, center_ty + dy
            url = url_template.format(z=ZOOM, x=tx, y=ty, **fmt_kwargs)
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        tile = Image.open(BytesIO(await resp.read())).convert("RGBA")
                        canvas.paste(tile, ((dx + GRID_HALF) * TILE_SIZE, (dy + GRID_HALF) * TILE_SIZE))
            except Exception as e:
                print(f"  Failed ({tx},{ty}): {e}")
    return canvas


async def fetch_radar_grid(session, frame_path, center_tx, center_ty):
    size = (2 * GRID_HALF + 1) * TILE_SIZE
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for dy in range(-GRID_HALF, GRID_HALF + 1):
        for dx in range(-GRID_HALF, GRID_HALF + 1):
            tx, ty = center_tx + dx, center_ty + dy
            url = f"{TILE_BASE_URL}{frame_path}/{TILE_SIZE}/{ZOOM}/{tx}/{ty}/{COLOUR_SCHEME}/0.png"
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        tile = Image.open(BytesIO(await resp.read())).convert("RGBA")
                        filtered = filter_precipitation_only(tile)
                        px = (dx + GRID_HALF) * TILE_SIZE
                        py = (dy + GRID_HALF) * TILE_SIZE
                        canvas.paste(filtered, (px, py), filtered)
            except Exception as e:
                print(f"  Failed radar ({tx},{ty}): {e}")
    return canvas


def location_to_pixel(lat, lon, center_tx, center_ty, zoom, grid_half, tile_size):
    """Convert lat/lon to pixel coordinates in the tile grid canvas."""
    canvas_size = (2 * grid_half + 1) * tile_size
    min_tx = center_tx - grid_half
    min_ty = center_ty - grid_half
    max_tx = center_tx + grid_half
    max_ty = center_ty + grid_half
    top_left = _tile_bounds(min_tx, min_ty, zoom)
    bottom_right = _tile_bounds(max_tx, max_ty, zoom)
    clon_min = top_left.lon_min
    clat_max = top_left.lat_max
    clon_max = bottom_right.lon_max
    clat_min = bottom_right.lat_min

    loc_x = int((lon - clon_min) / (clon_max - clon_min) * canvas_size)
    loc_y = int((clat_max - lat) / (clat_max - clat_min) * canvas_size)
    return loc_x, loc_y


def draw_location_marker(img, x, y):
    draw = ImageDraw.Draw(img)
    r_outer, r_inner, cross_len = 12, 4, 20
    colour = (255, 60, 60, 255)
    outline = (255, 255, 255, 200)
    gap = r_outer + 4

    draw.ellipse([x-r_outer-1, y-r_outer-1, x+r_outer+1, y+r_outer+1], outline=outline, width=2)
    draw.ellipse([x-r_outer, y-r_outer, x+r_outer, y+r_outer], outline=colour, width=2)
    draw.ellipse([x-r_inner, y-r_inner, x+r_inner, y+r_inner], fill=colour)
    for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.line([(x-cross_len+ox, y+oy), (x-gap+ox, y+oy)], fill=outline, width=1)
        draw.line([(x+gap+ox, y+oy), (x+cross_len+ox, y+oy)], fill=outline, width=1)
        draw.line([(x+ox, y-cross_len+oy), (x+ox, y-gap+oy)], fill=outline, width=1)
        draw.line([(x+ox, y+gap+oy), (x+ox, y+cross_len+oy)], fill=outline, width=1)
    draw.line([(x-cross_len, y), (x-gap, y)], fill=colour, width=2)
    draw.line([(x+gap, y), (x+cross_len, y)], fill=colour, width=2)
    draw.line([(x, y-cross_len), (x, y-gap)], fill=colour, width=2)
    draw.line([(x, y+gap), (x, y+cross_len)], fill=colour, width=2)


def draw_range_ring(img, cx, cy, radius_km, kpp, label):
    """Draw a subtle range ring at the given radius."""
    draw = ImageDraw.Draw(img)
    r_px = int(radius_km / kpp)
    draw.ellipse(
        [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
        outline=(255, 255, 255, 60), width=1,
    )
    # Label at top of ring
    draw.text((cx + 4, cy - r_px - 2), f"{label}", fill=(255, 255, 255, 100))


def crop_to_radius(composite, loc_x, loc_y, radius_km, kpp):
    """Crop image to a square centered on location covering the given radius."""
    r_px = int(radius_km / kpp)
    left = loc_x - r_px
    top = loc_y - r_px
    right = loc_x + r_px
    bottom = loc_y + r_px

    # Clamp to image bounds
    w, h = composite.size
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    cropped = composite.crop((left, top, right, bottom))
    return cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


async def main():
    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        print("Fetching manifest...")
        async with session.get(MANIFEST_URL) as resp:
            manifest = await resp.json(content_type=None)

        latest = manifest["radar"]["past"][-1]
        center_tx, center_ty = lat_lon_to_tile(LAT, LON, ZOOM)
        print(f"Radar frame: {latest['path']}, center tile: ({center_tx}, {center_ty})")

        print("Fetching map + radar tiles...")
        base_map, radar = await asyncio.gather(
            fetch_tile_grid(session, MAP_TILE_URL, center_tx, center_ty),
            fetch_radar_grid(session, latest["path"], center_tx, center_ty),
        )

    composite = Image.alpha_composite(base_map, radar)

    loc_x, loc_y = location_to_pixel(LAT, LON, center_tx, center_ty, ZOOM, GRID_HALF, TILE_SIZE)
    kpp = km_per_pixel(LAT, ZOOM)
    print(f"Resolution: {kpp:.2f} km/pixel at lat {LAT}")

    for radius_km in [64, 128, 256]:
        img = composite.copy()

        # Draw range rings on the full image before cropping
        draw_range_ring(img, loc_x, loc_y, radius_km // 2, kpp, f"{radius_km // 2}km")
        draw_range_ring(img, loc_x, loc_y, radius_km, kpp, f"{radius_km}km")
        draw_location_marker(img, loc_x, loc_y)

        cropped = crop_to_radius(img, loc_x, loc_y, radius_km, kpp)

        filename = f"mockup_radar_{radius_km}km.png"
        cropped.save(filename)
        print(f"Saved {filename} ({OUTPUT_SIZE}x{OUTPUT_SIZE}, {radius_km}km radius)")

    # Clean up old file
    import os
    try:
        os.remove("mockup_radar.png")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
