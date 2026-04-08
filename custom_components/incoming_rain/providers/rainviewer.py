from __future__ import annotations

import math
from datetime import datetime, timezone
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image

from .base import BoundingBox, RadarFrame, RadarProvider
from ..radar.geo import lat_lon_to_tile

import os


def _resolve_url(env_key: str, default: str) -> str:
    """Return the env var value if set and non-empty, otherwise the default."""
    return os.environ.get(env_key) or default


_API_BASE = _resolve_url("RAINVIEWER_API_URL", "https://api.rainviewer.com")
_TILE_BASE = _resolve_url("RAINVIEWER_TILE_URL", "https://tilecache.rainviewer.com")
MANIFEST_URL = f"{_API_BASE}/public/weather-maps.json"
TILE_BASE_URL = _TILE_BASE
TILE_SIZE = 256

# RainViewer precipitation colour table for colour scheme 6 (ordered light→heavy).
# Each entry: (R, G, B, intensity 0.0-1.0).
# Intensities are calibrated against approximate dBZ equivalents.
PRECIP_COLOURS: list[tuple[int, int, int, float]] = [
    (0, 91, 142, 0.10),    # very light (~16 dBZ)
    (0, 119, 170, 0.18),   # light (~20 dBZ)
    (0, 154, 213, 0.28),   # light-moderate (~24 dBZ)
    (81, 197, 232, 0.38),  # moderate (~28 dBZ)
    (255, 224, 0, 0.50),   # moderate (~32 dBZ)
    (255, 170, 0, 0.63),   # moderate-heavy (~36 dBZ)
    (255, 68, 0, 0.77),    # heavy (~40 dBZ)
    (193, 0, 0, 0.88),     # very heavy (~44 dBZ)
    (255, 119, 255, 1.00), # extreme (>48 dBZ)
]

# Maximum L2 colour distance to match a pixel to a known precipitation colour.
# Pixels further than this threshold are treated as land mask or unknown = 0.
MAX_COLOUR_DISTANCE = 60.0


def _colour_distance(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def _colour_to_intensity(r: int, g: int, b: int, alpha: int) -> float:
    """Map an RGBA pixel to a precipitation intensity (0.0-1.0)."""
    if alpha < 10:
        return 0.0
    best_dist = float("inf")
    best_intensity = 0.0
    for pr, pg, pb, intensity in PRECIP_COLOURS:
        d = _colour_distance(r, g, b, pr, pg, pb)
        if d < best_dist:
            best_dist = d
            best_intensity = intensity
    if best_dist > MAX_COLOUR_DISTANCE:
        return 0.0  # land mask or unrecognised colour
    return best_intensity



def _tile_bounds(tx: int, ty: int, zoom: int) -> BoundingBox:
    """Return the geographic bounding box of a tile."""
    n = 2 ** zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return BoundingBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def _tile_to_intensity_array(image_bytes: bytes) -> np.ndarray:
    """Convert raw tile PNG bytes to a float32 intensity array (TILE_SIZE x TILE_SIZE)."""
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)  # shape (H, W, 4)

    result = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    alpha_mask = arr[:, :, 3] >= 10
    if not alpha_mask.any():
        return result

    # Build colour/intensity lookup from the precipitation table
    colours = np.array(
        [[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32
    )
    intensities = np.array(
        [i for _, _, _, i in PRECIP_COLOURS], dtype=np.float32
    )

    # Compute L2 distance from each pixel to each precipitation colour
    rgb = arr[:, :, :3]  # (H, W, 3)
    diff = rgb[:, :, np.newaxis, :] - colours[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W, N_colours)

    best_idx = distances.argmin(axis=-1)
    best_dist = distances.min(axis=-1)

    valid = alpha_mask & (best_dist <= MAX_COLOUR_DISTANCE)
    result[valid] = intensities[best_idx[valid]]
    return result


class RainViewerFrame(RadarFrame):
    """A single radar frame from RainViewer."""

    def __init__(self, timestamp: datetime, path: str, zoom: int, colour_scheme: int) -> None:
        self._timestamp = timestamp
        self._path = path
        self._zoom = zoom
        self._colour_scheme = colour_scheme
        self._cached_grid: np.ndarray | None = None
        self._cached_bounds: BoundingBox | None = None

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def path(self) -> str:
        return self._path

    def get_intensity_at(self, lat: float, lon: float) -> float:
        if self._cached_grid is None or self._cached_bounds is None:
            return 0.0
        b = self._cached_bounds
        col = int((lon - b.lon_min) / (b.lon_max - b.lon_min) * self._cached_grid.shape[1])
        row = int((b.lat_max - lat) / (b.lat_max - b.lat_min) * self._cached_grid.shape[0])
        row = max(0, min(row, self._cached_grid.shape[0] - 1))
        col = max(0, min(col, self._cached_grid.shape[1] - 1))
        return float(self._cached_grid[row, col])

    def get_intensity_grid(self, bounds: BoundingBox, width: int, height: int) -> np.ndarray:
        if (
            self._cached_grid is not None
            and self._cached_bounds == bounds
            and self._cached_grid.shape == (height, width)
        ):
            return self._cached_grid
        # Return empty grid until async fetch has been called
        return np.zeros((height, width), dtype=np.float32)

    async def _fetch_stitched_grid(
        self,
        bounds: BoundingBox,
        width: int,
        height: int,
        session: aiohttp.ClientSession,
    ) -> np.ndarray:
        """Fetch tiles covering bounds, stitch, and resample to (height, width)."""
        cx, cy = lat_lon_to_tile(
            (bounds.lat_min + bounds.lat_max) / 2,
            (bounds.lon_min + bounds.lon_max) / 2,
            self._zoom,
        )

        # Determine how many tiles we need in each direction
        # Collect the tiles that overlap the bounding box
        tiles_needed: list[tuple[int, int]] = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                tx, ty = cx + dx, cy + dy
                tb = _tile_bounds(tx, ty, self._zoom)
                # Check overlap
                if (
                    tb.lon_max > bounds.lon_min
                    and tb.lon_min < bounds.lon_max
                    and tb.lat_max > bounds.lat_min
                    and tb.lat_min < bounds.lat_max
                ):
                    tiles_needed.append((tx, ty))

        if not tiles_needed:
            return np.zeros((height, width), dtype=np.float32)

        # Determine overall tile grid extent
        min_tx = min(t[0] for t in tiles_needed)
        max_tx = max(t[0] for t in tiles_needed)
        min_ty = min(t[1] for t in tiles_needed)
        max_ty = max(t[1] for t in tiles_needed)

        cols = max_tx - min_tx + 1
        rows = max_ty - min_ty + 1
        canvas = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)

        for tx, ty in tiles_needed:
            url = (
                f"{TILE_BASE_URL}{self._path}"
                f"/{TILE_SIZE}/{self._zoom}/{tx}/{ty}/{self._colour_scheme}/0.png"
            )
            async with session.get(url) as resp:
                resp.raise_for_status()
                tile_bytes = await resp.read()

            tile_arr = _tile_to_intensity_array(tile_bytes)
            px = (tx - min_tx) * TILE_SIZE
            py = (ty - min_ty) * TILE_SIZE
            canvas[py : py + TILE_SIZE, px : px + TILE_SIZE] = tile_arr

        # Canvas geographic bounds
        canvas_bounds = BoundingBox(
            lat_min=_tile_bounds(min_tx, max_ty, self._zoom).lat_min,
            lat_max=_tile_bounds(min_tx, min_ty, self._zoom).lat_max,
            lon_min=_tile_bounds(min_tx, min_ty, self._zoom).lon_min,
            lon_max=_tile_bounds(max_tx, min_ty, self._zoom).lon_max,
        )

        # Crop to requested bounds via pixel coordinates
        def lat_to_row(lat: float, cb: BoundingBox, h: int) -> int:
            return int((cb.lat_max - lat) / (cb.lat_max - cb.lat_min) * h)

        def lon_to_col(lon: float, cb: BoundingBox, w: int) -> int:
            return int((lon - cb.lon_min) / (cb.lon_max - cb.lon_min) * w)

        ch, cw = canvas.shape
        r0 = max(0, lat_to_row(bounds.lat_max, canvas_bounds, ch))
        r1 = min(ch, lat_to_row(bounds.lat_min, canvas_bounds, ch))
        c0 = max(0, lon_to_col(bounds.lon_min, canvas_bounds, cw))
        c1 = min(cw, lon_to_col(bounds.lon_max, canvas_bounds, cw))

        if r1 <= r0 or c1 <= c0:
            return np.zeros((height, width), dtype=np.float32)

        cropped = canvas[r0:r1, c0:c1]

        # Resample to requested (height, width) using PIL
        pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        resampled = np.array(pil_img, dtype=np.float32) / 255.0

        self._cached_grid = resampled
        self._cached_bounds = bounds
        return resampled


class RainViewerProvider(RadarProvider):
    """Fetches radar frames from the RainViewer public API."""

    ZOOM = 7
    COLOUR_SCHEME = 6

    async def get_frames(self, lat: float, lon: float, count: int) -> list[RadarFrame]:
        """Fetch the most recent `count` frames, oldest-first."""
        manifest = await self._fetch_manifest()
        past = manifest.get("radar", {}).get("past", [])
        selected = past[-count:]  # most recent N, still oldest-first
        return [
            RainViewerFrame(
                timestamp=datetime.fromtimestamp(entry["time"], tz=timezone.utc),
                path=entry["path"],
                zoom=self.ZOOM,
                colour_scheme=self.COLOUR_SCHEME,
            )
            for entry in selected
        ]

    async def _fetch_manifest(self) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(MANIFEST_URL) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)
