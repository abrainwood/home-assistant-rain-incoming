from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from io import BytesIO

import aiohttp
import numpy as np
from PIL import Image

from .base import BoundingBox, RadarFrame, RadarProvider
from ..http_retry import RateLimitBudget, fetch_with_retry
from ..radar.geo import lat_lon_to_tile

import os

_LOGGER = logging.getLogger(__name__)


def _resolve_url(env_key: str, default: str) -> str:
    """Return the env var value if set and non-empty, otherwise the default."""
    return os.environ.get(env_key) or default


_API_BASE = _resolve_url("RAINVIEWER_API_URL", "https://api.rainviewer.com")
_TILE_BASE = _resolve_url("RAINVIEWER_TILE_URL", "https://tilecache.rainviewer.com")
MANIFEST_URL = f"{_API_BASE}/public/weather-maps.json"
TILE_BASE_URL = _TILE_BASE
TILE_SIZE = 256

# RainViewer Universal Blue colour scheme (scheme 2, the only available scheme)
# Reference: https://www.rainviewer.com/api/color-schemes.html
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


# Pre-compute colour/intensity arrays once at module level
_COLOUR_ARRAY = np.array(
    [[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32
)
_INTENSITY_ARRAY = np.array(
    [i for _, _, _, i in PRECIP_COLOURS], dtype=np.float32
)


def _tile_to_intensity_array(image_bytes: bytes) -> np.ndarray:
    """Convert raw tile PNG bytes to a float32 intensity array (TILE_SIZE x TILE_SIZE).

    Uses unique-colour lookup: real radar tiles have <10 unique colours,
    so we match only the unique values instead of broadcasting across all
    65,536 pixels. ~13x faster than per-pixel L2 distance.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)  # shape (H, W, 4)

    result = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    alpha_mask = arr[:, :, 3] >= 10
    if not alpha_mask.any():
        return result

    # Pack RGB into a single uint32 for fast unique/equality operations
    rgb_packed = (
        arr[:, :, 0].astype(np.uint32) << 16
        | arr[:, :, 1].astype(np.uint32) << 8
        | arr[:, :, 2].astype(np.uint32)
    )
    unique_packed = np.unique(rgb_packed[alpha_mask])

    # Match only the unique colours (typically <10) against the palette
    max_dist_sq = MAX_COLOUR_DISTANCE ** 2
    for packed in unique_packed:
        r = float((packed >> 16) & 0xFF)
        g = float((packed >> 8) & 0xFF)
        b = float(packed & 0xFF)
        rgb_vec = np.array([r, g, b], dtype=np.float32)
        dists_sq = ((_COLOUR_ARRAY - rgb_vec) ** 2).sum(axis=1)
        best = int(dists_sq.argmin())
        if dists_sq[best] <= max_dist_sq:
            mask = alpha_mask & (rgb_packed == packed)
            result[mask] = _INTENSITY_ARRAY[best]

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
        budget: RateLimitBudget | None = None,
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
            resp = await fetch_with_retry(session, url, budget=budget)
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
    COLOUR_SCHEME = 2

    async def get_frames(
        self, lat: float, lon: float, count: int, session=None,
    ) -> list[RadarFrame]:
        """Fetch the most recent `count` frames, oldest-first."""
        manifest = await self._fetch_manifest(session)
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

    async def prefetch_frames(
        self,
        frames: list[RadarFrame],
        bounds: BoundingBox,
        width: int,
        height: int,
        session,
        **kwargs,
    ) -> None:
        """Pre-fetch stitched grids for all frames concurrently.

        Failures are logged as warnings and the frame is skipped - the
        coordinator checks _cached_grid to detect partial failures.
        """
        budget = kwargs.get("budget")

        async def _fetch_one(frame: RainViewerFrame) -> None:
            try:
                await frame._fetch_stitched_grid(bounds, width, height, session, budget=budget)
            except aiohttp.ClientResponseError as e:
                _LOGGER.warning(
                    "Radar grid fetch failed: HTTP %d for frame %s",
                    e.status, frame.timestamp,
                )
            except Exception as e:
                _LOGGER.warning(
                    "Radar grid fetch failed: %s: %s for frame %s",
                    type(e).__name__, e, frame.timestamp,
                )

        rv_frames = [f for f in frames if isinstance(f, RainViewerFrame)]
        if rv_frames:
            await asyncio.gather(*[_fetch_one(f) for f in rv_frames])

    async def _fetch_manifest(self, session=None) -> dict:
        if session is not None:
            resp = await fetch_with_retry(session, MANIFEST_URL)
            return await resp.json(content_type=None)
        async with aiohttp.ClientSession() as fallback_session:
            resp = await fetch_with_retry(fallback_session, MANIFEST_URL)
            return await resp.json(content_type=None)


async def check_coverage(
    lat: float, lon: float, session: aiohttp.ClientSession | None = None,
) -> bool:
    """Probe RainViewer for radar coverage at the given coordinates.

    Fetches a spread of recent radar frames and checks whether any tile
    at this location contains non-transparent pixels. If all frames are
    transparent, there is likely no radar coverage for this area.

    Raises on network/API errors so the caller can decide how to handle.
    """
    tx, ty = lat_lon_to_tile(lat, lon, RainViewerProvider.ZOOM)

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()

    try:
        resp = await fetch_with_retry(session, MANIFEST_URL)
        manifest = await resp.json(content_type=None)

        past = manifest.get("radar", {}).get("past", [])
        if not past:
            return False

        indices = {0, len(past) // 2, len(past) - 1}

        async def _probe(idx: int) -> bool:
            frame = past[idx]
            url = (
                f"{TILE_BASE_URL}{frame['path']}"
                f"/{TILE_SIZE}/{RainViewerProvider.ZOOM}/{tx}/{ty}"
                f"/{RainViewerProvider.COLOUR_SCHEME}/0.png"
            )
            tile_resp = await fetch_with_retry(session, url)
            tile_bytes = await tile_resp.read()
            img = Image.open(BytesIO(tile_bytes)).convert("RGBA")
            alpha = np.array(img)[:, :, 3]
            return bool(alpha.max() > 10)

        results = await asyncio.gather(*[_probe(i) for i in indices])
        return any(results)
    finally:
        if owns_session:
            await session.close()
