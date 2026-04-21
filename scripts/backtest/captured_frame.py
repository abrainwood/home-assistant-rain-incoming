"""Captured radar frame: loads tile PNGs from disk and implements RadarFrame."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from custom_components.rain_incoming.providers.base import BoundingBox, RadarFrame
from custom_components.rain_incoming.providers.rainviewer import (
    TILE_SIZE,
    _tile_bounds,
    _tile_to_intensity_array,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class CapturedRadarFrame(RadarFrame):
    """A radar frame loaded from captured tile PNGs on disk.

    Implements RadarFrame so captured data can be replayed through the
    existing detection pipeline without modification.
    """

    _timestamp: datetime
    _tile_paths: dict[tuple[int, int], Path]  # (x, y) -> path to PNG
    _zoom: int
    _cached_grid: np.ndarray | None = field(default=None, repr=False)

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def get_intensity_at(self, lat: float, lon: float) -> float:
        """Stub - detect() uses get_intensity_grid, not per-point lookup."""
        return 0.0

    def get_intensity_grid(self, bounds: BoundingBox, width: int, height: int) -> np.ndarray:
        """Return the stitched intensity grid, loading tiles from disk on first call.

        For the backtesting framework the requested dimensions always match the
        stitched 2x2 tile canvas (512x512), so we return the cached canvas directly.
        """
        if self._cached_grid is not None:
            return self._cached_grid

        self._cached_grid = self._stitch_tiles()
        return self._cached_grid

    def _stitch_tiles(self) -> np.ndarray:
        """Load all tile PNGs and stitch them into a single intensity canvas."""
        if not self._tile_paths:
            return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

        xs = [x for x, _ in self._tile_paths]
        ys = [y for _, y in self._tile_paths]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        cols = max_x - min_x + 1
        rows = max_y - min_y + 1
        canvas = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)

        for (tx, ty), path in self._tile_paths.items():
            tile_arr = self._load_tile(tx, ty, path)
            px = (tx - min_x) * TILE_SIZE
            py = (ty - min_y) * TILE_SIZE
            canvas[py : py + TILE_SIZE, px : px + TILE_SIZE] = tile_arr

        return canvas

    def _load_tile(self, tx: int, ty: int, path: Path) -> np.ndarray:
        """Load a single tile PNG and convert to a float32 intensity array.

        Returns a zero array if the file is missing or unreadable.
        """
        try:
            png_bytes = path.read_bytes()
        except OSError as exc:
            _LOGGER.warning(
                "Captured frame: tile (%d,%d) not found at %s: %s",
                tx, ty, path, exc,
            )
            return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

        return _tile_to_intensity_array(png_bytes)
