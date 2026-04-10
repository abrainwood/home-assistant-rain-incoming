from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


class RadarFrame(ABC):
    """A single radar snapshot at a point in time."""

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        ...

    @abstractmethod
    def get_intensity_at(self, lat: float, lon: float) -> float:
        """Precipitation intensity at a coordinate. 0.0 = none, 1.0 = maximum."""

    @abstractmethod
    def get_intensity_grid(self, bounds: BoundingBox, width: int, height: int) -> np.ndarray:
        """
        Return a 2D float32 intensity grid (0.0-1.0) resampled to (height, width).
        Row 0 = lat_max (north), col 0 = lon_min (west).
        """


class RadarProvider(ABC):
    """Source of radar frames. All providers must be swappable behind this interface."""

    @abstractmethod
    async def get_frames(self, lat: float, lon: float, count: int) -> list[RadarFrame]:
        """
        Fetch the most recent `count` frames centred on (lat, lon),
        ordered oldest-first. May return fewer than `count` if unavailable.
        """
