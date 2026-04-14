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
    async def get_frames(
        self, lat: float, lon: float, count: int, session=None,
    ) -> list[RadarFrame]:
        """
        Fetch the most recent `count` frames centred on (lat, lon),
        ordered oldest-first. May return fewer than `count` if unavailable.

        session: optional HTTP session. When provided, the provider should
        use it instead of creating its own. In HA context this is the shared
        session from async_get_clientsession.
        """

    @abstractmethod
    async def prefetch_frames(
        self,
        frames: list[RadarFrame],
        bounds: BoundingBox,
        width: int,
        height: int,
        session,
        **kwargs,
    ) -> None:
        """Pre-fetch grid data for frames so get_intensity_grid returns real data.

        Each provider handles its own transport (HTTP, file, etc.) and
        concurrency strategy. The coordinator calls this once per update
        cycle instead of reaching into individual frame internals.

        Frames that fail to fetch should log a warning and be skipped -
        the coordinator handles partial failures via the _cached_grid check.
        """
