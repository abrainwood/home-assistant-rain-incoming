from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_MAX_SECONDS,
    BACKOFF_MULTIPLIER,
    CONF_LOOKAHEAD_MINUTES,
    DEFAULT_LOOKAHEAD_MINUTES,
    INTENSITY_THRESHOLD,
    MAX_ANGULAR_VARIANCE_RADIANS,
    MAX_STORM_SPEED_KMH,
    MIN_CELL_AREA_PIXELS,
    MIN_TEMPORAL_FRAMES,
    POLL_INTERVAL_SECONDS,
    PROXIMITY_RADIUS_KM,
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_ZOOM,
    RAINVIEWER_TILE_SIZE,
)
from .providers.base import BoundingBox
from .providers.rainviewer import RainViewerProvider
from .radar.detector import Confidence, DetectionResult, DetectorConfig, detect

_LOGGER = logging.getLogger(__name__)

# Frames to fetch: enough history for motion tracking + temporal persistence
FRAMES_TO_FETCH = 8


def _build_analysis_bounds(lat: float, lon: float) -> BoundingBox:
    """Build a bounding box centred on the location covering the analysis area."""
    import math
    grid_half = RAINVIEWER_ANALYSIS_GRID
    # At zoom 7 each tile is ~2.8125° of longitude
    tile_deg_lon = 360.0 / (2 ** RAINVIEWER_ZOOM)
    tile_deg_lat = tile_deg_lon  # approximate at mid-latitudes
    half_lon = tile_deg_lon * grid_half
    half_lat = tile_deg_lat * grid_half
    return BoundingBox(
        lat_min=lat - half_lat,
        lat_max=lat + half_lat,
        lon_min=lon - half_lon,
        lon_max=lon + half_lon,
    )


class RainDetectorCoordinator(DataUpdateCoordinator[DetectionResult]):
    """Fetches radar frames, runs detection, and manages backoff on failure."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="incoming_rain",
            update_interval=timedelta(seconds=POLL_INTERVAL_SECONDS),
        )
        self._entry = entry
        self._provider = RainViewerProvider()
        self._consecutive_failures = 0
        self.latest_frame_path: str | None = None
        self.last_update_success_time: datetime | None = None

    def _build_config(self) -> DetectorConfig:
        data = self._entry.data
        lat = data.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = data.get(CONF_LONGITUDE, self.hass.config.longitude)
        lookahead = data.get(CONF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES)
        return DetectorConfig(
            lookahead_seconds=lookahead * 60,
            intensity_threshold=INTENSITY_THRESHOLD,
            min_cell_area_pixels=MIN_CELL_AREA_PIXELS,
            min_temporal_frames=MIN_TEMPORAL_FRAMES,
            max_angular_variance=MAX_ANGULAR_VARIANCE_RADIANS,
            max_storm_speed_kmh=MAX_STORM_SPEED_KMH,
            proximity_radius_km=PROXIMITY_RADIUS_KM,
            analysis_bounds=_build_analysis_bounds(lat, lon),
            grid_width=RAINVIEWER_TILE_SIZE * (2 * RAINVIEWER_ANALYSIS_GRID + 1),
            grid_height=RAINVIEWER_TILE_SIZE * (2 * RAINVIEWER_ANALYSIS_GRID + 1),
        )

    async def _async_update_data(self) -> DetectionResult:
        data = self._entry.data
        lat = data.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = data.get(CONF_LONGITUDE, self.hass.config.longitude)

        try:
            if self.data is None:
                # First fetch - fail fast, let HA's ConfigEntryNotReady handle retries
                frames = await self._provider.get_frames(lat, lon, count=FRAMES_TO_FETCH)
            else:
                frames = await self._fetch_with_backoff(lat, lon)
        except Exception as err:
            raise UpdateFailed(f"RainViewer fetch failed: {err}") from err

        config = self._build_config()
        bounds = config.analysis_bounds
        W, H = config.grid_width, config.grid_height

        # Fetch tile grids BEFORE running detection - get_intensity_grid returns
        # zeros until the tiles have been fetched and stitched.
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for frame in frames:
                try:
                    await frame._fetch_stitched_grid(bounds, W, H, session)
                except Exception:
                    _LOGGER.debug("Failed to fetch grid for frame %s", frame.timestamp)

        result = detect(frames=frames, location=(lat, lon), config=config)

        # Store the latest frame path for the radar image entity
        if frames:
            latest = frames[-1]
            if hasattr(latest, "path"):
                self.latest_frame_path = latest.path

        self._consecutive_failures = 0
        self.last_update_success_time = datetime.now(timezone.utc)
        return result

    async def _fetch_with_backoff(self, lat: float, lon: float):
        backoff = BACKOFF_BASE_SECONDS
        while True:
            try:
                frames = await self._provider.get_frames(lat, lon, count=FRAMES_TO_FETCH)
                self._consecutive_failures = 0
                return frames
            except Exception as err:
                self._consecutive_failures += 1
                if backoff >= BACKOFF_MAX_SECONDS:
                    raise
                _LOGGER.warning(
                    "RainViewer fetch failed (attempt %d), retrying in %ds: %s",
                    self._consecutive_failures,
                    backoff,
                    err,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX_SECONDS)
