from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
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
    RAINVIEWER_ZOOM,
    RAINVIEWER_TILE_SIZE,
)
from .providers.base import BoundingBox
from .providers.rainviewer import RainViewerProvider
from .radar.detector import Confidence, DetectionResult, DetectorConfig, TrackedCell, detect
from .radar.qc import (
    ClutterMap,
    QCConfig,
    compute_confidence_map,
    get_clutter_frequency,
    load_clutter_map,
    refine_confidence_post_detection,
    save_clutter_map,
    update_clutter_map,
)

_LOGGER = logging.getLogger(__name__)

# Frames to fetch: enough history for motion tracking + temporal persistence
FRAMES_TO_FETCH = 8

# Clutter map persistence
CLUTTER_MAP_FILENAME = "rain_incoming_clutter.npz"
CLUTTER_SAVE_INTERVAL = 36  # save every 36 cycles (~6 hours at 10-min intervals)
CLUTTER_MATURITY_CYCLES = 72  # fully mature after ~12 hours


def _build_analysis_tiles(lat: float, lon: float, zoom: int) -> list[tuple[int, int]]:
    """Return the 4 tile coordinates of the 2x2 block that best centres the location."""
    import math
    from .radar.geo import lat_lon_to_tile

    cx, cy = lat_lon_to_tile(lat, lon, zoom)

    # Fractional tile position of the location
    n = 2 ** zoom
    fx = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    fy = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n

    # Position within the tile (0 to 1)
    frac_x = fx - cx
    frac_y = fy - cy

    # Pick the 2x2 block anchored at the nearest corner
    x0 = cx - 1 if frac_x < 0.5 else cx
    y0 = cy - 1 if frac_y < 0.5 else cy

    return [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]


def _build_analysis_bounds(lat: float, lon: float) -> BoundingBox:
    """Build a bounding box from the smart 2x2 tile selection."""
    from .providers.rainviewer import _tile_bounds

    tiles = _build_analysis_tiles(lat, lon, RAINVIEWER_ZOOM)
    all_bounds = [_tile_bounds(tx, ty, RAINVIEWER_ZOOM) for tx, ty in tiles]
    return BoundingBox(
        lat_min=min(b.lat_min for b in all_bounds),
        lat_max=max(b.lat_max for b in all_bounds),
        lon_min=min(b.lon_min for b in all_bounds),
        lon_max=max(b.lon_max for b in all_bounds),
    )


class RainDetectorCoordinator(DataUpdateCoordinator[DetectionResult]):
    """Fetches radar frames, runs detection, and manages backoff on failure."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="rain_incoming",
            update_interval=timedelta(seconds=POLL_INTERVAL_SECONDS),
        )
        self._entry = entry
        self._provider = RainViewerProvider()
        self._consecutive_failures = 0
        self.latest_frame_path: str | None = None
        self.frame_paths: list[str] = []
        self.frame_timestamps: list[datetime] = []
        self.tracked_cells: list[TrackedCell] = []
        self.confidence_maps: list = []
        self.last_update_success_time: datetime | None = None
        self.last_rain_nearby_time: datetime | None = None

        # Clutter map - loaded lazily in _async_update_data to avoid blocking I/O
        import os
        self._clutter_path = os.path.join(
            hass.config.path(".storage"), CLUTTER_MAP_FILENAME
        )
        self._clutter_map: ClutterMap | None = None
        self._clutter_loaded = False
        self._clutter_cycle_count = 0

        # Frame cache: path -> RainViewerFrame with pre-fetched grid.
        # Avoids re-fetching tile grids for frames that haven't changed between polls.
        self._frame_cache: dict[str, object] = {}

    async def async_save_clutter_map(self) -> None:
        """Persist the clutter map to disk. Called on HA shutdown."""
        if self._clutter_map is not None:
            try:
                await self.hass.async_add_executor_job(
                    save_clutter_map, self._clutter_map, self._clutter_path
                )
            except Exception as e:
                _LOGGER.warning(
                    "Failed to save clutter map on shutdown: %s: %s",
                    type(e).__name__, e,
                )

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
            grid_width=RAINVIEWER_TILE_SIZE * 2,
            grid_height=RAINVIEWER_TILE_SIZE * 2,
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

        # Load clutter map on first update (off the event loop)
        if not self._clutter_loaded:
            self._clutter_map = await self.hass.async_add_executor_job(
                load_clutter_map, self._clutter_path
            )
            self._clutter_loaded = True

        # Fetch tile grids BEFORE running detection - get_intensity_grid returns
        # zeros until the tiles have been fetched and stitched.
        session = async_get_clientsession(self.hass)

        async def _fetch_frame(frame):
            try:
                await frame._fetch_stitched_grid(bounds, W, H, session)
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

        # Incremental fetch: reuse cached frames for paths we've already fetched,
        # only hit the network for genuinely new frames.
        reused = []
        to_fetch = []
        for frame in frames:
            cached = self._frame_cache.get(frame.path)
            if cached is not None:
                reused.append(cached)
            else:
                to_fetch.append(frame)

        _LOGGER.debug(
            "Frames: %d cached, %d new", len(reused), len(to_fetch)
        )

        if to_fetch:
            await asyncio.gather(*[_fetch_frame(f) for f in to_fetch])

        # Build the resolved frame list in manifest order, substituting cached
        # objects for known paths so their pre-populated _cached_grid is used.
        frames = [
            self._frame_cache.get(f.path, f)
            for f in frames
        ]

        # Only cache frames that have successfully fetched grids.
        # If a fetch failed (rate limited, timeout), the frame will have no
        # cached grid - don't cache it so we retry on the next poll.
        self._frame_cache = {
            f.path: f
            for f in frames
            if getattr(f, "_cached_grid", None) is not None
        }

        # Compute per-frame QC confidence maps BEFORE detection so they can
        # be used to gate intensity thresholding in the detector.
        grids = [f.get_intensity_grid(bounds, W, H) for f in frames]

        # If NO frames were successfully fetched, raise so sensors show
        # unavailable instead of a false "no rain". We check _cached_grid
        # (set by _fetch_stitched_grid on success) rather than grid content,
        # because all-zero grids are valid - they mean "no precipitation".
        import numpy as np
        if frames and all(getattr(f, "_cached_grid", None) is None for f in frames):
            raise UpdateFailed(
                "All radar tile fetches failed - no frames have data"
            )
        qc_config = QCConfig()

        # Update clutter map with the latest frame
        latest_grid = grids[-1] if grids else None
        if latest_grid is not None:
            if self._clutter_map is None:
                self._clutter_map = ClutterMap(
                    echo_count=np.zeros_like(latest_grid, dtype=np.float32),
                    update_count=0.0,
                )
            update_clutter_map(self._clutter_map, latest_grid)
            self._clutter_cycle_count += 1

            # Save periodically
            if self._clutter_cycle_count % CLUTTER_SAVE_INTERVAL == 0:
                try:
                    await self.hass.async_add_executor_job(
                        save_clutter_map, self._clutter_map, self._clutter_path
                    )
                except Exception as e:
                    _LOGGER.warning(
                        "Failed to save clutter map: %s: %s",
                        type(e).__name__, e,
                    )

        # Get clutter frequency and maturity for QC scoring
        clutter_freq = None
        clutter_maturity = 0.0
        if self._clutter_map is not None:
            clutter_freq = get_clutter_frequency(self._clutter_map)
            clutter_maturity = min(1.0, self._clutter_map.update_count / CLUTTER_MATURITY_CYCLES)

        self.confidence_maps = []
        for i, grid in enumerate(grids):
            grids_up_to_i = grids[: i + 1]
            cmap = compute_confidence_map(
                grid,
                config=qc_config,
                grids=grids_up_to_i,
                clutter_freq=clutter_freq,
                clutter_maturity=clutter_maturity,
            )
            self.confidence_maps.append(cmap.confidence)

        result = detect(
            frames=frames,
            location=(lat, lon),
            config=config,
            confidence_maps=self.confidence_maps,
        )

        # Post-detection pass: refine confidence with speed/motion factors
        # (only for the last frame, which is what gets rendered)
        if (
            result.tracked_cells
            and result.last_labeled is not None
            and self.confidence_maps
        ):
            refined = refine_confidence_post_detection(
                pre_confidence=self.confidence_maps[-1],
                cells=result.tracked_cells,
                labeled=result.last_labeled,
                config=qc_config,
            )
            self.confidence_maps[-1] = refined

        # Store frame paths, timestamps, and tracked cells for the radar image entity
        self.frame_paths = [f.path for f in frames if hasattr(f, "path")]
        self.frame_timestamps = [f.timestamp for f in frames]
        self.tracked_cells = result.tracked_cells
        if self.frame_paths:
            self.latest_frame_path = self.frame_paths[-1]

        # Update last_rain_nearby_time when rain is currently at the location
        if (
            result.rain_incoming
            and result.arrival_time is not None
            and frames
            and result.arrival_time == frames[-1].timestamp
        ):
            self.last_rain_nearby_time = datetime.now(timezone.utc)

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
