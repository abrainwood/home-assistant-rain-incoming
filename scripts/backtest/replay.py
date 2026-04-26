"""ReplayEngine: slide a window of frames across CaptureRecords and run detect().

Usage::

    from scripts.backtest.replay import ReplayEngine, ReplayConfig
    engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
    predictions = engine.replay(captures)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from scripts.backtest.captured_frame import CapturedRadarFrame
from scripts.backtest.data_loader import CaptureRecord

from custom_components.rain_incoming.const import (
    INTENSITY_THRESHOLD,
    MAX_ANGULAR_VARIANCE_RADIANS,
    MAX_STORM_SPEED_KMH,
    MIN_CELL_AREA_PIXELS,
    MIN_TEMPORAL_FRAMES,
    PROXIMITY_RADIUS_KM,
    RAINVIEWER_TILE_SIZE,
)
from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import _tile_bounds
from custom_components.rain_incoming.radar.detector import (
    DetectorConfig,
    detect,
)
from custom_components.rain_incoming.radar.qc.clutter_map import (
    ClutterMap,
    get_clutter_frequency,
    update_clutter_map,
)
from custom_components.rain_incoming.radar.qc.scoring import (
    QCConfig,
    compute_confidence_map,
)

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """Result of running detect() on one window of frames."""

    window_end_ts: int            # frame_ts of the last frame in the window
    rain_incoming: bool
    rain_at_location: bool
    arrival_minutes: float | None  # minutes from last frame to arrival, or None
    confidence: str               # "normal", "degraded", "unavailable"
    cell_count: int
    max_intensity: float


@dataclass
class ReplayConfig:
    """Configuration for the replay engine."""

    window_size: int = 8          # number of frames per window
    max_gap_seconds: int = 900    # skip windows with consecutive gap > this
    qc_enabled: bool = True       # compute QC confidence maps
    lookahead_seconds: int = 1800
    intensity_threshold: float = INTENSITY_THRESHOLD
    min_cell_area_pixels: int = MIN_CELL_AREA_PIXELS
    use_acceleration: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_analysis_bounds(capture: CaptureRecord) -> BoundingBox:
    """Derive the analysis bounding box from the tile coords in a capture."""
    tiles = list(capture.tile_paths.keys())
    all_bounds = [_tile_bounds(tx, ty, capture.zoom) for tx, ty in tiles]
    return BoundingBox(
        lat_min=min(b.lat_min for b in all_bounds),
        lat_max=max(b.lat_max for b in all_bounds),
        lon_min=min(b.lon_min for b in all_bounds),
        lon_max=max(b.lon_max for b in all_bounds),
    )


def _build_detector_config(
    capture: CaptureRecord,
    replay_config: ReplayConfig,
) -> DetectorConfig:
    """Build a DetectorConfig from a representative capture in the window."""
    analysis_bounds = _build_analysis_bounds(capture)
    grid_size = RAINVIEWER_TILE_SIZE * 2  # 512 for a 2x2 tile block

    return DetectorConfig(
        lookahead_seconds=replay_config.lookahead_seconds,
        intensity_threshold=replay_config.intensity_threshold,
        min_cell_area_pixels=replay_config.min_cell_area_pixels,
        min_temporal_frames=MIN_TEMPORAL_FRAMES,
        max_angular_variance=MAX_ANGULAR_VARIANCE_RADIANS,
        max_storm_speed_kmh=MAX_STORM_SPEED_KMH,
        proximity_radius_km=PROXIMITY_RADIUS_KM,
        analysis_bounds=analysis_bounds,
        grid_width=grid_size,
        grid_height=grid_size,
        use_acceleration=replay_config.use_acceleration,
    )


def _has_gap(window: list[CaptureRecord], max_gap_seconds: int) -> bool:
    """Return True if any consecutive pair in the window exceeds max_gap_seconds."""
    for i in range(len(window) - 1):
        diff = window[i + 1].frame_ts - window[i].frame_ts
        if diff > max_gap_seconds:
            return True
    return False


def _frame_from_record(record: CaptureRecord) -> CapturedRadarFrame:
    return CapturedRadarFrame(
        _timestamp=datetime.fromtimestamp(record.frame_ts, tz=timezone.utc),
        _tile_paths=record.tile_paths,
        _zoom=record.zoom,
    )


def _compute_confidence_maps(
    frames: list[CapturedRadarFrame],
    detector_config: DetectorConfig,
    clutter_map: ClutterMap | None,
    clutter_maturity: float,
) -> list[np.ndarray]:
    """Compute per-frame QC confidence maps for the given window."""
    qc_config = QCConfig()
    bounds = detector_config.analysis_bounds
    W = detector_config.grid_width
    H = detector_config.grid_height

    grids = [f.get_intensity_grid(bounds, W, H) for f in frames]

    clutter_freq: np.ndarray | None = None
    if clutter_map is not None:
        clutter_freq = get_clutter_frequency(clutter_map)

    confidence_maps: list[np.ndarray] = []
    for i, grid in enumerate(grids):
        cmap = compute_confidence_map(
            grid,
            config=qc_config,
            grids=grids[: i + 1],
            clutter_freq=clutter_freq,
            clutter_maturity=clutter_maturity,
        )
        confidence_maps.append(cmap.confidence)

    return confidence_maps


def _update_clutter(
    clutter_map: ClutterMap | None,
    frames: list[CapturedRadarFrame],
    detector_config: DetectorConfig,
) -> ClutterMap:
    """Update or initialise the clutter map with the latest frame's grid."""
    bounds = detector_config.analysis_bounds
    W = detector_config.grid_width
    H = detector_config.grid_height

    latest_grid = frames[-1].get_intensity_grid(bounds, W, H)

    if clutter_map is None:
        clutter_map = ClutterMap(
            echo_count=np.zeros_like(latest_grid, dtype=np.float32),
            update_count=0.0,
        )
    update_clutter_map(clutter_map, latest_grid)
    return clutter_map


def _to_prediction(
    result,
    frames: list[CapturedRadarFrame],
    window: list[CaptureRecord],
) -> PredictionRecord:
    """Convert a DetectionResult into a PredictionRecord."""
    last_frame_ts = window[-1].frame_ts

    arrival_minutes: float | None = None
    if result.arrival_time is not None and frames:
        last_frame_time = frames[-1].timestamp
        delta_seconds = (result.arrival_time - last_frame_time).total_seconds()
        arrival_minutes = delta_seconds / 60.0

    return PredictionRecord(
        window_end_ts=last_frame_ts,
        rain_incoming=result.rain_incoming,
        rain_at_location=result.rain_at_location,
        arrival_minutes=arrival_minutes,
        confidence=result.confidence.value,
        cell_count=len(result.tracked_cells),
        max_intensity=result.max_approaching_intensity,
    )


# ---------------------------------------------------------------------------
# ReplayEngine
# ---------------------------------------------------------------------------


class ReplayEngine:
    """Slides an 8-frame window across CaptureRecords and calls detect()."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self._config = config or ReplayConfig()

    def replay(self, captures: list[CaptureRecord]) -> list[PredictionRecord]:
        """Slide a window across captures and run detect() on each valid window.

        Records are sorted by frame_ts before processing.  For each window of
        ``window_size`` consecutive captures:

        1. Skip if any consecutive frame_ts difference > max_gap_seconds.
        2. Build CapturedRadarFrame objects.
        3. Build DetectorConfig from the first capture's location and tiles.
        4. Optionally compute QC confidence maps (clutter map warms naturally).
        5. Call detect().
        6. Record and return a PredictionRecord.
        """
        config = self._config
        window_size = config.window_size

        if len(captures) < window_size:
            return []

        sorted_captures = sorted(captures, key=lambda r: r.frame_ts)
        predictions: list[PredictionRecord] = []
        clutter_map: ClutterMap | None = None
        clutter_update_count = 0

        # Build frames once and reuse across overlapping windows.
        # Without this, the same tile PNG is decoded in every window
        # that contains it (~12ms per tile × 4 tiles × 8 frames wasted
        # per overlapping window).
        frame_cache: dict[int, CapturedRadarFrame] = {
            r.frame_ts: _frame_from_record(r) for r in sorted_captures
        }

        for start in range(len(sorted_captures) - window_size + 1):
            window = sorted_captures[start : start + window_size]

            if _has_gap(window, config.max_gap_seconds):
                _LOGGER.debug(
                    "Skipping window ending at ts=%d: gap > %ds",
                    window[-1].frame_ts,
                    config.max_gap_seconds,
                )
                continue

            frames = [frame_cache[r.frame_ts] for r in window]
            detector_config = _build_detector_config(window[0], config)

            confidence_maps: list[np.ndarray] | None = None
            if config.qc_enabled:
                clutter_maturity = min(1.0, clutter_update_count / 2016.0)
                confidence_maps = _compute_confidence_maps(
                    frames, detector_config, clutter_map, clutter_maturity
                )
                clutter_map = _update_clutter(clutter_map, frames, detector_config)
                clutter_update_count += 1

            result = detect(
                frames=frames,
                location=(window[0].lat, window[0].lon),
                config=detector_config,
                confidence_maps=confidence_maps,
            )

            predictions.append(_to_prediction(result, frames, window))

        return predictions

    def replay_manifest(
        self,
        captures: list[CaptureRecord],
        manifest_entries: list,
    ) -> list[PredictionRecord]:
        """Replay only specific windows listed in a manifest.

        For each manifest entry, finds the window of captures ending at
        the entry's window_end_ts and runs detect() on it.
        """
        from scripts.backtest.manifest import ManifestEntry

        config = self._config
        window_size = config.window_size

        sorted_captures = sorted(captures, key=lambda r: r.frame_ts)

        ts_list = [r.frame_ts for r in sorted_captures]
        ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

        # Collect target timestamps from manifest
        target_ts = {e.window_end_ts for e in manifest_entries}

        # Only build frames for captures that are part of a manifest window
        needed_indices: set[int] = set()
        for end_ts in target_ts:
            end_idx = ts_to_idx.get(end_ts)
            if end_idx is not None and end_idx >= window_size - 1:
                for i in range(end_idx - window_size + 1, end_idx + 1):
                    needed_indices.add(i)

        frame_cache: dict[int, CapturedRadarFrame] = {
            sorted_captures[i].frame_ts: _frame_from_record(sorted_captures[i])
            for i in needed_indices
        }

        predictions: list[PredictionRecord] = []

        for end_ts in sorted(target_ts):
            end_idx = ts_to_idx.get(end_ts)
            if end_idx is None or end_idx < window_size - 1:
                continue  # can't build a full window

            start_idx = end_idx - window_size + 1
            window = sorted_captures[start_idx : end_idx + 1]

            if _has_gap(window, config.max_gap_seconds):
                continue

            frames = [frame_cache[r.frame_ts] for r in window]
            detector_config = _build_detector_config(window[0], config)

            confidence_maps: list[np.ndarray] | None = None
            if config.qc_enabled:
                confidence_maps = _compute_confidence_maps(
                    frames, detector_config, None, 0.0
                )

            result = detect(
                frames=frames,
                location=(window[0].lat, window[0].lon),
                config=detector_config,
                confidence_maps=confidence_maps,
            )

            predictions.append(_to_prediction(result, frames, window))

        return predictions
