from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from scipy import ndimage

from ..providers.base import BoundingBox, RadarFrame
from .filters import filter_by_area, threshold_intensity
from .motion import (
    estimate_velocity,
    extract_cell_centroids,
    is_directionally_coherent,
    match_cells_across_frames,
)


class Confidence(Enum):
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    NORMAL = "normal"


@dataclass
class TrackedCell:
    """A rain cell identified as real by the detection pipeline."""

    lat: float
    lon: float
    velocity_kmh: float  # speed in km/h
    bearing: float  # direction in degrees (0=N, 90=E, 180=S, 270=W)
    confidence: float = 1.0  # QC confidence for this cell (0.0-1.0)


@dataclass
class DetectorConfig:
    lookahead_seconds: int
    intensity_threshold: float
    min_cell_area_pixels: int
    min_temporal_frames: int
    max_angular_variance: float
    max_storm_speed_kmh: float
    proximity_radius_km: float
    analysis_bounds: BoundingBox
    grid_width: int
    grid_height: int


@dataclass
class DetectionResult:
    rain_incoming: bool
    arrival_time: datetime | None
    confidence: Confidence
    frame_count: int
    max_approaching_intensity: float = 0.0
    tracked_cells: list[TrackedCell] = None  # type: ignore[assignment]
    last_labeled: np.ndarray | None = None  # labeled array of last frame's cells

    def __post_init__(self) -> None:
        if self.tracked_cells is None:
            self.tracked_cells = []


def _location_to_pixel(
    lat: float, lon: float, bounds: BoundingBox, width: int, height: int
) -> tuple[int, int]:
    """Convert a lat/lon to (row, col) in a grid with the given bounds."""
    col = int((lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * width)
    row = int((bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * height)
    return row, col


def _pixel_to_location(
    row: float, col: float, bounds: BoundingBox, width: int, height: int
) -> tuple[float, float]:
    """Convert (row, col) in the analysis grid back to (lat, lon)."""
    lat = bounds.lat_max - (row / height) * (bounds.lat_max - bounds.lat_min)
    lon = bounds.lon_min + (col / width) * (bounds.lon_max - bounds.lon_min)
    return lat, lon


def _pixel_size_km(bounds: BoundingBox, width: int, height: int) -> tuple[float, float]:
    """Return approximate (km_per_row, km_per_col) for the grid."""
    lat_span_km = (bounds.lat_max - bounds.lat_min) * 111.0
    lon_span_km = (
        (bounds.lon_max - bounds.lon_min)
        * 111.0
        * math.cos(math.radians((bounds.lat_min + bounds.lat_max) / 2))
    )
    return lat_span_km / height, lon_span_km / width


# A track entry: (frame_index, cell_label_in_that_frame, centroid)
_TrackEntry = tuple[int, int, tuple[float, float]]


def _track_velocity_kmh_bearing(
    track: list[_TrackEntry],
    frames: list[RadarFrame],
    km_per_row: float,
    km_per_col: float,
) -> tuple[float, float]:
    """Compute mean speed (km/h) and bearing (degrees) for a track.

    Returns (0.0, 0.0) if velocity cannot be computed.
    """
    velocities: list[tuple[float, float]] = []
    for i in range(len(track) - 1):
        fi, _, ci = track[i]
        fj, _, cj = track[i + 1]
        t_delta = (frames[fj].timestamp - frames[fi].timestamp).total_seconds()
        if t_delta > 0:
            velocities.append(estimate_velocity(ci, cj, t_delta))

    if not velocities:
        return 0.0, 0.0

    vy = sum(v[0] for v in velocities) / len(velocities)
    vx = sum(v[1] for v in velocities) / len(velocities)

    vy_km_s = vy * km_per_row
    vx_km_s = vx * km_per_col
    speed_kmh = math.sqrt(vy_km_s**2 + vx_km_s**2) * 3600
    bearing = math.degrees(math.atan2(vx_km_s, -vy_km_s)) % 360
    return speed_kmh, bearing


def _build_cell_tracks(
    per_frame_centroids: list[dict[int, tuple[float, float]]],
    max_match_distance: float,
) -> list[list[_TrackEntry]]:
    """
    Match cells across consecutive frames and build tracks.

    A track is a list of (frame_idx, label, centroid) where each consecutive
    pair represents the same physical cell matched across frames.
    """
    frame_count = len(per_frame_centroids)
    if frame_count == 0:
        return []

    # Initialise: one track per cell in frame 0
    active: list[list[_TrackEntry]] = [
        [(0, lbl, centroid)]
        for lbl, centroid in per_frame_centroids[0].items()
    ]
    completed: list[list[_TrackEntry]] = []

    for frame_idx in range(1, frame_count):
        next_centroids = per_frame_centroids[frame_idx]
        prev_centroids = per_frame_centroids[frame_idx - 1]

        matches = match_cells_across_frames(
            prev_centroids,
            next_centroids,
            max_distance=max_match_distance,
        )
        match_map: dict[int, int] = {t0: t1 for t0, t1 in matches}
        matched_next: set[int] = set(match_map.values())

        new_active: list[list[_TrackEntry]] = []
        for track in active:
            last_fi, last_lbl, _ = track[-1]
            if last_fi == frame_idx - 1 and last_lbl in match_map:
                next_lbl = match_map[last_lbl]
                new_active.append(
                    track + [(frame_idx, next_lbl, next_centroids[next_lbl])]
                )
            else:
                completed.append(track)

        # New single-frame tracks for unmatched cells in this frame
        for lbl, centroid in next_centroids.items():
            if lbl not in matched_next:
                new_active.append([(frame_idx, lbl, centroid)])

        active = new_active

    completed.extend(active)
    return completed


def detect(
    frames: list[RadarFrame],
    location: tuple[float, float],
    config: DetectorConfig,
    confidence_maps: list[np.ndarray] | None = None,
) -> DetectionResult:
    """Run the full rain detection pipeline over a list of radar frames.

    Parameters
    ----------
    confidence_maps: optional per-frame QC confidence arrays. When provided,
        intensity grids are multiplied by confidence before thresholding,
        so low-confidence pixels (clutter) are effectively filtered out.
    """
    frame_count = len(frames)

    if frame_count < 2:
        return DetectionResult(
            rain_incoming=False,
            arrival_time=None,
            confidence=Confidence.UNAVAILABLE,
            frame_count=frame_count,
            max_approaching_intensity=0.0,
        )

    confidence = Confidence.DEGRADED if frame_count < 3 else Confidence.NORMAL

    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    # 1. Extract intensity grids from all frames
    grids = [f.get_intensity_grid(bounds, W, H) for f in frames]

    # Apply QC confidence as a multiplier before thresholding
    if confidence_maps is not None:
        effective_grids = []
        for i, g in enumerate(grids):
            if i < len(confidence_maps) and confidence_maps[i] is not None:
                effective_grids.append(g * confidence_maps[i])
            else:
                effective_grids.append(g)
    else:
        effective_grids = grids

    # 2. Threshold + spatial filter each frame independently
    masks = [threshold_intensity(g, config.intensity_threshold) for g in effective_grids]
    masks = [filter_by_area(m, config.min_cell_area_pixels) for m in masks]

    # 3. Label each frame independently
    per_frame_centroids: list[dict[int, tuple[float, float]]] = []
    per_frame_labeled: list[np.ndarray] = []
    for mask in masks:
        labeled, _ = ndimage.label(mask)
        per_frame_labeled.append(labeled)
        per_frame_centroids.append(extract_cell_centroids(labeled))

    # 4. Build cell tracks across frames
    max_match_dist = max(W, H) * 0.25  # allow up to 25% of grid size per frame
    all_tracks = _build_cell_tracks(per_frame_centroids, max_match_dist)

    # 5. Keep only tracks that span >= min_temporal_frames AND end in the last frame
    last_frame_idx = frame_count - 1
    valid_tracks = [
        t for t in all_tracks
        if len(t) >= config.min_temporal_frames and t[-1][0] == last_frame_idx
    ]

    if not valid_tracks:
        return DetectionResult(
            rain_incoming=False,
            arrival_time=None,
            confidence=confidence,
            frame_count=frame_count,
            max_approaching_intensity=0.0,
        )

    # 6. For each valid track: compute velocity, check coherence, project forward
    loc_row, loc_col = _location_to_pixel(location[0], location[1], bounds, W, H)
    km_per_row, km_per_col = _pixel_size_km(bounds, W, H)
    proximity_px = config.proximity_radius_km / ((km_per_row + km_per_col) / 2)
    last_frame_time = frames[-1].timestamp

    earliest_arrival: datetime | None = None
    max_intensity: float = 0.0
    last_grid = grids[-1]
    last_labeled = per_frame_labeled[-1]
    tracked_cells: list[TrackedCell] = []

    # Pre-compute confidence map for the last frame (for per-cell confidence)
    last_conf_map = (
        confidence_maps[-1]
        if confidence_maps is not None and len(confidence_maps) == len(frames)
        else None
    )

    def _cell_mean_confidence(label: int) -> float:
        """Compute mean QC confidence for pixels belonging to a cell."""
        if last_conf_map is None:
            return 1.0
        cell_mask = last_labeled == label
        cell_conf = last_conf_map[cell_mask]
        if cell_conf.size == 0:
            return 1.0
        return float(cell_conf.mean())

    # Minimum cell confidence to report rain_incoming
    _MIN_CELL_CONFIDENCE = 0.4

    for track in valid_tracks:
        # Check current position first - overhead rain triggers immediately
        _, last_label, final_centroid = track[-1]
        cur_row, cur_col = final_centroid
        dist_to_loc = math.sqrt((cur_row - loc_row) ** 2 + (cur_col - loc_col) ** 2)
        cell_conf = _cell_mean_confidence(last_label)

        if dist_to_loc <= proximity_px:
            # Rain is already at the location - report as arriving now
            if cell_conf >= _MIN_CELL_CONFIDENCE:
                arrival_dt = last_frame_time
                if earliest_arrival is None or arrival_dt < earliest_arrival:
                    earliest_arrival = arrival_dt
                cell_pixels = last_grid[last_labeled == last_label]
                if cell_pixels.size > 0:
                    max_intensity = max(max_intensity, float(cell_pixels.max()))

            # Build TrackedCell - compute velocity if we have enough track points
            cell_lat, cell_lon = _pixel_to_location(cur_row, cur_col, bounds, W, H)
            speed_kmh, bearing = _track_velocity_kmh_bearing(
                track, frames, km_per_row, km_per_col,
            )
            tracked_cells.append(TrackedCell(
                lat=cell_lat, lon=cell_lon,
                velocity_kmh=speed_kmh, bearing=bearing,
                confidence=cell_conf,
            ))
            continue

        # For cells not yet overhead, compute velocity and require directional coherence
        velocities: list[tuple[float, float]] = []
        for i in range(len(track) - 1):
            fi, _, ci = track[i]
            fj, _, cj = track[i + 1]
            t_delta = (frames[fj].timestamp - frames[fi].timestamp).total_seconds()
            if t_delta > 0:
                velocities.append(estimate_velocity(ci, cj, t_delta))

        if not velocities:
            continue

        if not is_directionally_coherent(velocities, config.max_angular_variance):
            continue

        vy = sum(v[0] for v in velocities) / len(velocities)
        vx = sum(v[1] for v in velocities) / len(velocities)

        # Project cell forward in 60-second steps up to the lookahead limit
        step_s = 60.0
        t = 0.0
        arrival_seconds: float | None = None

        while t <= config.lookahead_seconds:
            proj_row = cur_row + vy * t
            proj_col = cur_col + vx * t
            d = math.sqrt((proj_row - loc_row) ** 2 + (proj_col - loc_col) ** 2)
            if d <= proximity_px:
                arrival_seconds = t
                break
            t += step_s

        if arrival_seconds is not None and cell_conf >= _MIN_CELL_CONFIDENCE:
            arrival_dt = last_frame_time + timedelta(seconds=arrival_seconds)
            if earliest_arrival is None or arrival_dt < earliest_arrival:
                earliest_arrival = arrival_dt
            cell_pixels = last_grid[last_labeled == last_label]
            if cell_pixels.size > 0:
                max_intensity = max(max_intensity, float(cell_pixels.max()))

        # All coherent tracks become tracked cells (even if they won't arrive
        # within the lookahead - they're still real rain the user should see)
        cell_lat, cell_lon = _pixel_to_location(cur_row, cur_col, bounds, W, H)
        vy_km_s = vy * km_per_row
        vx_km_s = vx * km_per_col
        speed_kmh_val = math.sqrt(vy_km_s**2 + vx_km_s**2) * 3600
        bearing_val = math.degrees(math.atan2(vx_km_s, -vy_km_s)) % 360
        tracked_cells.append(TrackedCell(
            lat=cell_lat, lon=cell_lon,
            velocity_kmh=speed_kmh_val, bearing=bearing_val,
            confidence=cell_conf,
        ))

    rain_incoming = earliest_arrival is not None
    return DetectionResult(
        rain_incoming=rain_incoming,
        arrival_time=earliest_arrival,
        confidence=confidence,
        frame_count=frame_count,
        max_approaching_intensity=max_intensity if rain_incoming else 0.0,
        tracked_cells=tracked_cells,
        last_labeled=last_labeled,
    )
