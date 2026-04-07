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


def _location_to_pixel(
    lat: float, lon: float, bounds: BoundingBox, width: int, height: int
) -> tuple[int, int]:
    """Convert a lat/lon to (row, col) in a grid with the given bounds."""
    col = int((lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * width)
    row = int((bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * height)
    return row, col


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
) -> DetectionResult:
    """Run the full rain detection pipeline over a list of radar frames."""
    frame_count = len(frames)

    if frame_count < 2:
        return DetectionResult(
            rain_incoming=False,
            arrival_time=None,
            confidence=Confidence.UNAVAILABLE,
            frame_count=frame_count,
        )

    confidence = Confidence.DEGRADED if frame_count < 3 else Confidence.NORMAL

    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    # 1. Extract intensity grids from all frames
    grids = [f.get_intensity_grid(bounds, W, H) for f in frames]

    # 2. Threshold + spatial filter each frame independently
    masks = [threshold_intensity(g, config.intensity_threshold) for g in grids]
    masks = [filter_by_area(m, config.min_cell_area_pixels) for m in masks]

    # 3. Label each frame independently
    per_frame_centroids: list[dict[int, tuple[float, float]]] = []
    for mask in masks:
        labeled, _ = ndimage.label(mask)
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
        )

    # 6. For each valid track: compute velocity, check coherence, project forward
    loc_row, loc_col = _location_to_pixel(location[0], location[1], bounds, W, H)
    km_per_row, km_per_col = _pixel_size_km(bounds, W, H)
    proximity_px = config.proximity_radius_km / ((km_per_row + km_per_col) / 2)
    last_frame_time = frames[-1].timestamp

    earliest_arrival: datetime | None = None

    for track in valid_tracks:
        # Compute per-step velocities from consecutive track entries
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

        # Final centroid of the track (current position)
        _, _, final_centroid = track[-1]
        cur_row, cur_col = final_centroid

        # Skip cells already at the location (rain already overhead)
        dist_to_loc = math.sqrt((cur_row - loc_row) ** 2 + (cur_col - loc_col) ** 2)
        if dist_to_loc <= proximity_px:
            continue

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

        if arrival_seconds is not None:
            arrival_dt = last_frame_time + timedelta(seconds=arrival_seconds)
            if earliest_arrival is None or arrival_dt < earliest_arrival:
                earliest_arrival = arrival_dt

    rain_incoming = earliest_arrival is not None
    return DetectionResult(
        rain_incoming=rain_incoming,
        arrival_time=earliest_arrival,
        confidence=confidence,
        frame_count=frame_count,
    )
