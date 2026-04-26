from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from scipy import ndimage

from ..providers.base import BoundingBox, RadarFrame
from .filters import filter_by_area, threshold_intensity
from .motion import (
    estimate_acceleration,
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
    use_acceleration: bool = False
    use_intensity_trend: bool = False
    frame_scale_by_lookahead: bool = False


@dataclass
class DetectionResult:
    rain_incoming: bool
    arrival_time: datetime | None
    confidence: Confidence
    frame_count: int
    max_approaching_intensity: float = 0.0
    rain_at_location: bool = False
    tracked_cells: list[TrackedCell] = field(default_factory=list)
    last_labeled: np.ndarray | None = None  # labeled array of last frame's cells


@dataclass
class FrameDiagnostic:
    """Diagnostic information for a single radar frame."""

    cell_count: int


@dataclass
class TrackDiagnostic:
    """Diagnostic information for a single tracked cell."""

    frame_indices: list[int]
    status: str
    reason: str | None = None


@dataclass
class DecisionDiagnostic:
    """The final decision made by detect()."""

    rain_incoming: bool
    arrival_minutes: float | None
    rain_at_location: bool


@dataclass
class DiagnosticTrace:
    """Accumulates per-frame diagnostic data during a detect() call."""

    frames: list[FrameDiagnostic] = field(default_factory=list)
    tracks: list[TrackDiagnostic] = field(default_factory=list)
    decision: DecisionDiagnostic | None = None


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


# How many recent frames to check for rain directly at the location pixel.
# We check the last 2 frames: the final frame and the one before it. This
# catches rain that was overhead moments ago but has just moved/dissipated,
# without reaching too far back (which would false-positive on receding rain).
_RECENT_FRAMES_FOR_LOCATION_CHECK = 2


def _check_rain_at_location(
    effective_grids: list[np.ndarray],
    intensity_threshold: float,
    loc_row: int,
    loc_col: int,
    proximity_px: int = 10,
    confidence_maps: list[np.ndarray] | None = None,
    min_confidence: float = 0.35,
) -> tuple[int, float] | None:
    """Check if rain exists near the location in recent frames.

    Scans a neighborhood (proximity_px radius) around the location pixel
    in the last few effective grids. This catches rain that's nearby but
    not on the exact pixel - critical because a 1km pixel grid can miss
    rain that's clearly overhead on a wider view.

    Returns (frame_index, max_intensity) for the most recent frame with rain,
    or None if no rain found.
    """
    h, w = effective_grids[0].shape
    if not (0 <= loc_row < h and 0 <= loc_col < w):
        return None

    r1 = max(0, loc_row - proximity_px)
    r2 = min(h, loc_row + proximity_px + 1)
    c1 = max(0, loc_col - proximity_px)
    c2 = min(w, loc_col + proximity_px + 1)

    n_recent = min(_RECENT_FRAMES_FOR_LOCATION_CHECK, len(effective_grids))
    for i in range(len(effective_grids) - 1, len(effective_grids) - 1 - n_recent, -1):
        neighborhood = effective_grids[i][r1:r2, c1:c2]
        max_intensity = float(neighborhood.max())
        if max_intensity >= intensity_threshold:
            # Check confidence of the brightest pixel in the neighborhood
            if confidence_maps is not None and i < len(confidence_maps):
                conf_hood = confidence_maps[i][r1:r2, c1:c2]
                # Use confidence at the pixel with max intensity
                max_pos = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
                pixel_conf = float(conf_hood[max_pos])
                if pixel_conf < min_confidence:
                    continue
            return i, max_intensity
    return None


# Minimum cell confidence to report rain_incoming
_MIN_CELL_CONFIDENCE = 0.35

# Intensity ratio (final / initial frame in track) below which a cell is
# considered to be dissipating and is suppressed when use_intensity_trend
# is enabled. 0.5 means "intensity has at least halved across the track."
_INTENSITY_DECLINE_THRESHOLD = 0.5


def _cell_mean_confidence(
    label: int,
    last_labeled: np.ndarray,
    last_conf_map: np.ndarray | None,
) -> float:
    """Compute mean QC confidence for pixels belonging to a cell."""
    if last_conf_map is None:
        return 1.0
    cell_mask = last_labeled == label
    cell_conf = last_conf_map[cell_mask]
    if cell_conf.size == 0:
        return 1.0
    return float(cell_conf.mean())


def _apply_qc_confidence(
    grids: list[np.ndarray],
    confidence_maps: list[np.ndarray] | None,
) -> list[np.ndarray]:
    """Apply QC confidence as a multiplier before thresholding."""
    if confidence_maps is None:
        return grids
    effective_grids = []
    for i, g in enumerate(grids):
        if i < len(confidence_maps) and confidence_maps[i] is not None:
            effective_grids.append(g * confidence_maps[i])
        else:
            effective_grids.append(g)
    return effective_grids


def _evaluate_overhead_cell(
    track: list[_TrackEntry],
    frames: list[RadarFrame],
    config: DetectorConfig,
    km_per_row: float,
    km_per_col: float,
    last_grid: np.ndarray,
    last_labeled: np.ndarray,
    last_conf_map: np.ndarray | None,
) -> tuple[TrackedCell, datetime | None, float] | None:
    """Evaluate a tracked cell that is already within proximity of the location.

    Returns (TrackedCell, arrival_dt_or_None, max_intensity_contribution)
    or None if the cell should be skipped (e.g. speed cap exceeded).
    """
    _, last_label, final_centroid = track[-1]
    cur_row, cur_col = final_centroid
    cell_conf = _cell_mean_confidence(last_label, last_labeled, last_conf_map)
    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    speed_kmh_oh, bearing_oh = _track_velocity_kmh_bearing(
        track, frames, km_per_row, km_per_col,
    )

    if speed_kmh_oh > config.max_storm_speed_kmh:
        return None

    arrival_dt: datetime | None = None
    intensity_contribution = 0.0

    if cell_conf >= _MIN_CELL_CONFIDENCE:
        arrival_dt = frames[-1].timestamp
        cell_pixels = last_grid[last_labeled == last_label]
        if cell_pixels.size > 0:
            intensity_contribution = float(cell_pixels.max())

    cell_lat, cell_lon = _pixel_to_location(cur_row, cur_col, bounds, W, H)
    tracked = TrackedCell(
        lat=cell_lat, lon=cell_lon,
        velocity_kmh=speed_kmh_oh, bearing=bearing_oh,
        confidence=cell_conf,
    )
    return tracked, arrival_dt, intensity_contribution


def _intensity_at_track_entry(
    track: list[_TrackEntry],
    idx: int,
    per_frame_grids: list[np.ndarray],
    per_frame_labeled_all: list[np.ndarray],
) -> float:
    """Return the max pixel intensity of the cell at track[idx] in its frame."""
    frame_idx, label, _ = track[idx]
    grid = per_frame_grids[frame_idx]
    labeled = per_frame_labeled_all[frame_idx]
    cell_pixels = grid[labeled == label]
    if cell_pixels.size == 0:
        return 0.0
    return float(cell_pixels.max())


def _evaluate_approaching_cell(
    track: list[_TrackEntry],
    frames: list[RadarFrame],
    config: DetectorConfig,
    loc_row: int,
    loc_col: int,
    proximity_px: float,
    km_per_row: float,
    km_per_col: float,
    last_grid: np.ndarray,
    last_labeled: np.ndarray,
    last_conf_map: np.ndarray | None,
    last_frame_time: datetime,
    per_frame_grids: list[np.ndarray] | None = None,
    per_frame_labeled_all: list[np.ndarray] | None = None,
) -> tuple[TrackedCell, datetime | None, float] | None:
    """Evaluate a tracked cell that is not yet overhead - approaching.

    Computes velocity, checks coherence, does forward projection and
    closing-distance fallback. Returns (TrackedCell, arrival_dt_or_None,
    max_intensity_contribution) or None if the cell should be skipped.
    """
    _, last_label, final_centroid = track[-1]
    cur_row, cur_col = final_centroid
    cell_conf = _cell_mean_confidence(last_label, last_labeled, last_conf_map)
    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    velocities: list[tuple[float, float]] = []
    for i in range(len(track) - 1):
        fi, _, ci = track[i]
        fj, _, cj = track[i + 1]
        t_delta = (frames[fj].timestamp - frames[fi].timestamp).total_seconds()
        if t_delta > 0:
            velocities.append(estimate_velocity(ci, cj, t_delta))

    if not velocities:
        return None

    if not is_directionally_coherent(velocities, config.max_angular_variance):
        return None

    vy = sum(v[0] for v in velocities) / len(velocities)
    vx = sum(v[1] for v in velocities) / len(velocities)

    vy_km_s = vy * km_per_row
    vx_km_s = vx * km_per_col
    speed_kmh_val = math.sqrt(vy_km_s**2 + vx_km_s**2) * 3600
    if speed_kmh_val > config.max_storm_speed_kmh:
        return None

    # Suppress dissipating cells when intensity trend check is enabled
    if config.use_intensity_trend and per_frame_grids is not None and per_frame_labeled_all is not None:
        initial_intensity = _intensity_at_track_entry(track, 0, per_frame_grids, per_frame_labeled_all)
        final_intensity = _intensity_at_track_entry(track, -1, per_frame_grids, per_frame_labeled_all)
        if initial_intensity > 0 and final_intensity / initial_intensity < _INTENSITY_DECLINE_THRESHOLD:
            return None

    # Compute acceleration if enabled
    ay, ax = 0.0, 0.0
    if config.use_acceleration:
        time_deltas = []
        for i in range(len(track) - 1):
            fi, _, _ = track[i]
            fj, _, _ = track[i + 1]
            time_deltas.append(
                (frames[fj].timestamp - frames[fi].timestamp).total_seconds()
            )
        ay, ax = estimate_acceleration(velocities, time_deltas)

    # Project cell forward in 60-second steps up to the lookahead limit
    step_s = 60.0
    t = 0.0
    arrival_seconds: float | None = None

    while t <= config.lookahead_seconds:
        proj_row = cur_row + vy * t + 0.5 * ay * t * t
        proj_col = cur_col + vx * t + 0.5 * ax * t * t
        d = math.sqrt((proj_row - loc_row) ** 2 + (proj_col - loc_col) ** 2)
        if d <= proximity_px:
            arrival_seconds = t
            break
        t += step_s

    # Fallback: if velocity projection didn't find arrival, check whether
    # the cell has been consistently closing distance to the location across
    # its track history.
    if arrival_seconds is None:
        arrival_seconds = _closing_distance_fallback(
            track, frames, config, loc_row, loc_col, km_per_row, km_per_col,
        )

    arrival_dt: datetime | None = None
    intensity_contribution = 0.0
    if arrival_seconds is not None and cell_conf >= _MIN_CELL_CONFIDENCE:
        arrival_dt = last_frame_time + timedelta(seconds=arrival_seconds)
        cell_pixels = last_grid[last_labeled == last_label]
        if cell_pixels.size > 0:
            intensity_contribution = float(cell_pixels.max())

    cell_lat, cell_lon = _pixel_to_location(cur_row, cur_col, bounds, W, H)
    bearing_val = math.degrees(math.atan2(vx_km_s, -vy_km_s)) % 360
    tracked = TrackedCell(
        lat=cell_lat, lon=cell_lon,
        velocity_kmh=speed_kmh_val, bearing=bearing_val,
        confidence=cell_conf,
    )
    return tracked, arrival_dt, intensity_contribution


def _closing_distance_fallback(
    track: list[_TrackEntry],
    frames: list[RadarFrame],
    config: DetectorConfig,
    loc_row: int,
    loc_col: int,
    km_per_row: float,
    km_per_col: float,
) -> float | None:
    """Check if a cell is closing distance and estimate arrival time.

    Storm cells can approach obliquely (the centroid velocity vector doesn't
    point at the location) while the cell edge still advances toward it.

    Returns arrival_seconds or None.
    """
    per_step_dists = [
        math.sqrt(
            (entry[2][0] - loc_row) ** 2 + (entry[2][1] - loc_col) ** 2
        )
        * ((km_per_row + km_per_col) / 2)
        for entry in track
    ]

    # Verify cell is still closing in recent frames (not reversing).
    still_closing = (
        len(per_step_dists) < 3
        or per_step_dists[-1] < per_step_dists[-2]
    )

    if not still_closing:
        return None

    first_dist_km = per_step_dists[0]
    final_dist_km = per_step_dists[-1]
    track_duration_s = (
        frames[track[-1][0]].timestamp - frames[track[0][0]].timestamp
    ).total_seconds()

    if track_duration_s <= 0 or first_dist_km <= final_dist_km:
        return None

    closing_km = first_dist_km - final_dist_km
    closing_rate_kmh = (closing_km / track_duration_s) * 3600

    if closing_km < 20.0 or closing_rate_kmh > config.max_storm_speed_kmh:
        return None

    remaining_km = final_dist_km - config.proximity_radius_km
    if remaining_km <= 0:
        return 0.0

    eta_s = (remaining_km / closing_rate_kmh) * 3600
    if eta_s <= config.lookahead_seconds:
        return eta_s

    return None


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
        match_map: dict[int, int] = dict(matches)
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


@dataclass
class _FrameAnalysis:
    """Intermediate results from per-frame processing."""

    grids: list[np.ndarray]
    effective_grids: list[np.ndarray]
    per_frame_labeled: list[np.ndarray]
    per_frame_centroids: list[dict[int, tuple[float, float]]]


def _prepare_frame_analysis(
    frames: list[RadarFrame],
    config: DetectorConfig,
    confidence_maps: list[np.ndarray] | None,
) -> _FrameAnalysis:
    """Steps 1-3: extract grids, apply QC, threshold+filter, label each frame."""
    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    grids = [f.get_intensity_grid(bounds, W, H) for f in frames]
    effective_grids = _apply_qc_confidence(grids, confidence_maps)

    masks = [threshold_intensity(g, config.intensity_threshold) for g in effective_grids]
    masks = [filter_by_area(m, config.min_cell_area_pixels) for m in masks]

    per_frame_labeled: list[np.ndarray] = []
    per_frame_centroids: list[dict[int, tuple[float, float]]] = []
    for mask in masks:
        labeled, _ = ndimage.label(mask)
        per_frame_labeled.append(labeled)
        per_frame_centroids.append(extract_cell_centroids(labeled))

    return _FrameAnalysis(grids, effective_grids, per_frame_labeled, per_frame_centroids)


def _no_tracks_result(
    rain_at_location: tuple[int, float] | None,
    frames: list[RadarFrame],
    confidence: Confidence,
    frame_count: int,
    last_labeled: np.ndarray | None,
) -> DetectionResult:
    """Build the result for the case where no valid cell tracks were found."""
    if rain_at_location is not None:
        rain_frame_idx, rain_intensity = rain_at_location
        return DetectionResult(
            rain_incoming=True,
            arrival_time=frames[rain_frame_idx].timestamp,
            confidence=confidence,
            frame_count=frame_count,
            max_approaching_intensity=rain_intensity,
            rain_at_location=True,
            tracked_cells=[],
            last_labeled=last_labeled,
        )
    return DetectionResult(
        rain_incoming=False,
        arrival_time=None,
        confidence=confidence,
        frame_count=frame_count,
        max_approaching_intensity=0.0,
    )


def _process_tracks(
    valid_tracks: list[list[_TrackEntry]],
    frames: list[RadarFrame],
    config: DetectorConfig,
    loc_row: int,
    loc_col: int,
    proximity_px: float,
    km_per_row: float,
    km_per_col: float,
    last_grid: np.ndarray,
    last_labeled: np.ndarray,
    last_conf_map: np.ndarray | None,
    per_frame_grids: list[np.ndarray] | None = None,
    per_frame_labeled_all: list[np.ndarray] | None = None,
) -> tuple[list[TrackedCell], datetime | None, float]:
    """Evaluate each valid track and accumulate earliest arrival and max intensity."""
    last_frame_time = frames[-1].timestamp
    earliest_arrival: datetime | None = None
    max_intensity: float = 0.0
    tracked_cells: list[TrackedCell] = []

    for track in valid_tracks:
        _, _, final_centroid = track[-1]
        cur_row, cur_col = final_centroid
        dist_to_loc = math.sqrt((cur_row - loc_row) ** 2 + (cur_col - loc_col) ** 2)

        if dist_to_loc <= proximity_px:
            result = _evaluate_overhead_cell(
                track, frames, config,
                km_per_row, km_per_col, last_grid, last_labeled, last_conf_map,
            )
        else:
            result = _evaluate_approaching_cell(
                track, frames, config, loc_row, loc_col, proximity_px,
                km_per_row, km_per_col, last_grid, last_labeled, last_conf_map,
                last_frame_time,
                per_frame_grids=per_frame_grids,
                per_frame_labeled_all=per_frame_labeled_all,
            )

        if result is None:
            continue

        cell, arrival_dt, intensity_contribution = result
        tracked_cells.append(cell)
        if arrival_dt is not None:
            if earliest_arrival is None or arrival_dt < earliest_arrival:
                earliest_arrival = arrival_dt
            max_intensity = max(max_intensity, intensity_contribution)

    return tracked_cells, earliest_arrival, max_intensity


def _merge_location_rain(
    rain_at_location: tuple[int, float] | None,
    frames: list[RadarFrame],
    earliest_arrival: datetime | None,
    max_intensity: float,
) -> tuple[datetime | None, float]:
    """Fold the direct rain-at-location check into the cell-tracking results."""
    if rain_at_location is None:
        return earliest_arrival, max_intensity
    rain_frame_idx, rain_intensity = rain_at_location
    rain_arrival = frames[rain_frame_idx].timestamp
    if earliest_arrival is None or rain_arrival < earliest_arrival:
        earliest_arrival = rain_arrival
    return earliest_arrival, max(max_intensity, rain_intensity)


def _effective_min_temporal_frames(config: DetectorConfig) -> int:
    """Compute the effective min_temporal_frames, optionally scaled by lookahead.

    Longer-range predictions need more evidence (more frames in the track) to
    avoid false positives from transient signals. The user's explicit
    config.min_temporal_frames is a floor.
    """
    if not config.frame_scale_by_lookahead:
        return config.min_temporal_frames
    lookahead_min = config.lookahead_seconds / 60.0
    if lookahead_min <= 20:
        scaled = 2
    elif lookahead_min <= 40:
        scaled = 3
    else:
        scaled = 4
    return max(config.min_temporal_frames, scaled)


def detect(
    frames: list[RadarFrame],
    location: tuple[float, float],
    config: DetectorConfig,
    confidence_maps: list[np.ndarray] | None = None,
    diagnostics: DiagnosticTrace | None = None,
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
        if diagnostics is not None:
            diagnostics.decision = DecisionDiagnostic(
                rain_incoming=False,
                arrival_minutes=None,
                rain_at_location=False,
            )
        return DetectionResult(
            rain_incoming=False,
            arrival_time=None,
            confidence=Confidence.UNAVAILABLE,
            frame_count=frame_count,
            max_approaching_intensity=0.0,
        )

    confidence = Confidence.DEGRADED if frame_count < 3 else Confidence.NORMAL

    analysis = _prepare_frame_analysis(frames, config, confidence_maps)

    if diagnostics is not None:
        for centroids in analysis.per_frame_centroids:
            diagnostics.frames.append(FrameDiagnostic(cell_count=len(centroids)))

    bounds = config.analysis_bounds
    W, H = config.grid_width, config.grid_height

    loc_row, loc_col = _location_to_pixel(location[0], location[1], bounds, W, H)
    km_per_row, km_per_col = _pixel_size_km(bounds, W, H)
    proximity_px = int(config.proximity_radius_km / ((km_per_row + km_per_col) / 2))

    rain_at_location = _check_rain_at_location(
        analysis.effective_grids, config.intensity_threshold, loc_row, loc_col,
        proximity_px=proximity_px,
        confidence_maps=confidence_maps,
    )

    max_match_dist = max(W, H) * 0.25  # allow up to 25% of grid size per frame
    all_tracks = _build_cell_tracks(analysis.per_frame_centroids, max_match_dist)

    if diagnostics is not None:
        _last_frame_idx = frame_count - 1
        for track in all_tracks:
            if len(track) < config.min_temporal_frames:
                track_status, track_reason = "dropped", "too_short"
            elif track[-1][0] != _last_frame_idx:
                track_status, track_reason = "dropped", "ended_early"
            else:
                track_status, track_reason = "accepted", None
            diagnostics.tracks.append(
                TrackDiagnostic(
                    frame_indices=[entry[0] for entry in track],
                    status=track_status,
                    reason=track_reason,
                )
            )

    last_frame_idx = frame_count - 1
    effective_min = _effective_min_temporal_frames(config)
    valid_tracks = [
        t for t in all_tracks
        if len(t) >= effective_min and t[-1][0] == last_frame_idx
    ]

    last_labeled = analysis.per_frame_labeled[-1]

    if not valid_tracks:
        if diagnostics is not None:
            no_track_rain_incoming = rain_at_location is not None
            no_track_arrival_minutes = (
                (frames[rain_at_location[0]].timestamp - frames[-1].timestamp).total_seconds() / 60.0
                if rain_at_location is not None
                else None
            )
            diagnostics.decision = DecisionDiagnostic(
                rain_incoming=no_track_rain_incoming,
                arrival_minutes=no_track_arrival_minutes,
                rain_at_location=rain_at_location is not None,
            )
        return _no_tracks_result(
            rain_at_location, frames, confidence, frame_count, last_labeled,
        )

    last_conf_map = (
        confidence_maps[-1]
        if confidence_maps is not None and len(confidence_maps) == len(frames)
        else None
    )
    proximity_px_f = config.proximity_radius_km / ((km_per_row + km_per_col) / 2)

    tracked_cells, earliest_arrival, max_intensity = _process_tracks(
        valid_tracks, frames, config,
        loc_row, loc_col, proximity_px_f,
        km_per_row, km_per_col,
        analysis.grids[-1], last_labeled, last_conf_map,
        per_frame_grids=analysis.grids,
        per_frame_labeled_all=analysis.per_frame_labeled,
    )

    earliest_arrival, max_intensity = _merge_location_rain(
        rain_at_location, frames, earliest_arrival, max_intensity,
    )

    rain_incoming = earliest_arrival is not None
    if diagnostics is not None:
        arrival_minutes = None
        if earliest_arrival is not None:
            arrival_minutes = (earliest_arrival - frames[-1].timestamp).total_seconds() / 60.0
        diagnostics.decision = DecisionDiagnostic(
            rain_incoming=rain_incoming,
            arrival_minutes=arrival_minutes,
            rain_at_location=rain_at_location is not None,
        )
    return DetectionResult(
        rain_incoming=rain_incoming,
        arrival_time=earliest_arrival,
        confidence=confidence,
        frame_count=frame_count,
        max_approaching_intensity=max_intensity if rain_incoming else 0.0,
        rain_at_location=rain_at_location is not None,
        tracked_cells=tracked_cells,
        last_labeled=last_labeled,
    )
