from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def extract_cell_centroids(labeled: np.ndarray) -> dict[int, tuple[float, float]]:
    """
    Return the centroid (row, col) for each non-zero label in a labeled component array.
    Label 0 (background) is always ignored.

    Uses scipy.ndimage.center_of_mass for a single-pass computation
    instead of per-label np.argwhere scans (~500x faster for many labels).
    """
    n_labels = int(labeled.max())
    if n_labels == 0:
        return {}
    labels = range(1, n_labels + 1)
    coms = ndimage.center_of_mass(labeled > 0, labeled, labels)
    return {
        lbl: (float(r), float(c))
        for lbl, (r, c) in zip(labels, coms)
    }


def match_cells_across_frames(
    centroids_t0: dict[int, tuple[float, float]],
    centroids_t1: dict[int, tuple[float, float]],
    max_distance: float,
) -> list[tuple[int, int]]:
    """
    Match cells between two frames by nearest-centroid proximity.
    Returns a list of (t0_label, t1_label) pairs.
    Each t1 cell is matched at most once (to the closest t0 cell within max_distance).
    """
    if not centroids_t0 or not centroids_t1:  # pragma: no mutate (equivalent - loop handles empty)
        return []

    # Build all candidate pairs with distances
    candidates: list[tuple[float, int, int]] = []
    for t0_label, (r0, c0) in centroids_t0.items():
        for t1_label, (r1, c1) in centroids_t1.items():
            dist = math.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
            if dist <= max_distance:
                candidates.append((dist, t0_label, t1_label))

    candidates.sort()  # closest first

    matched_t0: set[int] = set()
    matched_t1: set[int] = set()
    matches: list[tuple[int, int]] = []

    for dist, t0_label, t1_label in candidates:
        if t0_label in matched_t0 or t1_label in matched_t1:
            continue
        matches.append((t0_label, t1_label))
        matched_t0.add(t0_label)
        matched_t1.add(t1_label)

    return matches


def estimate_velocity(
    centroid_t0: tuple[float, float],
    centroid_t1: tuple[float, float],
    time_delta_seconds: float,
) -> tuple[float, float]:
    """
    Estimate velocity (vy, vx) in pixels/second from two centroid positions.
    vy > 0 means moving south (increasing row), vx > 0 means moving east (increasing col).
    """
    dy = centroid_t1[0] - centroid_t0[0]
    dx = centroid_t1[1] - centroid_t0[1]
    return dy / time_delta_seconds, dx / time_delta_seconds


def estimate_acceleration(
    velocities: list[tuple[float, float]],
    time_deltas: list[float],
) -> tuple[float, float]:
    """Estimate acceleration (ay, ax) in pixels/second^2 from a velocity series.

    Each velocity is (vy, vx) in pixels/second, and each time_delta is the
    interval in seconds between the frame pairs that produced the velocity.
    Acceleration is computed as the mean rate of velocity change across
    consecutive velocity estimates.

    Returns (0.0, 0.0) if fewer than 2 velocities are provided.
    """
    if len(velocities) < 2:
        return 0.0, 0.0

    ay_accum = 0.0
    ax_accum = 0.0
    count = 0
    for i in range(len(velocities) - 1):
        dt = time_deltas[i + 1]
        if dt > 0:
            ay_accum += (velocities[i + 1][0] - velocities[i][0]) / dt
            ax_accum += (velocities[i + 1][1] - velocities[i][1]) / dt
            count += 1

    if count == 0:
        return 0.0, 0.0
    return ay_accum / count, ax_accum / count


def is_directionally_coherent(
    velocities: list[tuple[float, float]],
    max_angular_variance: float,
) -> bool:
    """
    Return True if velocity directions are consistent (low angular variance).
    All-zero velocities (stationary noise) are considered incoherent.
    max_angular_variance is in radians.
    """
    if not velocities:
        return False

    # Reject all-zero velocities - stationary echoes have no meaningful direction
    non_zero = [(vy, vx) for vy, vx in velocities if abs(vy) > 1e-9 or abs(vx) > 1e-9]
    if not non_zero:
        return False

    if len(non_zero) == 1:
        return True

    angles = [math.atan2(vy, vx) for vy, vx in non_zero]

    # Compute circular mean and variance
    sin_mean = sum(math.sin(a) for a in angles) / len(angles)
    cos_mean = sum(math.cos(a) for a in angles) / len(angles)
    # R is the mean resultant length: 1 = all same direction, 0 = random
    r = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
    # Circular variance = 1 - R; smaller = more coherent
    circular_variance = 1.0 - r

    return circular_variance <= max_angular_variance
