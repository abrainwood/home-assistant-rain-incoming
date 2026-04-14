from __future__ import annotations

import math

import numpy as np


def extract_cell_centroids(labeled: np.ndarray) -> dict[int, tuple[float, float]]:
    """
    Return the centroid (row, col) for each non-zero label in a labeled component array.
    Label 0 (background) is always ignored.
    """
    centroids: dict[int, tuple[float, float]] = {}
    unique_labels = np.unique(labeled)
    for label in unique_labels:
        if label == 0:
            continue
        positions = np.argwhere(labeled == label)
        centroid = positions.mean(axis=0)
        centroids[int(label)] = (float(centroid[0]), float(centroid[1]))
    return centroids


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
    non_zero = [(vy, vx) for vy, vx in velocities if vy != 0.0 or vx != 0.0]
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
