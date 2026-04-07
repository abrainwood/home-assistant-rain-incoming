from __future__ import annotations

import numpy as np
from scipy import ndimage


def threshold_intensity(grid: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a float intensity grid to a boolean precipitation mask."""
    return grid >= threshold


def filter_by_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    result = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_features + 1):
        component = labeled == label
        if component.sum() >= min_area:
            result |= component
    return result


def filter_by_temporal_persistence(
    frames: list[np.ndarray], min_frames: int
) -> np.ndarray:
    """
    Keep only pixels that appear in at least min_frames of the given frames
    AND are present in the most recent (last) frame.

    This ensures the output reflects current state while requiring historical
    persistence as evidence against noise.
    """
    if not frames:
        return np.zeros((0, 0), dtype=bool)
    stack = np.stack(frames, axis=0)
    presence_count = stack.sum(axis=0)
    persistent = presence_count >= min_frames
    return frames[-1] & persistent
