from __future__ import annotations

import numpy as np


def score_temporal(
    grids: list[np.ndarray],
    threshold: float = 0.01,
    min_frames: int = 3,
) -> np.ndarray:
    """Score each pixel by how persistently it appears across frames.

    Pixels that show up in multiple frames are more likely real precipitation.
    Transient speckle (AP clutter, interference) tends to appear in only one
    or two frames.

    Returns a float32 array with the same shape as the last grid, values
    in [0.0, 1.0]. Pixels not present in the latest grid get 0.0.
    Returns an empty array if grids is empty.
    """
    if not grids:
        return np.empty(0, dtype=np.float32)

    latest = np.asarray(grids[-1], dtype=np.float32)
    latest_mask = latest > threshold

    # Count per-pixel presence across all frames
    count = np.zeros_like(latest, dtype=np.float32)
    for grid in grids:
        count += (np.asarray(grid, dtype=np.float32) > threshold).astype(np.float32)

    confidence = np.clip(count / min_frames, 0.0, 1.0)

    # Only score where the latest grid has echo
    confidence = np.where(latest_mask, confidence, 0.0).astype(np.float32)

    return confidence
