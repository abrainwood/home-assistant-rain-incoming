from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def score_texture(
    grid: np.ndarray,
    kernel_size: int = 5,
    low_threshold: float = 0.05,
    high_threshold: float = 0.15,
) -> np.ndarray:
    """Score each pixel by local texture smoothness.

    Smooth areas (low local std dev) score high - they look like real rain.
    Speckly areas (high local std dev) score low - they look like clutter.
    No-echo pixels (grid == 0) always get 0.0.

    Returns a float32 array with the same shape as grid, values in [0.0, 1.0].
    """
    grid = np.asarray(grid, dtype=np.float32)
    mean = uniform_filter(grid, size=kernel_size)
    mean_sq = uniform_filter(grid**2, size=kernel_size)
    # Clamp to zero: float32 rounding can make (mean_sq - mean**2) slightly negative
    local_std = np.sqrt(np.maximum(mean_sq - mean**2, 0))

    # Map std dev to confidence: low std = high confidence, high std = low confidence
    score = 1.0 - np.clip(
        (local_std - low_threshold) / (high_threshold - low_threshold), 0, 1
    )

    # Only score where there is echo; no-echo pixels get 0.0
    score = np.where(grid > 0, score, 0.0).astype(np.float32)

    return score
