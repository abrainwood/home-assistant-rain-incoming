from __future__ import annotations

import numpy as np


def score_clutter(grid: np.ndarray, clutter_freq: np.ndarray) -> np.ndarray:
    """Score pixels inversely to their clutter frequency.

    Pixels at locations that frequently show echo (clutter hotspots) get low
    confidence. Pixels at clean locations get high confidence.
    Non-echo pixels always get 0.0.

    Returns float32 array with values in [0.0, 1.0].
    """
    grid = np.asarray(grid, dtype=np.float32)
    clutter_freq = np.asarray(clutter_freq, dtype=np.float32)

    score = 1.0 - clutter_freq
    score = np.where(grid > 0, score, 0.0).astype(np.float32)
    return score
