from __future__ import annotations

import numpy as np

from ..detector import TrackedCell


def score_speed(
    cells: list[TrackedCell],
    labeled: np.ndarray,
    min_kmh: float = 3.0,
    max_kmh: float = 130.0,
) -> np.ndarray:
    """Score pixels based on the speed of the cell they belong to.

    Cells moving at realistic storm speeds score high. Stationary echoes
    (likely clutter) and unrealistically fast echoes (likely artifacts) score low.
    Pixels not belonging to any cell get a neutral 0.5.

    Parameters
    ----------
    cells: tracked cells from the detector. Cell index i corresponds to
        label (i+1) in the labeled array.
    labeled: integer array where each pixel's value is the cell label
        (0 = background, 1 = first cell, etc.)

    Returns float32 array with values in [0.0, 1.0].
    """
    score = np.full(labeled.shape, 0.5, dtype=np.float32)

    for i, cell in enumerate(cells):
        label = i + 1
        mask = labeled == label

        if cell.velocity_kmh < min_kmh:
            cell_score = 0.2  # stationary
        elif cell.velocity_kmh > max_kmh:
            cell_score = 0.1  # too fast
        else:
            cell_score = 1.0  # sane range

        score[mask] = cell_score

    return score
