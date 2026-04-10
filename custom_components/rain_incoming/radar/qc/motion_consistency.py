from __future__ import annotations

import numpy as np

from ..detector import TrackedCell

# Confidence threshold to consider a cell "coherent" - matches the
# detector's _MIN_CELL_CONFIDENCE used for directional coherence gating.
_COHERENCE_THRESHOLD = 0.4


def score_motion_consistency(
    cells: list[TrackedCell],
    labeled: np.ndarray,
) -> np.ndarray:
    """Score pixels based on whether their cell passed directional coherence.

    Cells with high confidence (>= 0.4) are assumed to have passed
    directional coherence checks and score 1.0. Low-confidence cells
    score 0.3. Untracked pixels score 0.5.

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

        if cell.confidence >= _COHERENCE_THRESHOLD:
            cell_score = 1.0
        else:
            cell_score = 0.3

        score[mask] = cell_score

    return score
