from __future__ import annotations

import logging

import numpy as np

from .temporal import score_temporal
from .texture import score_texture
from .types import ConfidenceMap, QCConfig

_LOGGER = logging.getLogger(__name__)


def compute_confidence_map(
    grid: np.ndarray,
    config: QCConfig | None = None,
    grids: list[np.ndarray] | None = None,
) -> ConfidenceMap:
    """Compute a per-pixel confidence map by combining QC factor scores.

    Parameters
    ----------
    grid: the current (latest) frame's intensity grid.
    config: QC configuration. Uses defaults if None.
    grids: optional list of intensity grids (all frames) for temporal scoring.
        When provided, temporal persistence scoring is computed and combined
        with texture.
    """
    if config is None:
        config = QCConfig()

    texture = score_texture(
        grid,
        kernel_size=config.texture_kernel_size,
        low_threshold=config.texture_low_threshold,
        high_threshold=config.texture_high_threshold,
    )

    factor_scores: dict[str, np.ndarray] = {"texture": texture}

    if grids is not None and len(grids) > 0:
        temporal = score_temporal(grids)
        factor_scores["temporal"] = temporal

    # Weighted combination of all available factors
    unmatched = set(config.weights.keys()) - set(factor_scores.keys())
    if unmatched:
        _LOGGER.warning("QC weight keys have no matching factor score: %s", unmatched)

    confidence = np.zeros_like(texture, dtype=np.float32)
    total_weight = 0.0
    for name, weight in config.weights.items():
        if name in factor_scores:
            confidence += weight * factor_scores[name]
            total_weight += weight

    if total_weight > 0:
        confidence /= total_weight

    return ConfidenceMap(confidence=confidence, factor_scores=factor_scores)
