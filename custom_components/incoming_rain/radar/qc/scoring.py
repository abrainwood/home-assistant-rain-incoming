from __future__ import annotations

import logging

import numpy as np

from .texture import score_texture
from .types import ConfidenceMap, QCConfig

_LOGGER = logging.getLogger(__name__)


def compute_confidence_map(
    grid: np.ndarray,
    config: QCConfig | None = None,
) -> ConfidenceMap:
    """Compute a per-pixel confidence map by combining QC factor scores.

    For Phase 1, only texture scoring is implemented. Later phases will add
    temporal persistence, static clutter maps, etc.
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

    # Weighted combination of all factors (Phase 1: texture only)
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
