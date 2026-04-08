from __future__ import annotations

import logging

import numpy as np

from .clutter import score_clutter
from .temporal import score_temporal
from .texture import score_texture
from .types import ConfidenceMap, QCConfig

_LOGGER = logging.getLogger(__name__)


def compute_confidence_map(
    grid: np.ndarray,
    config: QCConfig | None = None,
    grids: list[np.ndarray] | None = None,
    clutter_freq: np.ndarray | None = None,
    clutter_maturity: float = 0.0,
) -> ConfidenceMap:
    """Compute a per-pixel confidence map by combining QC factor scores.

    Parameters
    ----------
    grid: the current (latest) frame's intensity grid.
    config: QC configuration. Uses defaults if None.
    grids: optional list of intensity grids (all frames) for temporal scoring.
    clutter_freq: optional per-pixel clutter frequency array from the clutter map.
    clutter_maturity: 0.0-1.0, how mature the clutter map is. When < 1.0 the
        clutter weight is reduced proportionally and redistributed to other factors.
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

    if clutter_freq is not None:
        clutter_score = score_clutter(grid, clutter_freq)
        factor_scores["clutter"] = clutter_score

    # Apply maturity gating to clutter weight
    effective_weights = dict(config.weights)
    if "clutter" in effective_weights and clutter_maturity < 1.0:
        original_clutter_w = effective_weights["clutter"]
        effective_clutter_w = original_clutter_w * clutter_maturity
        redistributed = original_clutter_w - effective_clutter_w
        effective_weights["clutter"] = effective_clutter_w

        # Redistribute to other active factors proportionally
        other_total = sum(
            w for k, w in effective_weights.items()
            if k != "clutter" and k in factor_scores
        )
        if other_total > 0:
            for k in effective_weights:
                if k != "clutter" and k in factor_scores:
                    effective_weights[k] += redistributed * (effective_weights[k] / other_total)

    # Weighted combination of all available factors
    unmatched = set(effective_weights.keys()) - set(factor_scores.keys())
    if unmatched:
        _LOGGER.warning("QC weight keys have no matching factor score: %s", unmatched)

    confidence = np.zeros_like(texture, dtype=np.float32)
    total_weight = 0.0
    for name, weight in effective_weights.items():
        if name in factor_scores:
            confidence += weight * factor_scores[name]
            total_weight += weight

    if total_weight > 0:
        confidence /= total_weight

    return ConfidenceMap(confidence=confidence, factor_scores=factor_scores)
