from __future__ import annotations

import logging

import numpy as np

from .clutter import score_clutter
from .motion_consistency import score_motion_consistency
from .speed_sanity import score_speed
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
    model_precipitation_mm: float | None = None,
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

    # Model-based modifier: asymmetric treatment.
    # "Model says rain" is a strong positive (boost). "Model says dry" is a
    # weak negative (mild skepticism) - radar might see virga or cells about
    # to reach the ground that the model hasn't picked up yet.
    if model_precipitation_mm is not None:
        if model_precipitation_mm >= 0.5:
            confidence *= 1.15  # model confirms rain - boost
            np.clip(confidence, 0, 1, out=confidence)
        elif model_precipitation_mm > 0.01:
            pass  # trace amounts - no change either way
        else:
            confidence *= 0.85  # model says dry - mild skepticism only

    return ConfidenceMap(confidence=confidence, factor_scores=factor_scores)


def refine_confidence_post_detection(
    pre_confidence: np.ndarray,
    cells: list,
    labeled: np.ndarray,
    config: QCConfig | None = None,
) -> np.ndarray:
    """Apply post-detection QC factors (speed, motion) to refine confidence.

    The pre-detection confidence map is combined with speed and motion
    consistency scores using a weighted average of all five factors.

    Parameters
    ----------
    pre_confidence: the pre-detection confidence map (texture + temporal + clutter).
    cells: list of TrackedCell from the detector.
    labeled: labeled array from the last frame's cell segmentation.
    config: QC configuration. Uses defaults if None.

    Returns float32 confidence array.
    """
    if config is None:
        config = QCConfig()

    speed = score_speed(cells, labeled)
    motion = score_motion_consistency(cells, labeled)

    # Pre-detection weights from config.weights, post from config.post_weights
    pre_weight = sum(config.weights.values())
    post_scores = {"speed": speed, "motion": motion}

    post_weight = 0.0
    post_combined = np.zeros_like(pre_confidence, dtype=np.float32)
    for name, w in config.post_weights.items():
        if name in post_scores and w > 0:
            post_combined += w * post_scores[name]
            post_weight += w

    total_weight = pre_weight + post_weight
    if total_weight <= 0:
        return pre_confidence

    # Re-weight: pre_confidence already has its factors normalised,
    # so scale it by its share of the total weight
    refined = (pre_weight * pre_confidence + post_combined) / total_weight

    return np.clip(refined, 0.0, 1.0).astype(np.float32)
