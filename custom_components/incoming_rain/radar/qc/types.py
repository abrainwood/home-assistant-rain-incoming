from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QCConfig:
    """Configuration for radar quality-control scoring."""

    texture_kernel_size: int = 5
    texture_high_threshold: float = 0.15
    texture_low_threshold: float = 0.05
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "texture": 0.30, "temporal": 0.25, "clutter": 0.20,
            "speed": 0.15, "motion": 0.10,
        }
    )


@dataclass(frozen=True)
class ConfidenceMap:
    """Per-pixel confidence scores from the QC pipeline."""

    confidence: np.ndarray  # float32 (H, W), 0.0-1.0
    factor_scores: dict[str, np.ndarray]
