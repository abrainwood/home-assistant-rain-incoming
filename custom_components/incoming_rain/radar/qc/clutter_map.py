from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class ClutterMap:
    """Running statistics for identifying static clutter locations."""

    echo_count: np.ndarray  # float32 (H, W) - exponentially decayed echo count
    update_count: float  # total updates applied (also decayed)


def update_clutter_map(
    clutter: ClutterMap,
    grid: np.ndarray,
    threshold: float = 0.01,
    decay: float = 0.99966,
) -> None:
    """Update the clutter map in place with exponential decay.

    Each update decays existing counts and adds 1.0 where echo is present.
    This means pixels that consistently show echo accumulate high counts
    relative to the total update count, while transient echoes decay away.
    """
    clutter.echo_count *= decay
    clutter.update_count *= decay

    echo_present = (np.asarray(grid, dtype=np.float32) > threshold).astype(np.float32)
    clutter.echo_count += echo_present
    clutter.update_count += 1.0


def get_clutter_frequency(clutter: ClutterMap) -> np.ndarray:
    """Return per-pixel echo frequency in [0.0, 1.0]."""
    if clutter.update_count <= 0:
        return np.zeros_like(clutter.echo_count, dtype=np.float32)
    freq = clutter.echo_count / clutter.update_count
    return np.clip(freq, 0.0, 1.0).astype(np.float32)


def save_clutter_map(clutter: ClutterMap, path: str) -> None:
    """Persist the clutter map to a .npz file."""
    np.savez_compressed(
        path,
        echo_count=clutter.echo_count,
        update_count=np.array(clutter.update_count, dtype=np.float64),
    )


def load_clutter_map(path: str) -> ClutterMap | None:
    """Load a clutter map from a .npz file. Returns None if file doesn't exist."""
    try:
        data = np.load(path)
        return ClutterMap(
            echo_count=data["echo_count"].astype(np.float32),
            update_count=float(data["update_count"]),
        )
    except (FileNotFoundError, OSError, KeyError):
        _LOGGER.debug("Could not load clutter map from %s", path)
        return None
