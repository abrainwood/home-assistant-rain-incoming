"""Convert golden data numpy grids to RainViewer-compatible tile PNGs."""
from __future__ import annotations

import numpy as np
from PIL import Image
from io import BytesIO

# RainViewer scheme 6 color table (matches PRECIP_COLOURS in rainviewer.py)
_COLOUR_LUT = np.array([
    (0, 91, 142),     # very light
    (0, 119, 170),    # light
    (0, 154, 213),    # light-moderate
    (81, 197, 232),   # moderate
    (255, 224, 0),    # moderate
    (255, 170, 0),    # moderate-heavy
    (255, 68, 0),     # heavy
    (193, 0, 0),      # very heavy
    (255, 119, 255),  # extreme
], dtype=np.uint8)

_INTENSITY_LUT = np.array(
    [0.10, 0.18, 0.28, 0.38, 0.50, 0.63, 0.77, 0.88, 1.00],
    dtype=np.float32,
)

# Precompute bin boundaries - each intensity maps to the nearest LUT entry
_BOUNDARIES: list[tuple[float, float]] = []
for _idx in range(len(_INTENSITY_LUT)):
    _lo = (
        (_INTENSITY_LUT[_idx - 1] + _INTENSITY_LUT[_idx]) / 2
        if _idx > 0
        else 0.001
    )
    _hi = (
        (_INTENSITY_LUT[_idx] + _INTENSITY_LUT[_idx + 1]) / 2
        if _idx < len(_INTENSITY_LUT) - 1
        else 2.0
    )
    _BOUNDARIES.append((_lo, _hi))


def grid_to_tile_png(grid: np.ndarray) -> bytes:
    """Convert a float32 intensity grid to a RainViewer-compatible RGBA PNG.

    Zero-intensity pixels are fully transparent. Non-zero pixels are mapped
    to the nearest RainViewer scheme 6 colour with full opacity.
    """
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    for idx, (lo, hi) in enumerate(_BOUNDARIES):
        mask = (grid >= lo) & (grid < hi)
        rgba[mask, :3] = _COLOUR_LUT[idx]
        rgba[mask, 3] = 255
    buf = BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()
