"""Shared image comparison helpers for E2E tests."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS

PRECIP_COLOURS_RGB = [[r, g, b] for r, g, b, _ in PRECIP_COLOURS]


def images_differ_significantly(
    rain_gif_bytes: bytes, no_rain_gif_bytes: bytes, threshold: float = 0.01
) -> tuple[bool, float]:
    """Compare two GIFs and return whether they differ significantly.

    Returns (differs, fraction_different).
    If the rain GIF looks identical to the no-rain GIF, rain is NOT being rendered.
    """
    def last_frame_array(gif_bytes):
        img = Image.open(BytesIO(gif_bytes))
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(img.n_frames - 1)
        return np.array(img.convert("RGB")).astype(np.float32)

    rain_arr = last_frame_array(rain_gif_bytes)
    clean_arr = last_frame_array(no_rain_gif_bytes)

    # Per-pixel difference
    diff = np.abs(rain_arr - clean_arr)
    # A pixel "differs" if any channel changes by more than 20
    pixel_differs = diff.max(axis=2) > 20
    fraction = pixel_differs.sum() / (rain_arr.shape[0] * rain_arr.shape[1])

    return fraction > threshold, float(fraction)


def gif_has_precipitation_pixels(
    gif_bytes: bytes, threshold: float = 0.005, max_colour_dist: float = 40.0
) -> tuple[bool, float]:
    """Check if a GIF contains visible precipitation-coloured pixels.

    Matches pixels against known RainViewer precipitation colours using
    L2 distance. Returns (has_precip, fraction_of_pixels).
    """
    _precip_colours_rgb = np.array(PRECIP_COLOURS_RGB, dtype=np.float32)

    img = Image.open(BytesIO(gif_bytes))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    arr = np.array(img.convert("RGB")).astype(np.float32)

    # Compute L2 distance from each pixel to each precipitation colour
    diff = arr[:, :, np.newaxis, :] - _precip_colours_rgb[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W, N)
    best_dist = distances.min(axis=-1)  # (H, W)

    precip_mask = best_dist < max_colour_dist
    fraction = precip_mask.sum() / (arr.shape[0] * arr.shape[1])
    return fraction > threshold, float(fraction)
