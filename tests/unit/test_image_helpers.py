"""Unit tests for tests/e2e/image_helpers.py shared test helpers."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS
from tests.e2e.image_helpers import gif_has_precipitation_pixels


def test_helper_detects_production_palette_pixels():
    """gif_has_precipitation_pixels must return (True, nonzero) for a GIF
    containing a production palette colour, and (False, zero) for a black GIF.

    Regression test for the hardcoded-palette divergence bug: if the helper's
    palette drifts from the production palette, it will fail to detect real
    precipitation pixels and return False for rainy frames.
    """
    # --- positive case: GIF filled with a production precipitation colour ---
    r, g, b, _ = PRECIP_COLOURS[0]
    img_rain = Image.new("RGB", (32, 32), (r, g, b))
    buf = BytesIO()
    img_rain.save(buf, format="GIF")
    buf.seek(0)
    has_precip, fraction = gif_has_precipitation_pixels(buf.getvalue())

    assert has_precip, (
        f"GIF filled with PRECIP_COLOURS[0] ({r},{g},{b}) should be detected "
        f"as precipitation, but got has_precip={has_precip}, fraction={fraction:.4f}"
    )
    assert fraction > 0.0, (
        f"Precipitation fraction should be positive for a rainy GIF, got {fraction:.4f}"
    )

    # --- negative case: pure black GIF should have no precipitation pixels ---
    img_black = Image.new("RGB", (32, 32), (0, 0, 0))
    buf = BytesIO()
    img_black.save(buf, format="GIF")
    buf.seek(0)
    has_precip_black, fraction_black = gif_has_precipitation_pixels(buf.getvalue())

    assert not has_precip_black, (
        f"All-black GIF should not be detected as precipitation, "
        f"got has_precip={has_precip_black}, fraction={fraction_black:.4f}"
    )
