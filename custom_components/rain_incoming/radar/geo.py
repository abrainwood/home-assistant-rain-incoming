"""Shared tile math for slippy-map tile coordinate conversions."""

from __future__ import annotations

import math


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile (x, y) at the given zoom level."""
    x = int((lon + 180.0) / 360.0 * (2**zoom))
    lat_r = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi)
        / 2.0
        * (2**zoom)
    )
    return x, y
