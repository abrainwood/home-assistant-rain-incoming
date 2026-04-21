"""Unit tests for scripts/capture_golden_tiles.py."""
from __future__ import annotations

import sys
import os

# The script does its own sys.path manipulation at import time; replicate that
# here so the import resolves correctly in the test environment too.
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

# Import the script module under test
import importlib
import types

# We import the script's _tile_coords directly.  The script uses a relative
# sys.path.insert at module level which we've already replicated above, so a
# plain import works.
from scripts.capture_golden_tiles import _tile_coords, ZOOM, GRID_HALF  # noqa: E402

# Import the production helpers we'll use to build expected values - never
# re-encode constants.
from custom_components.rain_incoming.radar.geo import lat_lon_to_tile
from custom_components.rain_incoming.const import (
    RAINVIEWER_ZOOM,
    RAINVIEWER_ANALYSIS_GRID,
)


# ---------------------------------------------------------------------------
# _tile_coords
# ---------------------------------------------------------------------------


class TestTileCoordsCount:
    """The function must return exactly (2*GRID_HALF+1)^2 tiles."""

    def test_tile_coords_returns_correct_count(self) -> None:
        lat, lon = -33.8688, 151.2093  # Sydney
        coords = _tile_coords(lat, lon)
        side = 2 * GRID_HALF + 1
        assert len(coords) == side * side


class TestTileCoordsCentered:
    """The centre tile must match what lat_lon_to_tile returns for the location."""

    def test_tile_coords_centered_on_location(self) -> None:
        lat, lon = -37.8136, 144.9631  # Melbourne
        coords = _tile_coords(lat, lon)
        expected_center = lat_lon_to_tile(lat, lon, ZOOM)
        # _tile_coords iterates dy then dx; the centre element is at index
        # GRID_HALF * (2*GRID_HALF+1) + GRID_HALF (the middle row, middle col).
        side = 2 * GRID_HALF + 1
        center_index = GRID_HALF * side + GRID_HALF
        assert coords[center_index] == expected_center


class TestTileCoordsSydney:
    """Known coordinates must produce a deterministic tile list."""

    # Sydney CBD at zoom 7 -> tile (116, 77).  Derived from the production
    # lat_lon_to_tile function - not hardcoded from memory.
    LAT = -33.8688
    LON = 151.2093

    def test_tile_coords_sydney(self) -> None:
        coords = _tile_coords(self.LAT, self.LON)
        cx, cy = lat_lon_to_tile(self.LAT, self.LON, ZOOM)

        # Build the expected grid in the same order the function does
        # (dy outer, dx inner).
        expected = []
        for dy in range(-GRID_HALF, GRID_HALF + 1):
            for dx in range(-GRID_HALF, GRID_HALF + 1):
                expected.append((cx + dx, cy + dy))

        assert coords == expected

    def test_tile_coords_sydney_center_tile_x_in_range(self) -> None:
        """Sanity: center tile x at zoom 7 must be in [0, 127]."""
        cx, _cy = lat_lon_to_tile(self.LAT, self.LON, ZOOM)
        assert 0 <= cx < 2**ZOOM


class TestTileCoordsDifferentLocations:
    """Different lat/lon must produce different tile lists."""

    def test_tile_coords_different_locations(self) -> None:
        sydney_coords = _tile_coords(-33.8688, 151.2093)
        melbourne_coords = _tile_coords(-37.8136, 144.9631)
        assert sydney_coords != melbourne_coords

    def test_tile_coords_antipodal_locations_differ(self) -> None:
        """Even locations that happen to share one axis still differ overall."""
        # Same latitude, very different longitude.
        east_coords = _tile_coords(-33.8688, 151.2093)   # Sydney
        west_coords = _tile_coords(-33.8688, -70.6693)   # Santiago (same lat)
        assert east_coords != west_coords


# ---------------------------------------------------------------------------
# main() / argparse
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    """main() must require lat, lon, and name as positional arguments."""

    def test_main_requires_lat_lon_name(self) -> None:
        """Calling main() with no args must raise SystemExit (argparse error)."""
        import pytest
        from unittest.mock import patch

        # Patch sys.argv to simulate running with no arguments.
        with patch("sys.argv", ["capture_golden_tiles.py"]):
            from scripts.capture_golden_tiles import main
            with pytest.raises(SystemExit) as exc_info:
                main()
        # argparse exits with code 2 on missing required arguments.
        assert exc_info.value.code == 2

    def test_main_requires_all_three_args(self) -> None:
        """Providing only lat and lon (missing name) must also fail."""
        import pytest
        from unittest.mock import patch

        with patch("sys.argv", ["capture_golden_tiles.py", "-33.8688", "151.2093"]):
            from scripts.capture_golden_tiles import main
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2
