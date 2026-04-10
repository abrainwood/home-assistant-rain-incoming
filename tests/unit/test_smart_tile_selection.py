"""Tests for smart 2x2 tile selection and lookahead limit changes.

TDD: these tests are written FIRST, before the implementation.
They should FAIL until the production code is updated.
"""
from __future__ import annotations

import math

import pytest

from custom_components.rain_incoming.radar.geo import lat_lon_to_tile


class TestSmartTileSelection:
    """Smart 2x2 tile selection picks the 4 tiles surrounding the user's location.

    Instead of a fixed NxN grid centred on one tile, we pick the 2x2 block
    where the location falls near the centre. This guarantees ~130km+ radius
    in all directions at zoom 7, using only 4 tiles instead of 25.
    """

    def test_build_analysis_tiles_returns_four_tiles(self):
        """Smart selection always returns exactly 4 tiles."""
        from custom_components.rain_incoming.coordinator import _build_analysis_tiles

        tiles = _build_analysis_tiles(-33.701, 151.209, zoom=7)
        assert len(tiles) == 4, f"Expected 4 tiles, got {len(tiles)}"

    def test_location_is_within_tile_coverage(self):
        """The user's location must fall inside the area covered by the 4 tiles."""
        from custom_components.rain_incoming.coordinator import _build_analysis_tiles
        from custom_components.rain_incoming.providers.rainviewer import _tile_bounds

        lat, lon = -33.701, 151.209
        tiles = _build_analysis_tiles(lat, lon, zoom=7)

        # The 4 tiles form a 2x2 block. The location must be inside.
        all_bounds = [_tile_bounds(tx, ty, 7) for tx, ty in tiles]
        min_lat = min(b.lat_min for b in all_bounds)
        max_lat = max(b.lat_max for b in all_bounds)
        min_lon = min(b.lon_min for b in all_bounds)
        max_lon = max(b.lon_max for b in all_bounds)

        assert min_lat < lat < max_lat, f"Location lat {lat} outside tile coverage [{min_lat}, {max_lat}]"
        assert min_lon < lon < max_lon, f"Location lon {lon} outside tile coverage [{min_lon}, {max_lon}]"

    def test_location_not_at_edge_of_coverage(self):
        """The location should be roughly centred, not at a tile edge.

        With smart 2x2 selection, the worst case is the location is at the
        centre of the 2x2 block (best case) or near a corner (still has
        ~1 tile of coverage in every direction). We verify there's at least
        100km of coverage in every direction.
        """
        from custom_components.rain_incoming.coordinator import _build_analysis_tiles
        from custom_components.rain_incoming.providers.rainviewer import _tile_bounds

        lat, lon = -33.701, 151.209
        tiles = _build_analysis_tiles(lat, lon, zoom=7)

        all_bounds = [_tile_bounds(tx, ty, 7) for tx, ty in tiles]
        min_lat = min(b.lat_min for b in all_bounds)
        max_lat = max(b.lat_max for b in all_bounds)
        min_lon = min(b.lon_min for b in all_bounds)
        max_lon = max(b.lon_max for b in all_bounds)

        # Approximate km conversion at Sydney latitude
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.32 * math.cos(math.radians(lat))

        km_north = (max_lat - lat) * km_per_deg_lat
        km_south = (lat - min_lat) * km_per_deg_lat
        km_east = (max_lon - lon) * km_per_deg_lon
        km_west = (lon - min_lon) * km_per_deg_lon

        min_radius = 100  # km
        assert km_north > min_radius, f"Only {km_north:.0f}km north coverage (need {min_radius}km)"
        assert km_south > min_radius, f"Only {km_south:.0f}km south coverage (need {min_radius}km)"
        assert km_east > min_radius, f"Only {km_east:.0f}km east coverage (need {min_radius}km)"
        assert km_west > min_radius, f"Only {km_west:.0f}km west coverage (need {min_radius}km)"

    def test_tiles_form_contiguous_2x2_block(self):
        """The 4 tiles must be adjacent - a 2x2 block, not scattered."""
        from custom_components.rain_incoming.coordinator import _build_analysis_tiles

        tiles = _build_analysis_tiles(-33.701, 151.209, zoom=7)
        xs = sorted(set(t[0] for t in tiles))
        ys = sorted(set(t[1] for t in tiles))

        assert len(xs) == 2, f"Expected 2 unique x values, got {xs}"
        assert len(ys) == 2, f"Expected 2 unique y values, got {ys}"
        assert xs[1] - xs[0] == 1, f"X tiles not adjacent: {xs}"
        assert ys[1] - ys[0] == 1, f"Y tiles not adjacent: {ys}"

    def test_works_at_various_locations(self):
        """Smart selection works at different latitudes (equator, poles, dateline)."""
        from custom_components.rain_incoming.coordinator import _build_analysis_tiles

        locations = [
            (-33.701, 151.209, "Sydney"),
            (-12.463, 130.845, "Darwin"),
            (51.507, -0.128, "London"),
            (0.0, 0.0, "Null Island"),
            (-41.886, 145.360, "Lake Margaret Tasmania"),
            (-17.350, 145.950, "Babinda QLD"),
        ]
        for lat, lon, name in locations:
            tiles = _build_analysis_tiles(lat, lon, zoom=7)
            assert len(tiles) == 4, f"{name}: expected 4 tiles, got {len(tiles)}"


class TestLookaheadLimits:
    """Lookahead should be 20-60 minutes, not 15-120."""

    def test_max_lookahead_is_60(self):
        from custom_components.rain_incoming.const import MAX_LOOKAHEAD_MINUTES
        assert MAX_LOOKAHEAD_MINUTES == 60

    def test_min_lookahead_is_20(self):
        from custom_components.rain_incoming.const import MIN_LOOKAHEAD_MINUTES
        assert MIN_LOOKAHEAD_MINUTES == 20

    def test_config_flow_rejects_over_60(self):
        """Config flow should reject lookahead > 60."""
        from custom_components.rain_incoming.config_flow import _validate_input
        errors = _validate_input({"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 90})
        assert "lookahead_minutes" in errors

    def test_config_flow_rejects_under_20(self):
        """Config flow should reject lookahead < 20."""
        from custom_components.rain_incoming.config_flow import _validate_input
        errors = _validate_input({"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 10})
        assert "lookahead_minutes" in errors

    def test_config_flow_accepts_60(self):
        """Config flow should accept lookahead = 60."""
        from custom_components.rain_incoming.config_flow import _validate_input
        errors = _validate_input({"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 60})
        assert "lookahead_minutes" not in errors

    def test_config_flow_accepts_20(self):
        """Config flow should accept lookahead = 20."""
        from custom_components.rain_incoming.config_flow import _validate_input
        errors = _validate_input({"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 20})
        assert "lookahead_minutes" not in errors
