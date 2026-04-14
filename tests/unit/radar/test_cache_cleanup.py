"""Tests for tile cache cleanup and bounds.

C5: Module-level tile caches in composite.py must be cleared when all
entries are unloaded, and the map cache must have a size limit.
"""
from __future__ import annotations

from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    _map_tile_cache,
    _radar_tile_cache,
    _MAP_CACHE_MAX,
    clear_tile_caches,
)


class TestClearTileCaches:
    """clear_tile_caches must empty both module-level caches."""

    def test_clears_both_caches(self):
        # Populate caches with dummy data
        _map_tile_cache[("test", 7, 10, 20)] = Image.new("RGBA", (1, 1))
        _radar_tile_cache[("/v2/radar/x", 7, 10, 20, 2)] = Image.new("RGBA", (1, 1))

        assert len(_map_tile_cache) > 0
        assert len(_radar_tile_cache) > 0

        clear_tile_caches()

        assert len(_map_tile_cache) == 0, "Map cache must be empty after clear"
        assert len(_radar_tile_cache) == 0, "Radar cache must be empty after clear"


class TestMapCacheBounded:
    """Map tile cache must have a size limit to prevent unbounded memory growth."""

    def test_map_cache_max_is_defined(self):
        assert _MAP_CACHE_MAX > 0, "Map cache must have a positive size limit"

    def test_map_cache_max_is_reasonable(self):
        # Each RGBA 256x256 tile is ~256KB. At 200 entries that's ~50MB.
        assert _MAP_CACHE_MAX <= 500, (
            f"Map cache limit {_MAP_CACHE_MAX} is too large - each tile is ~256KB"
        )
