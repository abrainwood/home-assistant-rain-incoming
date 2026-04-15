"""Tests for LRU cache eviction (#108).

Tile caches must evict least-recently-used entries, not FIFO.
A recently accessed tile should survive eviction even if it was
inserted early.
"""
from __future__ import annotations

from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    _map_tile_cache,
    _radar_tile_cache,
    _MAP_CACHE_MAX,
    _RADAR_CACHE_MAX,
    _evict_map_cache_if_full,
    _evict_radar_cache_if_full,
)


class TestMapCacheLRU:
    def test_recently_accessed_tile_survives_eviction(self):
        """A tile accessed recently must not be evicted even if inserted first."""
        _map_tile_cache.clear()
        try:
            tiny = Image.new("RGBA", (1, 1))
            # Insert MAX + 10 entries
            for i in range(_MAP_CACHE_MAX + 10):
                _map_tile_cache[("style", 7, i, 0)] = tiny

            # Access the very first entry via the LRU touch pattern used in production:
            # .get() followed by .move_to_end()
            first_key = ("style", 7, 0, 0)
            _ = _map_tile_cache.get(first_key)
            _map_tile_cache.move_to_end(first_key)

            _evict_map_cache_if_full()

            assert first_key in _map_tile_cache, (
                "Recently accessed tile was evicted - cache is FIFO, not LRU"
            )
        finally:
            _map_tile_cache.clear()


class TestRadarCacheLRU:
    def test_recently_accessed_tile_survives_eviction(self):
        """A radar tile accessed recently must not be evicted."""
        _radar_tile_cache.clear()
        try:
            tiny = Image.new("RGBA", (1, 1))
            for i in range(_RADAR_CACHE_MAX + 10):
                _radar_tile_cache[(f"/v2/radar/{i}", 7, 0, 0, 2)] = tiny

            first_key = ("/v2/radar/0", 7, 0, 0, 2)
            _ = _radar_tile_cache.get(first_key)
            _radar_tile_cache.move_to_end(first_key)

            _evict_radar_cache_if_full()

            assert first_key in _radar_tile_cache, (
                "Recently accessed radar tile was evicted - cache is FIFO, not LRU"
            )
        finally:
            _radar_tile_cache.clear()
