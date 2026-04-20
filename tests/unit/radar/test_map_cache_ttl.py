"""Tests for map tile cache TTL expiry (#87).

Map tiles should expire after a configurable TTL so stale tiles from
provider updates are eventually refreshed.
"""
from __future__ import annotations

import time
from unittest.mock import patch

from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    _map_tile_cache,
    _MAP_CACHE_TTL_SECONDS,
)


class TestMapCacheTTL:
    def test_ttl_constant_is_24_hours(self):
        assert _MAP_CACHE_TTL_SECONDS == 86400

    def test_expired_entry_treated_as_cache_miss(self):
        """A map tile older than TTL must not be returned from cache."""
        _map_tile_cache.clear()
        try:
            key = ("voyager", 7, 100, 50)
            tile = Image.new("RGBA", (1, 1))
            # Insert with a timestamp in the past (beyond TTL)
            expired_time = time.monotonic() - _MAP_CACHE_TTL_SECONDS - 1
            _map_tile_cache[key] = (tile, expired_time)

            from custom_components.rain_incoming.radar.composite import _get_cached_map_tile
            result = _get_cached_map_tile(key)
            assert result is None, "Expired tile must not be returned from cache"
        finally:
            _map_tile_cache.clear()

    def test_fresh_entry_returned_from_cache(self):
        """A map tile within TTL must be returned from cache."""
        _map_tile_cache.clear()
        try:
            key = ("voyager", 7, 100, 50)
            tile = Image.new("RGBA", (1, 1))
            fresh_time = time.monotonic()
            _map_tile_cache[key] = (tile, fresh_time)

            from custom_components.rain_incoming.radar.composite import _get_cached_map_tile
            result = _get_cached_map_tile(key)
            assert result is tile, "Fresh tile must be returned from cache"
        finally:
            _map_tile_cache.clear()
