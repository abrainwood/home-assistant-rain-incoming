"""Tests for tile cache cleanup and bounds.

C5: Module-level tile caches in composite.py must be cleared when all
entries are unloaded, and the map cache must have a size limit.
"""
from __future__ import annotations

import pytest
from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    _map_tile_cache,
    _radar_tile_cache,
    _tile_semaphore,
    _MAP_CACHE_MAX,
    clear_tile_caches,
)


class TestClearTileCaches:
    """clear_tile_caches must empty both module-level caches."""

    def test_clears_both_caches(self):
        import time
        # Populate caches with dummy data
        _map_tile_cache[("test", 7, 10, 20)] = (Image.new("RGBA", (1, 1)), time.monotonic())
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

    def test_map_cache_evicts_when_full(self):
        """Filling the map cache past _MAP_CACHE_MAX must trigger eviction."""
        import time
        from custom_components.rain_incoming.radar.composite import _evict_map_cache_if_full

        _map_tile_cache.clear()
        try:
            tiny = Image.new("RGBA", (1, 1))
            now = time.monotonic()
            for i in range(_MAP_CACHE_MAX + 10):
                _map_tile_cache[("style", 7, i, 0)] = (tiny, now)

            assert len(_map_tile_cache) > _MAP_CACHE_MAX

            _evict_map_cache_if_full()

            assert len(_map_tile_cache) <= _MAP_CACHE_MAX, (
                f"Map cache has {len(_map_tile_cache)} entries after eviction, "
                f"expected <= {_MAP_CACHE_MAX}"
            )
        finally:
            _map_tile_cache.clear()


class TestUnloadClearsCaches:
    """async_unload_entry must clear tile caches when the last entry is removed."""

    @pytest.mark.asyncio
    async def test_unload_last_entry_clears_caches(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from custom_components.rain_incoming import async_unload_entry
        from custom_components.rain_incoming.const import DOMAIN

        import time
        # Populate caches
        _map_tile_cache[("test", 7, 1, 1)] = (Image.new("RGBA", (1, 1)), time.monotonic())
        _radar_tile_cache[("/v2/radar/x", 7, 1, 1, 2)] = Image.new("RGBA", (1, 1))

        hass = MagicMock()
        entry = MagicMock()
        entry.entry_id = "last_entry"

        coordinator = AsyncMock()
        coordinator.async_save_clutter_map = AsyncMock()

        # Simulate last entry: after pop, the domain dict is empty
        hass.data = {DOMAIN: {entry.entry_id: coordinator}}
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

        await async_unload_entry(hass, entry)

        assert len(_map_tile_cache) == 0, "Map cache must be cleared after last entry unload"
        assert len(_radar_tile_cache) == 0, "Radar cache must be cleared after last entry unload"


class TestAsyncPrimitivesResetOnClear:
    """clear_tile_caches must reset the tile semaphore and render lock
    so they're recreated fresh after integration reload (#122)."""

    def test_tile_semaphore_reset_on_clear(self):
        import custom_components.rain_incoming.radar.composite as comp

        # Force creation
        comp._get_tile_semaphore()
        assert comp._tile_semaphore is not None

        clear_tile_caches()

        assert comp._tile_semaphore is None, (
            "Tile semaphore must be reset to None after clear_tile_caches"
        )

    def test_render_lock_reset_on_clear(self):
        import custom_components.rain_incoming.image as img_mod

        # Force creation for two entries
        img_mod._get_render_lock("entry_a")
        img_mod._get_render_lock("entry_b")
        assert len(img_mod._render_locks) == 2

        # Reset specific entry
        from custom_components.rain_incoming.image import reset_render_lock
        reset_render_lock("entry_a")
        assert "entry_a" not in img_mod._render_locks
        assert "entry_b" in img_mod._render_locks

        # Reset all (last entry unload)
        reset_render_lock()
        assert len(img_mod._render_locks) == 0, (
            "Render locks must be cleared after reset_render_lock()"
        )
