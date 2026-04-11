"""Tests for stale image handling.

TDD: these tests are written FIRST, before the implementation.

When a coordinator poll fails (rate limited, network error), the image
entity should keep showing the previous render rather than going blank.
A stale_since attribute should indicate when data became stale.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from custom_components.rain_incoming.image import RadarImageEntity


def _make_coordinator(
    latest_frame_path: str | None = "/v2/radar/latest",
    frame_paths: list[str] | None = None,
    last_update_success: bool = True,
    last_update_success_time: datetime | None = None,
):
    """Create a mock coordinator."""
    coord = MagicMock()
    coord.latest_frame_path = latest_frame_path
    coord.frame_paths = frame_paths or ["/v2/radar/f1", "/v2/radar/f2"]
    coord.frame_timestamps = [
        datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 10, 10, 10, tzinfo=timezone.utc),
    ]
    coord.confidence_maps = None
    coord.tracked_cells = []
    coord.last_update_success = last_update_success
    coord.last_update_success_time = last_update_success_time or datetime.now(timezone.utc)
    coord.data = MagicMock()
    return coord


def _make_entry():
    entry = MagicMock()
    entry.data = {"latitude": -33.7, "longitude": 151.2}
    entry.entry_id = "test_entry"
    return entry


class TestStaleImageHandling:
    """Image entity returns cached image when coordinator has no fresh data."""

    @pytest.mark.asyncio
    async def test_returns_cached_image_when_coordinator_has_no_new_data(self):
        """After a successful render, if coordinator.latest_frame_path becomes
        None (failed fetch), async_image should return the cached image."""
        coord = _make_coordinator(latest_frame_path=None)
        entry = _make_entry()
        entity = RadarImageEntity(coord, entry, 128)

        # Simulate a previous successful render
        entity._cached_image = b"GIF89a_previous_render"
        entity._cached_frame_path = "/v2/radar/old"

        # Coordinator has no new data (failed fetch)
        result = await entity.async_image()
        assert result == b"GIF89a_previous_render", (
            "Should return cached image when coordinator has no fresh data"
        )

    @pytest.mark.asyncio
    async def test_returns_placeholder_when_no_cache_and_no_data(self):
        """On first load with no data, return a placeholder image (not None/broken)."""
        coord = _make_coordinator(latest_frame_path=None)
        entry = _make_entry()
        entity = RadarImageEntity(coord, entry, 128)

        result = await entity.async_image()
        assert result is not None, "Should return placeholder, not None (causes broken image icon)"
        assert len(result) > 100, "Placeholder should be a valid image"

    def test_available_true_when_stale_but_cached(self):
        """Entity should remain available when showing stale cached data."""
        coord = _make_coordinator(latest_frame_path=None, last_update_success=False)
        entry = _make_entry()
        entity = RadarImageEntity(coord, entry, 128)
        entity._cached_image = b"GIF89a_previous_render"

        assert entity.available is True, (
            "Entity should be available when it has cached data to show"
        )

    def test_extra_state_attributes_includes_stale_since(self):
        """When showing stale data, extra_state_attributes should include stale_since."""
        last_success = datetime(2026, 4, 10, 10, 1, 0, tzinfo=timezone.utc)
        coord = _make_coordinator(
            latest_frame_path=None,
            last_update_success=False,
            last_update_success_time=last_success,
        )
        entry = _make_entry()
        entity = RadarImageEntity(coord, entry, 128)
        entity._cached_image = b"GIF89a_previous_render"

        attrs = entity.extra_state_attributes
        assert "stale_since" in attrs, "Should include stale_since when showing cached data"
        assert attrs["stale_since"] == last_success.isoformat()

    def test_no_stale_since_when_fresh(self):
        """When data is fresh, no stale_since attribute."""
        coord = _make_coordinator(last_update_success=True)
        entry = _make_entry()
        entity = RadarImageEntity(coord, entry, 128)

        attrs = entity.extra_state_attributes
        assert "stale_since" not in attrs
