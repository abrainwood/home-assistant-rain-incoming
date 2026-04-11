"""Unit tests for LastRainNearbySensor state restore path.

TDD: these tests are written to verify the restore guard is robust to legacy
state values. In particular, old versions persisted the literal string "None"
as a sensor state. The guard must not silently skip these - it should attempt
datetime.fromisoformat("None"), catch the ValueError, log a WARNING, and leave
coordinator.last_rain_nearby_time unchanged.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.rain_incoming.sensor import LastRainNearbySensor


def _make_coordinator():
    coordinator = MagicMock()
    coordinator.last_rain_nearby_time = None
    return coordinator


def _make_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry_123"
    entry.data = {}
    return entry


def _make_sensor(coordinator, entry):
    """Build a LastRainNearbySensor with minimal mocks, bypassing CoordinatorEntity init."""
    sensor = LastRainNearbySensor.__new__(LastRainNearbySensor)
    sensor.coordinator = coordinator
    sensor._entry = entry
    sensor._attr_unique_id = f"{entry.entry_id}_last_rain_nearby"
    return sensor


class TestLastRainNearbyRestore:
    """Restore path for LastRainNearbySensor."""

    @pytest.mark.asyncio
    async def test_legacy_none_string_logs_warning_and_leaves_time_unchanged(self, caplog):
        """Restoring state "None" (legacy string) must log a WARNING and not update coordinator."""
        coordinator = _make_coordinator()
        entry = _make_entry()
        sensor = _make_sensor(coordinator, entry)

        last_state = MagicMock()
        last_state.state = "None"

        with (
            patch.object(
                LastRainNearbySensor,
                "async_get_last_state",
                new=AsyncMock(return_value=last_state),
            ),
        ):
            with patch(
                "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
                new=AsyncMock(),
            ), patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass",
                new=AsyncMock(),
            ):
                with caplog.at_level(logging.WARNING, logger="custom_components.rain_incoming.sensor"):
                    await LastRainNearbySensor.async_added_to_hass(sensor)

        # The guard must NOT suppress the parse - it should hit ValueError and warn
        assert coordinator.last_rain_nearby_time is None, (
            "coordinator.last_rain_nearby_time should remain None when "
            'restoring the legacy "None" string'
        )
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("None" in msg for msg in warning_messages), (
            f'Expected a WARNING mentioning "None" but got: {warning_messages}'
        )

    @pytest.mark.asyncio
    async def test_valid_iso_timestamp_restores_coordinator_time(self):
        """Restoring a valid ISO 8601 timestamp updates coordinator.last_rain_nearby_time."""
        from datetime import datetime, timezone

        coordinator = _make_coordinator()
        entry = _make_entry()
        sensor = _make_sensor(coordinator, entry)

        iso_ts = "2026-04-07T10:30:00+00:00"
        last_state = MagicMock()
        last_state.state = iso_ts

        with (
            patch.object(
                LastRainNearbySensor,
                "async_get_last_state",
                new=AsyncMock(return_value=last_state),
            ),
        ):
            with patch(
                "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
                new=AsyncMock(),
            ), patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass",
                new=AsyncMock(),
            ):
                await LastRainNearbySensor.async_added_to_hass(sensor)

        expected = datetime.fromisoformat(iso_ts)
        assert coordinator.last_rain_nearby_time == expected

    @pytest.mark.asyncio
    async def test_unknown_state_skips_restore(self, caplog):
        """State 'unknown' must not trigger any datetime parse attempt."""
        coordinator = _make_coordinator()
        entry = _make_entry()
        sensor = _make_sensor(coordinator, entry)

        last_state = MagicMock()
        last_state.state = "unknown"

        with (
            patch.object(
                LastRainNearbySensor,
                "async_get_last_state",
                new=AsyncMock(return_value=last_state),
            ),
        ):
            with patch(
                "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
                new=AsyncMock(),
            ), patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass",
                new=AsyncMock(),
            ):
                with caplog.at_level(logging.WARNING, logger="custom_components.rain_incoming.sensor"):
                    await LastRainNearbySensor.async_added_to_hass(sensor)

        assert coordinator.last_rain_nearby_time is None
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages == [], f"Expected no warnings, got: {warning_messages}"

    @pytest.mark.asyncio
    async def test_unavailable_state_skips_restore(self, caplog):
        """State 'unavailable' must not trigger any datetime parse attempt."""
        coordinator = _make_coordinator()
        entry = _make_entry()
        sensor = _make_sensor(coordinator, entry)

        last_state = MagicMock()
        last_state.state = "unavailable"

        with (
            patch.object(
                LastRainNearbySensor,
                "async_get_last_state",
                new=AsyncMock(return_value=last_state),
            ),
        ):
            with patch(
                "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
                new=AsyncMock(),
            ), patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass",
                new=AsyncMock(),
            ):
                with caplog.at_level(logging.WARNING, logger="custom_components.rain_incoming.sensor"):
                    await LastRainNearbySensor.async_added_to_hass(sensor)

        assert coordinator.last_rain_nearby_time is None
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages == [], f"Expected no warnings, got: {warning_messages}"

    @pytest.mark.asyncio
    async def test_no_last_state_skips_restore(self):
        """No persisted state (first boot) must leave coordinator.last_rain_nearby_time as None."""
        coordinator = _make_coordinator()
        entry = _make_entry()
        sensor = _make_sensor(coordinator, entry)

        with (
            patch.object(
                LastRainNearbySensor,
                "async_get_last_state",
                new=AsyncMock(return_value=None),
            ),
        ):
            with patch(
                "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
                new=AsyncMock(),
            ), patch(
                "homeassistant.helpers.update_coordinator.CoordinatorEntity.async_added_to_hass",
                new=AsyncMock(),
            ):
                await LastRainNearbySensor.async_added_to_hass(sensor)

        assert coordinator.last_rain_nearby_time is None
