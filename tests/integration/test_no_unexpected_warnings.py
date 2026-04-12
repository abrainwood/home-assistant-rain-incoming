"""Integration test: no unexpected warnings from rain_incoming during normal lifecycle.

This replaces the E2E test TestSystemLogs::test_no_unexpected_warnings_from_integration,
which did the same thing by scraping the HA Docker error log after a full stack run.
Using caplog here gives us the same signal in ~1 second without Docker.

Filter logic mirrors the E2E version exactly:
- Only look at loggers under custom_components.rain_incoming
- WARNING or above
- Exclude the one known-unavoidable HA core warning about untested custom integrations
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState, HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.rain_incoming.const import DOMAIN
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult

MOCK_RESULT_NO_RAIN = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.NORMAL,
    frame_count=6,
    max_approaching_intensity=0.0,
)

# Patterns that match known/expected warnings we cannot control (same list as E2E).
_EXPECTED_PATTERNS: list[str] = []

_OUR_LOGGER_PREFIX = "custom_components.rain_incoming"


def _unexpected_warnings(caplog_records):
    """Return records that are WARNING+ from our integration and not in the expected list."""
    our_records = [
        r for r in caplog_records
        if r.name.startswith(_OUR_LOGGER_PREFIX)
        and r.levelno >= logging.WARNING
    ]
    return [
        r for r in our_records
        if not any(pattern in r.getMessage() for pattern in _EXPECTED_PATTERNS)
    ]


@pytest.mark.asyncio
async def test_no_unexpected_warnings_from_integration(hass: HomeAssistant, caplog):
    """Integration should not emit any WARNING or ERROR logs during a normal lifecycle.

    Lifecycle: setup -> one coordinator update cycle -> unload.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
        entry_id="test_warn_check",
    )
    entry.add_to_hass(hass)

    with caplog.at_level(logging.WARNING, logger=_OUR_LOGGER_PREFIX):
        with patch(
            "custom_components.rain_incoming.coordinator.RainDetectorCoordinator._async_update_data",
            new=AsyncMock(return_value=MOCK_RESULT_NO_RAIN),
        ):
            # Setup
            await hass.config_entries.async_setup(entry.entry_id)
            await hass.async_block_till_done()

            # One extra update cycle (simulates coordinator ticking)
            coordinator = hass.data[DOMAIN][entry.entry_id]
            await coordinator.async_refresh()
            await hass.async_block_till_done()

            # Unload
            await hass.config_entries.async_unload(entry.entry_id)
            await hass.async_block_till_done()

    unexpected = _unexpected_warnings(caplog.records)

    assert not unexpected, (
        f"Found {len(unexpected)} unexpected warning(s) from rain_incoming:\n"
        + "\n".join(f"  [{r.levelname}] {r.name}: {r.getMessage()}" for r in unexpected)
    )


@pytest.mark.asyncio
async def test_setup_unload_does_not_emit_value_error(hass: HomeAssistant, caplog):
    """Setting up then unloading the entry should not log ValueError from listener cleanup.

    Regression test for the async_listen_once double-removal bug: when the entry is
    set up before HA has fully started, a once-listener is registered for
    EVENT_HOMEASSISTANT_STARTED. The listener auto-removes itself when it fires.
    When the entry is later unloaded, async_on_unload calls the cleanup callable a
    second time - against an already-removed listener - raising ValueError.

    This test exercises that path by temporarily making hass.is_running return False
    during async_setup_entry, so the pre-startup branch is taken.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
        entry_id="test_value_error_check",
    )
    entry.add_to_hass(hass)

    # Capture errors from homeassistant.core where the ValueError would surface.
    with caplog.at_level(logging.ERROR, logger="homeassistant"):
        with patch(
            "custom_components.rain_incoming.coordinator.RainDetectorCoordinator._async_update_data",
            new=AsyncMock(return_value=MOCK_RESULT_NO_RAIN),
        ):
            # Force hass.is_running to False during setup so the once-listener branch fires.
            original_state = hass.state
            hass.set_state(CoreState.not_running)
            try:
                await hass.config_entries.async_setup(entry.entry_id)
            finally:
                hass.set_state(original_state)

            # Fire EVENT_HOMEASSISTANT_STARTED so the once-listener fires and auto-removes.
            hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
            await hass.async_block_till_done()

            # Unload: on_unload callbacks fire, including the now-stale listener cleanup.
            # With the bug, this raises ValueError: list.remove(x): x not in list.
            await hass.config_entries.async_unload(entry.entry_id)
            await hass.async_block_till_done()

    value_errors = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and (
            "list.remove" in r.getMessage() or "unknown job listener" in r.getMessage()
        )
    ]
    assert not value_errors, (
        "Expected no ValueError from listener cleanup on unload, got: "
        + str([r.getMessage() for r in value_errors])
    )
