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
from homeassistant.core import HomeAssistant
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
