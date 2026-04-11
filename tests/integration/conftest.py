"""
Integration test conftest.

Prevents the coordinator from making real HTTP calls to RainViewer during tests
that exercise the config flow or other HA setup paths that auto-start the coordinator.
The sensor tests mock _async_update_data explicitly; this fixture covers the rest.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult

_MOCK_RESULT_UNAVAILABLE = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
    max_approaching_intensity=0.0,
)


async def setup_integration(hass: HomeAssistant, entry: MockConfigEntry, result: DetectionResult):
    entry.add_to_hass(hass)
    with patch(
        "custom_components.rain_incoming.coordinator.RainDetectorCoordinator._async_update_data",
        new=AsyncMock(return_value=result),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()


@pytest.fixture(autouse=True)
def mock_coordinator_network(request):
    """
    Patch coordinator network calls for integration tests that set up real HA entries.

    The coordinator tests call _async_update_data directly with their own patches,
    so we skip this fixture for that module to avoid double-patching.
    """
    if "test_coordinator" in request.module.__name__:
        yield
        return

    with patch(
        "custom_components.rain_incoming.coordinator.RainDetectorCoordinator._async_update_data",
        new=AsyncMock(return_value=_MOCK_RESULT_UNAVAILABLE),
    ):
        yield
