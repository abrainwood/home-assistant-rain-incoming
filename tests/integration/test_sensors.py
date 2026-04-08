from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.incoming_rain.const import DOMAIN
from custom_components.incoming_rain.radar.detector import Confidence, DetectionResult

MOCK_RESULT_UNAVAILABLE = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
)

MOCK_RESULT_RAIN_COMING = DetectionResult(
    rain_incoming=True,
    arrival_time=datetime(2026, 4, 7, 10, 30, tzinfo=timezone.utc),
    confidence=Confidence.NORMAL,
    frame_count=6,
)

MOCK_RESULT_NO_RAIN = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.NORMAL,
    frame_count=6,
)


@pytest.fixture
def mock_entry():
    return MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
        entry_id="test_abc123",
    )


async def _setup_integration(hass: HomeAssistant, entry: MockConfigEntry, result: DetectionResult):
    entry.add_to_hass(hass)
    with (
        patch(
            "custom_components.incoming_rain.coordinator.RainDetectorCoordinator._async_update_data",
            new=AsyncMock(return_value=result),
        ),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_binary_sensor_unavailable_when_no_data(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_UNAVAILABLE)
    state = hass.states.get("binary_sensor.incoming_rain_status")
    assert state is not None
    assert state.state == "unavailable"


@pytest.mark.asyncio
async def test_binary_sensor_on_when_rain_coming(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("binary_sensor.incoming_rain_status")
    assert state is not None
    assert state.state == "on"
    assert state.attributes["confidence"] == "normal"


@pytest.mark.asyncio
async def test_binary_sensor_off_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("binary_sensor.incoming_rain_status")
    assert state is not None
    assert state.state == "off"


@pytest.mark.asyncio
async def test_arrival_sensor_has_timestamp_when_rain_coming(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.incoming_rain_arrival_time")
    assert state is not None
    assert state.state != "unknown"
    assert state.state != "unavailable"


@pytest.mark.asyncio
async def test_arrival_sensor_unknown_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.incoming_rain_arrival_time")
    assert state is not None
    assert state.state in ("unknown", "None")


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_id", [
    "image.incoming_rain_radar_64km",
    "image.incoming_rain_radar_128km",
    "image.incoming_rain_radar_256km",
])
async def test_image_entity_created(hass: HomeAssistant, mock_entry, entity_id):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get(entity_id)
    assert state is not None, f"Entity {entity_id} not found"
