from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.rain_incoming.const import CONF_LOCATION_NAME, DOMAIN
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult

COMPONENT_DIR = pathlib.Path(__file__).resolve().parents[2] / "custom_components" / "rain_incoming"

MOCK_RESULT_UNAVAILABLE = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
    max_approaching_intensity=0.0,
)

MOCK_RESULT_RAIN_COMING = DetectionResult(
    rain_incoming=True,
    arrival_time=datetime(2026, 4, 7, 10, 30, tzinfo=timezone.utc),
    confidence=Confidence.NORMAL,
    frame_count=6,
    max_approaching_intensity=0.65,
)

MOCK_RESULT_NO_RAIN = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.NORMAL,
    frame_count=6,
    max_approaching_intensity=0.0,
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
            "custom_components.rain_incoming.coordinator.RainDetectorCoordinator._async_update_data",
            new=AsyncMock(return_value=result),
        ),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_binary_sensor_unavailable_when_no_data(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_UNAVAILABLE)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state is not None
    assert state.state == "unavailable"


@pytest.mark.asyncio
async def test_binary_sensor_on_when_rain_coming(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state is not None
    assert state.state == "on"
    assert state.attributes["confidence"] == "normal"


@pytest.mark.asyncio
async def test_binary_sensor_off_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state is not None
    assert state.state == "off"


@pytest.mark.asyncio
async def test_arrival_sensor_has_timestamp_when_rain_coming(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_arrival_time")
    assert state is not None
    assert state.state != "unknown"
    assert state.state != "unavailable"


@pytest.mark.asyncio
async def test_arrival_sensor_unknown_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_arrival_time")
    assert state is not None
    assert state.state in ("unknown", "None")


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_id", [
    "image.rain_incoming_radar_64km",
    "image.rain_incoming_radar_128km",
    "image.rain_incoming_radar_256km",
])
async def test_image_entity_created(hass: HomeAssistant, mock_entry, entity_id):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get(entity_id)
    assert state is not None, f"Entity {entity_id} not found"


@pytest.mark.asyncio
async def test_image_entity_content_type_is_gif(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    entity = hass.data[DOMAIN][mock_entry.entry_id]
    # Get the image entity from the entity registry
    from homeassistant.helpers.entity_component import EntityComponent
    from homeassistant.components.image import DOMAIN as IMAGE_DOMAIN
    component: EntityComponent = hass.data.get("entity_components", {}).get(IMAGE_DOMAIN)
    if component is not None:
        entities = [e for e in component.entities if "radar" in e.entity_id]
        for entity in entities:
            assert entity.content_type == "image/gif"
    else:
        # Fallback: check via state attributes
        state = hass.states.get("image.rain_incoming_radar_128km")
        assert state is not None


# --- strings.json ---


def test_strings_json_exists():
    strings_path = COMPONENT_DIR / "strings.json"
    assert strings_path.is_file(), "strings.json must exist"


def test_strings_json_has_config_step_user():
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text())
    user_step = strings["config"]["step"]["user"]
    assert "title" in user_step
    assert "data" in user_step
    assert "latitude" in user_step["data"]
    assert "longitude" in user_step["data"]
    assert "lookahead_minutes" in user_step["data"]


def test_strings_json_has_config_errors():
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text())
    errors = strings["config"]["error"]
    assert "invalid_latitude" in errors
    assert "invalid_longitude" in errors
    assert "invalid_lookahead" in errors


def test_translations_en_json_exists():
    en_path = COMPONENT_DIR / "translations" / "en.json"
    assert en_path.is_file(), "translations/en.json must exist"


def test_translations_en_json_matches_strings_json():
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text())
    en = json.loads((COMPONENT_DIR / "translations" / "en.json").read_text())
    assert strings == en


# --- Binary sensor device class ---


@pytest.mark.asyncio
async def test_binary_sensor_device_class_is_moisture(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state is not None
    assert state.attributes.get("device_class") == BinarySensorDeviceClass.MOISTURE


# --- Dynamic icons ---


@pytest.mark.asyncio
async def test_binary_sensor_icon_pouring_when_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state.attributes["icon"] == "mdi:weather-pouring"


@pytest.mark.asyncio
async def test_binary_sensor_icon_sunny_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state.attributes["icon"] == "mdi:weather-sunny"


@pytest.mark.asyncio
async def test_arrival_sensor_icon_alert_when_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_arrival_time")
    assert state.attributes["icon"] == "mdi:clock-alert-outline"


@pytest.mark.asyncio
async def test_arrival_sensor_icon_clock_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_arrival_time")
    assert state.attributes["icon"] == "mdi:clock-outline"


# --- Attributes cleanup ---


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_id", [
    "binary_sensor.rain_incoming_status",
    "sensor.rain_incoming_arrival_time",
])
async def test_config_values_not_in_attributes(hass: HomeAssistant, mock_entry, entity_id):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get(entity_id)
    assert state is not None
    assert "latitude" not in state.attributes
    assert "longitude" not in state.attributes
    assert "lookahead_minutes" not in state.attributes


@pytest.mark.asyncio
async def test_binary_sensor_still_has_dynamic_attributes(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert "confidence" in state.attributes
    assert "frame_count" in state.attributes


@pytest.mark.asyncio
async def test_arrival_sensor_still_has_dynamic_attributes(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_arrival_time")
    assert "confidence" in state.attributes
    assert "frame_count" in state.attributes


# --- Named location device name ---


@pytest.fixture
def mock_entry_named():
    return MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_LOCATION_NAME: "Beach House",
        },
        entry_id="test_named_abc",
    )


@pytest.mark.asyncio
async def test_named_location_device_name(hass: HomeAssistant, mock_entry_named):
    await _setup_integration(hass, mock_entry_named, MOCK_RESULT_NO_RAIN)
    # Entity IDs use the device name - with a named location
    # the device is "Rain Incoming - Beach House", so entity IDs change
    state = hass.states.get("binary_sensor.rain_incoming_beach_house_status")
    assert state is not None, (
        f"Expected named entity. Available: {[s.entity_id for s in hass.states.async_all()]}"
    )


@pytest.mark.asyncio
async def test_unnamed_location_device_name(hass: HomeAssistant, mock_entry):
    """Entries without location_name still use 'Rain Incoming' device name."""
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("binary_sensor.rain_incoming_status")
    assert state is not None


# --- strings.json location_name ---


def test_strings_json_has_location_name_in_config():
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text())
    assert "location_name" in strings["config"]["step"]["user"]["data"]


def test_strings_json_has_location_name_in_options():
    strings = json.loads((COMPONENT_DIR / "strings.json").read_text())
    assert "location_name" in strings["options"]["step"]["init"]["data"]


# --- Intensity sensor ---


MOCK_RESULT_RAIN_OVERHEAD = DetectionResult(
    rain_incoming=True,
    arrival_time=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
    confidence=Confidence.NORMAL,
    frame_count=6,
    max_approaching_intensity=0.65,
)


@pytest.mark.asyncio
async def test_intensity_sensor_exists(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_intensity")
    assert state is not None, (
        f"Intensity sensor not found. Available: {[s.entity_id for s in hass.states.async_all()]}"
    )


@pytest.mark.asyncio
async def test_intensity_sensor_none_when_no_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_intensity")
    assert state.state == "none"


@pytest.mark.asyncio
async def test_intensity_sensor_shows_level_when_rain(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_intensity")
    # 0.65 maps to "heavy"
    assert state.state == "heavy"


@pytest.mark.asyncio
async def test_intensity_sensor_has_raw_attribute(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_intensity")
    assert "intensity_raw" in state.attributes
    assert state.attributes["intensity_raw"] == pytest.approx(0.65, abs=0.01)


@pytest.mark.asyncio
async def test_intensity_sensor_icon_sunny_when_none(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_intensity")
    assert state.attributes["icon"] == "mdi:weather-sunny"


@pytest.mark.asyncio
async def test_intensity_sensor_icon_pouring_when_heavy(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_RAIN_COMING)
    state = hass.states.get("sensor.rain_incoming_intensity")
    assert state.attributes["icon"] == "mdi:weather-pouring"


# --- Last rain sensor ---


@pytest.mark.asyncio
async def test_last_rain_sensor_exists(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_last_rain")
    assert state is not None, (
        f"Last rain sensor not found. Available: {[s.entity_id for s in hass.states.async_all()]}"
    )


@pytest.mark.asyncio
async def test_last_rain_sensor_unknown_when_no_rain_history(hass: HomeAssistant, mock_entry):
    await _setup_integration(hass, mock_entry, MOCK_RESULT_NO_RAIN)
    state = hass.states.get("sensor.rain_incoming_last_rain")
    assert state.state in ("unknown", "None")
