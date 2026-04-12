from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import SelectSelector
from pytest_homeassistant_custom_component.common import MockConfigEntry
from unittest.mock import patch

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE

from custom_components.rain_incoming.const import (
    CONF_LOCATION_NAME,
    CONF_MAP_STYLE,
    DOMAIN,
)
from .conftest import setup_integration
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult

_MOCK_RESULT = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
    max_approaching_intensity=0.0,
)


@pytest.mark.asyncio
async def test_config_flow_shows_form(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"


@pytest.mark.asyncio
async def test_config_flow_creates_entry_with_valid_input(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"]["latitude"] == -33.701
    assert result["data"]["longitude"] == 151.209
    assert result["data"]["lookahead_minutes"] == 60


@pytest.mark.asyncio
async def test_config_flow_rejects_invalid_latitude(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": 999.0,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert "latitude" in result["errors"]


@pytest.mark.asyncio
async def test_config_flow_title_uses_location_name_when_provided(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "location_name": "Beach House",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Rain Incoming - Beach House"
    assert result["data"]["location_name"] == "Beach House"


@pytest.mark.asyncio
async def test_config_flow_title_uses_coordinates_when_name_empty(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "location_name": "",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Rain Incoming (-33.70, 151.21)"


@pytest.mark.asyncio
async def test_config_flow_title_uses_coordinates_when_name_omitted(hass: HomeAssistant):
    """Existing entries without location_name should still work."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Rain Incoming (-33.70, 151.21)"


@pytest.mark.asyncio
async def test_config_flow_rejects_lookahead_out_of_range(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 200,  # > 120
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert "lookahead_minutes" in result["errors"]


# ---------------------------------------------------------------------------
# PR 3: map_style field + location_name validator + options flow + migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_flow_rejects_location_name_over_30_chars(hass: HomeAssistant):
    """RED: location_name > 30 chars should be rejected with a validation error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "location_name": "A" * 40,  # 40 chars - over the 30-char limit
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert "location_name" in result["errors"]


@pytest.mark.asyncio
async def test_config_flow_stores_map_style_in_entry(hass: HomeAssistant):
    """RED: map_style should be stored in entry.data when submitted."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "map_style": "osm_dark",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MAP_STYLE] == "osm_dark"


@pytest.mark.asyncio
async def test_config_flow_defaults_map_style_to_voyager(hass: HomeAssistant):
    """Submitting without map_style should default to 'voyager'."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"].get(CONF_MAP_STYLE) == "voyager"


@pytest.mark.asyncio
async def test_config_flow_rejects_invalid_map_style(hass: HomeAssistant):
    """Submitting an unrecognised map_style string should be rejected.

    With SelectSelector, HA validates the value against the allowed list at the
    schema level and raises InvalidData before the flow returns a form with errors.
    """
    from homeassistant.data_entry_flow import InvalidData

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    with pytest.raises(InvalidData):
        await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "latitude": -33.701,
                "longitude": 151.209,
                "lookahead_minutes": 60,
                "map_style": "not_a_real_style",
            },
        )


@pytest.mark.asyncio
async def test_options_flow_changes_map_style(hass: HomeAssistant):
    """Options flow should persist a new map_style to entry.options."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        version=2,
    )
    await setup_integration(hass, entry, _MOCK_RESULT)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "location_name": "Beach House",
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "esri_imagery",
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_MAP_STYLE] == "esri_imagery"


@pytest.mark.asyncio
async def test_options_flow_rejects_location_name_over_30_chars(hass: HomeAssistant):
    """Options flow should reject location_name > 30 chars."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        version=2,
    )
    await setup_integration(hass, entry, _MOCK_RESULT)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "location_name": "B" * 40,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert "location_name" in result["errors"]


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_adds_map_style_to_v1_entry(hass: HomeAssistant):
    """RED: a v1 entry without map_style should be migrated to v2 with map_style=voyager."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
        version=1,
    )
    entry.add_to_hass(hass)

    # Loading the integration should trigger async_migrate_entry and bump the version.
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # After migration, the entry should be on version 2 with map_style set.
    assert entry.version == 2
    assert entry.data.get(CONF_MAP_STYLE) == "voyager"


@pytest.mark.asyncio
async def test_migration_preserves_long_location_name(hass: HomeAssistant):
    """v1 entry with location_name > 30 chars should migrate without rejection."""
    long_name = "A" * 50
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "location_name": long_name,
        },
        version=1,
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Migration should succeed - render-time truncation handles display.
    assert entry.version == 2
    assert entry.data.get(CONF_MAP_STYLE) == "voyager"
    assert entry.data["location_name"] == long_name  # preserved, not truncated


@pytest.mark.asyncio
async def test_migration_does_not_re_migrate_v2_entry(hass: HomeAssistant):
    """A v2 entry should not be migrated a second time."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "osm_dark",
        },
        version=2,
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # map_style should remain unchanged at the value set at v2 creation.
    assert entry.version == 2
    assert entry.data[CONF_MAP_STYLE] == "osm_dark"


# ---------------------------------------------------------------------------
# Coordinator wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_uses_options_map_style_over_data(hass: HomeAssistant):
    """Coordinator should prefer entry.options[CONF_MAP_STYLE] over entry.data."""
    from custom_components.rain_incoming.coordinator import RainDetectorCoordinator

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        options={CONF_MAP_STYLE: "esri_imagery"},
        version=2,
    )
    entry.add_to_hass(hass)

    coordinator = RainDetectorCoordinator(hass, entry)
    assert coordinator.map_style == "esri_imagery"


@pytest.mark.asyncio
async def test_coordinator_falls_back_to_data_map_style(hass: HomeAssistant):
    """Coordinator uses entry.data[CONF_MAP_STYLE] when options doesn't have it."""
    from custom_components.rain_incoming.coordinator import RainDetectorCoordinator

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "osm_standard",
        },
        options={},
        version=2,
    )
    entry.add_to_hass(hass)

    coordinator = RainDetectorCoordinator(hass, entry)
    assert coordinator.map_style == "osm_standard"


@pytest.mark.asyncio
async def test_coordinator_defaults_to_voyager_when_no_style_set(hass: HomeAssistant):
    """Coordinator defaults to voyager when neither options nor data has map_style."""
    from custom_components.rain_incoming.coordinator import RainDetectorCoordinator

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
        },
        options={},
        version=2,
    )
    entry.add_to_hass(hass)

    coordinator = RainDetectorCoordinator(hass, entry)
    assert coordinator.map_style == "voyager"


# ---------------------------------------------------------------------------
# Fix 1: SelectSelector for map_style dropdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_style_field_uses_select_selector_in_config_flow(hass: HomeAssistant):
    """CONF_MAP_STYLE in config flow schema must use SelectSelector for dropdown UI."""
    from custom_components.rain_incoming.config_flow import _build_schema

    schema = _build_schema(
        default_lat=-33.701,
        default_lon=151.209,
    )
    for key in schema.schema:
        if getattr(key, "schema", None) == CONF_MAP_STYLE:
            assert isinstance(schema.schema[key], SelectSelector), (
                f"CONF_MAP_STYLE should use SelectSelector for dropdown UI, "
                f"got {type(schema.schema[key]).__name__}"
            )
            return
    pytest.fail("CONF_MAP_STYLE not found in config flow schema")


@pytest.mark.asyncio
async def test_map_style_field_uses_select_selector_in_options_flow(hass: HomeAssistant):
    """CONF_MAP_STYLE in options flow schema must use SelectSelector for dropdown UI."""
    from custom_components.rain_incoming.config_flow import _build_options_schema

    schema = _build_options_schema()
    for key in schema.schema:
        if getattr(key, "schema", None) == CONF_MAP_STYLE:
            assert isinstance(schema.schema[key], SelectSelector), (
                f"CONF_MAP_STYLE should use SelectSelector for dropdown UI, "
                f"got {type(schema.schema[key]).__name__}"
            )
            return
    pytest.fail("CONF_MAP_STYLE not found in options flow schema")


# ---------------------------------------------------------------------------
# Fix 2: async_schedule_reload assertion in options flow test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_options_flow_schedules_reload_after_save(hass: HomeAssistant):
    """Options flow must call async_schedule_reload after saving so changes take effect."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        version=2,
    )
    await setup_integration(hass, entry, _MOCK_RESULT)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM

    with patch.object(
        hass.config_entries, "async_schedule_reload"
    ) as mock_reload:
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                "location_name": "Beach House",
                "lookahead_minutes": 60,
                CONF_MAP_STYLE: "esri_imagery",
            },
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    mock_reload.assert_called_once_with(entry.entry_id)


# ---------------------------------------------------------------------------
# Fix 3: boundary test for 30-char location name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_flow_accepts_location_name_exactly_30_chars(hass: HomeAssistant):
    """Location name of exactly 30 chars must be accepted (boundary at limit)."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            "location_name": "A" * 30,  # exactly at the 30-char limit
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_LOCATION_NAME] == "A" * 30


# ---------------------------------------------------------------------------
# Task 125 Fix 1: sticky inputs on validation error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_flow_preserves_user_input_on_validation_error(hass: HomeAssistant):
    """When validation fails the form re-renders with the user's typed values.

    We submit an invalid location name (too long) alongside a non-default lookahead.
    The re-rendered schema's default for lookahead_minutes must reflect what the
    user typed - not the system default - so the user doesn't have to retype it.
    """
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    # Submit with a too-long location name and a non-default lookahead.
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 45,  # non-default (default is 60)
            "location_name": "X" * 40,  # too long, will be rejected
            "map_style": "voyager",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert "location_name" in result["errors"]

    # Introspect the re-rendered schema to verify defaults reflect user input.
    schema = result["data_schema"]
    defaults = {
        (key.schema if hasattr(key, "schema") else str(key)): (
            key.default() if callable(getattr(key, "default", None)) else None
        )
        for key in schema.schema.keys()
    }

    assert defaults.get("lookahead_minutes") == 45, (
        f"Expected lookahead_minutes default to be 45 (user's value), got {defaults.get('lookahead_minutes')}"
    )
    assert defaults.get("latitude") == -33.701
    assert defaults.get("longitude") == 151.209


@pytest.mark.asyncio
async def test_options_flow_preserves_user_input_on_validation_error(hass: HomeAssistant):
    """Options flow: form re-renders with user's typed values after validation error."""
    from custom_components.rain_incoming.const import CONF_LOOKAHEAD_MINUTES

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        version=2,
    )
    await setup_integration(hass, entry, _MOCK_RESULT)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM

    # Submit with an invalid location name and a non-default lookahead.
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "location_name": "Y" * 40,  # too long
            "lookahead_minutes": 25,  # non-default
            CONF_MAP_STYLE: "osm_dark",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert "location_name" in result["errors"]

    schema = result["data_schema"]
    defaults = {
        (key.schema if hasattr(key, "schema") else str(key)): (
            key.default() if callable(getattr(key, "default", None)) else None
        )
        for key in schema.schema.keys()
    }

    assert defaults.get("lookahead_minutes") == 25, (
        f"Expected lookahead_minutes default to be 25 (user's value), got {defaults.get('lookahead_minutes')}"
    )


# ---------------------------------------------------------------------------
# Task 125 Fix 2: constraint hints in field labels (translation file checks)
# ---------------------------------------------------------------------------


def test_translation_includes_location_name_max_length():
    """location_name label in config flow must mention the 30-char limit."""
    import json
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "../../custom_components/rain_incoming/translations/en.json",
    )
    with open(path) as f:
        translations = json.load(f)

    label = translations["config"]["step"]["user"]["data"]["location_name"]
    assert "30" in label, f"Expected '30' in location_name label, got: {label!r}"
    assert "character" in label.lower(), (
        f"Expected 'character' in location_name label, got: {label!r}"
    )


def test_translation_includes_lookahead_range_in_config_flow():
    """lookahead_minutes label in config flow must mention the 20-60 range."""
    import json
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "../../custom_components/rain_incoming/translations/en.json",
    )
    with open(path) as f:
        translations = json.load(f)

    label = translations["config"]["step"]["user"]["data"]["lookahead_minutes"]
    assert "20" in label and "60" in label, (
        f"Expected '20' and '60' in lookahead_minutes label, got: {label!r}"
    )


def test_translation_includes_location_name_max_length_in_options_flow():
    """location_name label in options flow must mention the 30-char limit."""
    import json
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "../../custom_components/rain_incoming/translations/en.json",
    )
    with open(path) as f:
        translations = json.load(f)

    label = translations["options"]["step"]["init"]["data"]["location_name"]
    assert "30" in label, f"Expected '30' in options location_name label, got: {label!r}"
    assert "character" in label.lower(), (
        f"Expected 'character' in options location_name label, got: {label!r}"
    )


def test_translation_includes_lookahead_range_in_options_flow():
    """lookahead_minutes label in options flow must mention the 20-60 range."""
    import json
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "../../custom_components/rain_incoming/translations/en.json",
    )
    with open(path) as f:
        translations = json.load(f)

    label = translations["options"]["step"]["init"]["data"]["lookahead_minutes"]
    assert "20" in label and "60" in label, (
        f"Expected '20' and '60' in options lookahead_minutes label, got: {label!r}"
    )


# ---------------------------------------------------------------------------
# Fix 4: lat/lon immutability via options flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_options_flow_does_not_modify_latitude_longitude(hass: HomeAssistant):
    """Lat/lon in entry.data must be unchanged after an options flow save."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "latitude": -33.701,
            "longitude": 151.209,
            "lookahead_minutes": 60,
            CONF_MAP_STYLE: "voyager",
        },
        version=2,
    )
    await setup_integration(hass, entry, _MOCK_RESULT)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM

    # Submit with only the fields the options flow knows about (no lat/lon).
    # Lat/lon are not in the options schema so they cannot be submitted.
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_MAP_STYLE: "esri_imagery",
            "location_name": "Updated Name",
            "lookahead_minutes": 60,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY

    # Verify latitude/longitude in entry.data are unchanged.
    assert entry.data[CONF_LATITUDE] == -33.701
    assert entry.data[CONF_LONGITUDE] == 151.209
