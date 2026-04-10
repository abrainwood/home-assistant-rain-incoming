from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.rain_incoming.const import DOMAIN


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
