"""Shared entity helpers used across binary_sensor, sensor, and image entities."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity import DeviceInfo

from .const import CONF_LOCATION_NAME, DOMAIN


def device_info_from_entry(entry: ConfigEntry) -> DeviceInfo:
    """Build a DeviceInfo grouping all entities under a single Rain Incoming device."""
    location_name = entry.data.get(CONF_LOCATION_NAME) or ""
    device_name = "Rain Incoming"
    if location_name:
        device_name = f"Rain Incoming - {location_name}"
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=device_name,
    )
