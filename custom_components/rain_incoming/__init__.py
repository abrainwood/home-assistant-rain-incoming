from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.start import async_at_started

from .const import CONF_MAP_STYLE, DOMAIN
from .coordinator import RainDetectorCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["binary_sensor", "image", "sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = RainDetectorCoordinator(hass, entry)

    # Don't block HA startup with async_config_entry_first_refresh().
    # The first radar fetch can take 10-30+ seconds. Instead, set up
    # platforms immediately (sensors show unavailable) and schedule the
    # first data fetch after HA has fully started.
    #
    # async_at_started handles both cases (already started vs. waiting for startup)
    # and returns an unsubscribe callable that is safe to call at any time - including
    # after the listener has already fired and auto-removed itself. This avoids a
    # ValueError from the previous async_listen_once pattern where on_unload would
    # attempt to remove an already-removed listener.
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    async def _do_first_refresh(_hass: HomeAssistant) -> None:
        await coordinator.async_request_refresh()

    entry.async_on_unload(async_at_started(hass, _do_first_refresh))

    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entries to the current schema version.

    v1 -> v2: add map_style="voyager" to entry.data.
    Long location_name values are preserved as-is; PR2's render-time truncation
    handles display, and the validator only applies to new input via the flows.
    """
    _LOGGER.debug(
        "Migrating Rain Incoming config entry %s from version %d",
        entry.entry_id, entry.version,
    )

    if entry.version == 1:
        new_data = dict(entry.data)
        new_data.setdefault(CONF_MAP_STYLE, "voyager")
        hass.config_entries.async_update_entry(entry, data=new_data, version=2)
        _LOGGER.info(
            "Migrated Rain Incoming entry %s from v1 to v2: added map_style=voyager",
            entry.entry_id,
        )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator: RainDetectorCoordinator | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is not None:
        await coordinator.async_save_clutter_map()

    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok
