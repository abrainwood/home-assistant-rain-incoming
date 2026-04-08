from __future__ import annotations

from datetime import datetime

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_RADAR_RADIUS_KM, DEFAULT_RADAR_RADIUS_KM, DOMAIN
from .coordinator import RainDetectorCoordinator
from .radar.composite import render_composite


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: RainDetectorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([RadarImageEntity(coordinator, entry)])


class RadarImageEntity(CoordinatorEntity[RainDetectorCoordinator], ImageEntity):
    """Image entity showing a radar composite map."""

    _attr_has_entity_name = True
    _attr_name = "Radar"
    _attr_content_type = "image/png"

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        ImageEntity.__init__(self, coordinator.hass)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_radar"
        self._cached_image: bytes | None = None
        self._cached_frame_path: str | None = None

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name="Incoming Rain",
        )

    @property
    def image_last_updated(self) -> datetime | None:
        return self.coordinator.last_update_success_time

    async def async_image(self) -> bytes | None:
        frame_path = self.coordinator.latest_frame_path
        if frame_path is None:
            return None

        # Only re-render if the frame path changed
        if frame_path == self._cached_frame_path and self._cached_image is not None:
            return self._cached_image

        data = self._entry.data
        lat = data.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = data.get(CONF_LONGITUDE, self.hass.config.longitude)
        radius_km = data.get(CONF_RADAR_RADIUS_KM, DEFAULT_RADAR_RADIUS_KM)

        session = async_get_clientsession(self.hass)
        self._cached_image = await render_composite(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            frame_path=frame_path,
            session=session,
            run_in_executor=self.hass.async_add_executor_job,
        )
        self._cached_frame_path = frame_path
        return self._cached_image
