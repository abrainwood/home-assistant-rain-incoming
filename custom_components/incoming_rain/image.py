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

from .const import CONF_LOCATION_NAME, DOMAIN, RADAR_GIF_FRAME_DURATION_MS, RADAR_RADII_KM
from .coordinator import RainDetectorCoordinator
from .radar.composite import render_animated_composite


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: RainDetectorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        RadarImageEntity(coordinator, entry, radius_km)
        for radius_km in RADAR_RADII_KM
    ])


class RadarImageEntity(CoordinatorEntity[RainDetectorCoordinator], ImageEntity):
    """Image entity showing a radar composite map."""

    _attr_has_entity_name = True
    _attr_content_type = "image/gif"

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry, radius_km: int) -> None:
        super().__init__(coordinator)
        ImageEntity.__init__(self, coordinator.hass)
        self._entry = entry
        self._radius_km = radius_km
        self._attr_name = f"Radar {radius_km}km"
        self._attr_unique_id = f"{entry.entry_id}_radar_{radius_km}km"
        self._cached_image: bytes | None = None
        self._cached_frame_path: str | None = None

    @property
    def device_info(self) -> DeviceInfo:
        location_name = self._entry.data.get(CONF_LOCATION_NAME) or ""
        device_name = "Incoming Rain"
        if location_name:
            device_name = f"Incoming Rain - {location_name}"
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=device_name,
        )

    @property
    def image_last_updated(self) -> datetime | None:
        return self.coordinator.last_update_success_time

    async def async_image(self) -> bytes | None:
        frame_path = self.coordinator.latest_frame_path
        if frame_path is None:
            return None

        # Only re-render if the latest frame path changed
        if frame_path == self._cached_frame_path and self._cached_image is not None:
            return self._cached_image

        frame_paths = self.coordinator.frame_paths
        if not frame_paths:
            return None

        data = self._entry.data
        lat = data.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = data.get(CONF_LONGITUDE, self.hass.config.longitude)

        # Convert timestamps to HA's configured timezone for display
        import zoneinfo
        ha_tz = zoneinfo.ZoneInfo(self.hass.config.time_zone)
        local_timestamps = [
            ts.astimezone(ha_tz) for ts in self.coordinator.frame_timestamps
        ] if self.coordinator.frame_timestamps else None

        session = async_get_clientsession(self.hass)
        cells = self.coordinator.tracked_cells or None
        self._cached_image = await render_animated_composite(
            lat=lat,
            lon=lon,
            radius_km=self._radius_km,
            frame_paths=frame_paths,
            frame_duration_ms=RADAR_GIF_FRAME_DURATION_MS,
            frame_timestamps=local_timestamps,
            tz_name=self.hass.config.time_zone,
            tracked_cells=cells,
            session=session,
            run_in_executor=self.hass.async_add_executor_job,
        )
        self._cached_frame_path = frame_path
        return self._cached_image
