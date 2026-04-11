from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_LOCATION_NAME, DOMAIN
from .coordinator import RainDetectorCoordinator
from .radar.detector import Confidence


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: RainDetectorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([RainIncomingBinarySensor(coordinator, entry)])


class RainIncomingBinarySensor(CoordinatorEntity[RainDetectorCoordinator], BinarySensorEntity):
    """Binary sensor that is on when rain is detected approaching or at the location."""

    _attr_has_entity_name = True
    _attr_name = "Status"
    _attr_device_class = BinarySensorDeviceClass.MOISTURE

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_rain_incoming"

    @property
    def device_info(self) -> DeviceInfo:
        location_name = self._entry.data.get(CONF_LOCATION_NAME) or ""
        device_name = "Rain Incoming"
        if location_name:
            device_name = f"Rain Incoming - {location_name}"
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=device_name,
        )

    @property
    def icon(self) -> str:
        if self.is_on:
            return "mdi:weather-pouring"
        return "mdi:weather-sunny"

    @property
    def is_on(self) -> bool | None:
        if self.coordinator.data is None:
            return None
        if self.coordinator.data.confidence == Confidence.UNAVAILABLE:
            return None
        return self.coordinator.data.rain_incoming

    @property
    def available(self) -> bool:
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.coordinator.data.confidence != Confidence.UNAVAILABLE
        )

    @property
    def extra_state_attributes(self) -> dict:
        attrs: dict = {}
        if self.coordinator.data is not None:
            attrs["confidence"] = self.coordinator.data.confidence.value
            attrs["frame_count"] = self.coordinator.data.frame_count
        if self.coordinator.last_update_success_time is not None:
            attrs["last_update_success"] = self.coordinator.last_update_success_time.isoformat()
        return attrs
