from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE

from .const import CONF_LOOKAHEAD_MINUTES, DOMAIN
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
    _attr_name = "Rain Incoming"
    _attr_icon = "mdi:weather-rainy"

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_rain_incoming"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name="Incoming Rain",
        )

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
        attrs = {
            "latitude": self._entry.data.get(CONF_LATITUDE),
            "longitude": self._entry.data.get(CONF_LONGITUDE),
            "lookahead_minutes": self._entry.data.get(CONF_LOOKAHEAD_MINUTES),
        }
        if self.coordinator.data is not None:
            attrs["confidence"] = self.coordinator.data.confidence.value
            attrs["frame_count"] = self.coordinator.data.frame_count
        return attrs
