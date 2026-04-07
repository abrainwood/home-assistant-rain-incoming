from __future__ import annotations

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import RainDetectorCoordinator
from .radar.detector import Confidence


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: RainDetectorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([RainArrivalTimeSensor(coordinator, entry)])


class RainArrivalTimeSensor(CoordinatorEntity[RainDetectorCoordinator], SensorEntity):
    """Sensor showing the predicted arrival time of incoming rain."""

    _attr_has_entity_name = True
    _attr_name = "Rain Arrival Time"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:clock-time-four-outline"

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_rain_arrival_time"

    @property
    def native_value(self):
        if self.coordinator.data is None:
            return None
        if self.coordinator.data.confidence == Confidence.UNAVAILABLE:
            return None
        return self.coordinator.data.arrival_time

    @property
    def available(self) -> bool:
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.coordinator.data.confidence != Confidence.UNAVAILABLE
        )

    @property
    def extra_state_attributes(self) -> dict:
        if self.coordinator.data is None:
            return {}
        return {
            "confidence": self.coordinator.data.confidence.value,
            "frame_count": self.coordinator.data.frame_count,
        }
