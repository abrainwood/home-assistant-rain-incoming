from __future__ import annotations

import logging

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_LOCATION_NAME, DOMAIN
from .coordinator import RainDetectorCoordinator
from .radar.detector import Confidence

_LOGGER = logging.getLogger(__name__)


def _intensity_label(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value <= 0.20:
        return "light"
    if value <= 0.45:
        return "moderate"
    if value <= 0.75:
        return "heavy"
    return "extreme"


_INTENSITY_ICONS = {
    "none": "mdi:weather-sunny",
    "light": "mdi:weather-partly-rainy",
    "moderate": "mdi:weather-rainy",
    "heavy": "mdi:weather-pouring",
    "extreme": "mdi:weather-pouring",
}


def _device_info(entry: ConfigEntry) -> DeviceInfo:
    location_name = entry.data.get(CONF_LOCATION_NAME) or ""
    device_name = "Rain Incoming"
    if location_name:
        device_name = f"Rain Incoming - {location_name}"
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=device_name,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: RainDetectorCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        RainArrivalTimeSensor(coordinator, entry),
        RainIntensitySensor(coordinator, entry),
        LastRainNearbySensor(coordinator, entry),
    ])


class RainArrivalTimeSensor(CoordinatorEntity[RainDetectorCoordinator], SensorEntity):
    """Sensor showing the predicted arrival time of incoming rain."""

    _attr_has_entity_name = True
    _attr_name = "Arrival Time"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_rain_arrival_time"

    @property
    def device_info(self) -> DeviceInfo:
        return _device_info(self._entry)

    @property
    def icon(self) -> str:
        if self.coordinator.data and self.coordinator.data.rain_incoming:
            return "mdi:clock-alert-outline"
        return "mdi:clock-outline"

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
        attrs: dict = {}
        if self.coordinator.data is not None:
            attrs["confidence"] = self.coordinator.data.confidence.value
            attrs["frame_count"] = self.coordinator.data.frame_count
        return attrs


class RainIntensitySensor(CoordinatorEntity[RainDetectorCoordinator], SensorEntity):
    """Sensor showing the maximum expected precipitation intensity."""

    _attr_has_entity_name = True
    _attr_name = "Intensity"

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_rain_intensity"

    @property
    def device_info(self) -> DeviceInfo:
        return _device_info(self._entry)

    @property
    def icon(self) -> str:
        label = self._intensity_label
        return _INTENSITY_ICONS.get(label, "mdi:weather-sunny")

    @property
    def _intensity_label(self) -> str:
        if self.coordinator.data is None:
            return "none"
        return _intensity_label(self.coordinator.data.max_approaching_intensity)

    @property
    def native_value(self) -> str:
        return self._intensity_label

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
            attrs["intensity_raw"] = self.coordinator.data.max_approaching_intensity
        return attrs


class LastRainNearbySensor(
    CoordinatorEntity[RainDetectorCoordinator], RestoreEntity, SensorEntity
):
    """Sensor recording the last time rain was detected at the location."""

    _attr_has_entity_name = True
    _attr_name = "Last Rain"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: RainDetectorCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_last_rain_nearby"

    @property
    def device_info(self) -> DeviceInfo:
        return _device_info(self._entry)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            from datetime import datetime
            try:
                self.coordinator.last_rain_nearby_time = datetime.fromisoformat(
                    last_state.state
                )
            except (ValueError, TypeError):
                _LOGGER.warning(
                    "Failed to restore last rain nearby time from state '%s'",
                    last_state.state,
                )

    @property
    def native_value(self):
        return self.coordinator.last_rain_nearby_time

    @property
    def available(self) -> bool:
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
            and self.coordinator.data.confidence != Confidence.UNAVAILABLE
        )
