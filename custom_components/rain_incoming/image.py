from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from homeassistant.components.image import ImageEntity

_LOGGER = logging.getLogger(__name__)
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
        device_name = "Rain Incoming"
        if location_name:
            device_name = f"Rain Incoming - {location_name}"
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=device_name,
        )

    @property
    def image_last_updated(self) -> datetime | None:
        return self.coordinator.last_update_success_time

    @property
    def available(self) -> bool:
        # Always available - we show cached data or a placeholder
        return True

    @property
    def extra_state_attributes(self) -> dict:
        attrs: dict = {}
        if (
            self._cached_image is not None
            and not self.coordinator.last_update_success
            and self.coordinator.last_update_success_time is not None
        ):
            attrs["stale_since"] = self.coordinator.last_update_success_time.isoformat()
        return attrs

    def _make_placeholder(self) -> bytes:
        """Generate a placeholder GIF with 'Waiting for radar data...' text."""
        img = Image.new("RGB", (640, 640), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()
        text = "Waiting for radar data..."
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((640 - tw) // 2, (640 - th) // 2), text, fill=(180, 180, 180), font=font)
        buf = BytesIO()
        img.save(buf, format="GIF")
        return buf.getvalue()

    async def async_image(self) -> bytes | None:
        frame_path = self.coordinator.latest_frame_path
        if frame_path is None:
            if self._cached_image is not None:
                _LOGGER.debug("Radar %dkm: returning stale cached image (no fresh data)", self._radius_km)
                return self._cached_image
            _LOGGER.debug("Radar %dkm: returning placeholder (no data yet)", self._radius_km)
            return self._make_placeholder()

        # Only re-render if the latest frame path changed
        if frame_path == self._cached_frame_path and self._cached_image is not None:
            return self._cached_image

        frame_paths = self.coordinator.frame_paths
        if not frame_paths:
            if self._cached_image is not None:
                _LOGGER.debug("Radar %dkm: returning stale cached image (no frame paths)", self._radius_km)
                return self._cached_image
            _LOGGER.debug("Radar %dkm: returning placeholder (no frame paths)", self._radius_km)
            return self._make_placeholder()

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
        conf_maps = self.coordinator.confidence_maps or None
        location_name = data.get(CONF_LOCATION_NAME) or None

        _LOGGER.debug(
            "Radar %dkm: rendering %d frames for %s",
            self._radius_km, len(frame_paths), location_name or "default location",
        )
        try:
            self._cached_image = await render_animated_composite(
                lat=lat,
                lon=lon,
                radius_km=self._radius_km,
                frame_paths=frame_paths,
                frame_duration_ms=RADAR_GIF_FRAME_DURATION_MS,
                frame_timestamps=local_timestamps,
                tz_name=self.hass.config.time_zone,
                confidence_maps=conf_maps,
                location_name=location_name,
                session=session,
                run_in_executor=self.hass.async_add_executor_job,
            )
            self._cached_frame_path = frame_path
            _LOGGER.debug(
                "Radar %dkm: rendered %d bytes",
                self._radius_km, len(self._cached_image) if self._cached_image else 0,
            )
        except Exception:
            _LOGGER.exception(
                "Radar %dkm: rendering failed for %s",
                self._radius_km, location_name or "default location",
            )
            # Return whatever we have - stale cache or placeholder
            if self._cached_image is not None:
                return self._cached_image
            return self._make_placeholder()

        return self._cached_image
