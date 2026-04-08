from __future__ import annotations

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import callback

from .const import (
    CONF_LOOKAHEAD_MINUTES,
    DEFAULT_LOOKAHEAD_MINUTES,
    DOMAIN,
    MAX_LOOKAHEAD_MINUTES,
    MIN_LOOKAHEAD_MINUTES,
)


def _validate_input(user_input: dict) -> dict[str, str]:
    errors: dict[str, str] = {}
    lat = user_input[CONF_LATITUDE]
    lon = user_input[CONF_LONGITUDE]
    lookahead = user_input[CONF_LOOKAHEAD_MINUTES]

    if not (-90 <= lat <= 90):
        errors[CONF_LATITUDE] = "invalid_latitude"
    elif not (-180 <= lon <= 180):
        errors[CONF_LONGITUDE] = "invalid_longitude"
    elif not (MIN_LOOKAHEAD_MINUTES <= lookahead <= MAX_LOOKAHEAD_MINUTES):
        errors[CONF_LOOKAHEAD_MINUTES] = "invalid_lookahead"
    return errors


def _build_schema(
    default_lat: float, default_lon: float, default_lookahead: int = DEFAULT_LOOKAHEAD_MINUTES
) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_LATITUDE, default=default_lat): vol.Coerce(float),
            vol.Required(CONF_LONGITUDE, default=default_lon): vol.Coerce(float),
            vol.Required(CONF_LOOKAHEAD_MINUTES, default=default_lookahead): vol.Coerce(int),
        }
    )


class IncomingRainConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for Incoming Rain."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return IncomingRainOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_input(user_input)
            if not errors:
                lat = user_input[CONF_LATITUDE]
                lon = user_input[CONF_LONGITUDE]
                await self.async_set_unique_id(f"{lat:.4f}_{lon:.4f}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"Incoming Rain ({lat:.2f}, {lon:.2f})",
                    data=user_input,
                )

        schema = _build_schema(
            self.hass.config.latitude,
            self.hass.config.longitude,
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )


class IncomingRainOptionsFlow(OptionsFlow):
    """Options flow to reconfigure location and lookahead after setup."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_input(user_input)
            if not errors:
                # Update the config entry data with new values
                self.hass.config_entries.async_update_entry(
                    self._config_entry,
                    data=user_input,
                    title=f"Incoming Rain ({user_input[CONF_LATITUDE]:.2f}, {user_input[CONF_LONGITUDE]:.2f})",
                )
                await self.hass.config_entries.async_reload(self._config_entry.entry_id)
                return self.async_create_entry(data={})

        current = self._config_entry.data
        schema = _build_schema(
            current.get(CONF_LATITUDE, self.hass.config.latitude),
            current.get(CONF_LONGITUDE, self.hass.config.longitude),
            current.get(CONF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES),
        )
        return self.async_show_form(
            step_id="init", data_schema=schema, errors=errors
        )
