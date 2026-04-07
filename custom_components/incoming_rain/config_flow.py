from __future__ import annotations

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE

from .const import (
    CONF_LOOKAHEAD_MINUTES,
    DEFAULT_LOOKAHEAD_MINUTES,
    DOMAIN,
    MAX_LOOKAHEAD_MINUTES,
    MIN_LOOKAHEAD_MINUTES,
)


class IncomingRainConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for Incoming Rain."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            lat = user_input[CONF_LATITUDE]
            lon = user_input[CONF_LONGITUDE]
            lookahead = user_input[CONF_LOOKAHEAD_MINUTES]

            if not (-90 <= lat <= 90):
                errors[CONF_LATITUDE] = "invalid_latitude"
            elif not (-180 <= lon <= 180):
                errors[CONF_LONGITUDE] = "invalid_longitude"
            elif not (MIN_LOOKAHEAD_MINUTES <= lookahead <= MAX_LOOKAHEAD_MINUTES):
                errors[CONF_LOOKAHEAD_MINUTES] = "invalid_lookahead"
            else:
                await self.async_set_unique_id(f"{lat:.4f}_{lon:.4f}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"Incoming Rain ({lat:.2f}, {lon:.2f})",
                    data=user_input,
                )

        default_lat = self.hass.config.latitude
        default_lon = self.hass.config.longitude

        schema = vol.Schema(
            {
                vol.Required(CONF_LATITUDE, default=default_lat): vol.Coerce(float),
                vol.Required(CONF_LONGITUDE, default=default_lon): vol.Coerce(float),
                vol.Required(
                    CONF_LOOKAHEAD_MINUTES, default=DEFAULT_LOOKAHEAD_MINUTES
                ): vol.All(
                    vol.Coerce(int),
                    vol.Range(min=MIN_LOOKAHEAD_MINUTES, max=MAX_LOOKAHEAD_MINUTES),
                ),
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )
