from __future__ import annotations

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_LOCATION_NAME,
    CONF_LOOKAHEAD_MINUTES,
    CONF_MAP_STYLE,
    DEFAULT_LOOKAHEAD_MINUTES,
    DOMAIN,
    MAX_LOCATION_NAME_CHARS,
    MAX_LOOKAHEAD_MINUTES,
    MIN_LOOKAHEAD_MINUTES,
)

_VALID_MAP_STYLES = ["voyager", "osm_standard", "osm_dark", "esri_imagery", "dark_matter"]
_DEFAULT_MAP_STYLE = "voyager"


def _validate_map_style(value: str) -> str:
    if value not in _VALID_MAP_STYLES:
        raise vol.Invalid(f"Invalid map style: {value!r}")
    return value


def _validate_location_name(value: str) -> str:
    if len(value) > MAX_LOCATION_NAME_CHARS:
        raise vol.Invalid(
            f"Location name must be {MAX_LOCATION_NAME_CHARS} characters or fewer"
            " (avoids overlap with map attribution on the bottom of the composite)."
        )
    return value


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

    location_name = user_input.get(CONF_LOCATION_NAME, "")
    if location_name and len(location_name) > MAX_LOCATION_NAME_CHARS:
        errors[CONF_LOCATION_NAME] = "location_name_too_long"

    map_style = user_input.get(CONF_MAP_STYLE, _DEFAULT_MAP_STYLE)
    if map_style not in _VALID_MAP_STYLES:
        errors[CONF_MAP_STYLE] = "invalid_map_style"

    return errors


def _validate_options_input(user_input: dict) -> dict[str, str]:
    """Validate options flow input (no lat/lon)."""
    errors: dict[str, str] = {}

    lookahead = user_input.get(CONF_LOOKAHEAD_MINUTES)
    if lookahead is not None and not (MIN_LOOKAHEAD_MINUTES <= lookahead <= MAX_LOOKAHEAD_MINUTES):
        errors[CONF_LOOKAHEAD_MINUTES] = "invalid_lookahead"

    location_name = user_input.get(CONF_LOCATION_NAME, "")
    if location_name and len(location_name) > MAX_LOCATION_NAME_CHARS:
        errors[CONF_LOCATION_NAME] = "location_name_too_long"

    map_style = user_input.get(CONF_MAP_STYLE, _DEFAULT_MAP_STYLE)
    if map_style not in _VALID_MAP_STYLES:
        errors[CONF_MAP_STYLE] = "invalid_map_style"

    return errors


def _build_title(user_input: dict) -> str:
    location_name = user_input.get(CONF_LOCATION_NAME, "").strip()
    if location_name:
        return f"Rain Incoming - {location_name}"
    lat = user_input[CONF_LATITUDE]
    lon = user_input[CONF_LONGITUDE]
    return f"Rain Incoming ({lat:.2f}, {lon:.2f})"


def _build_schema(
    default_lat: float,
    default_lon: float,
    default_lookahead: int = DEFAULT_LOOKAHEAD_MINUTES,
    default_location_name: str = "",
    default_map_style: str = _DEFAULT_MAP_STYLE,
) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_LATITUDE, default=default_lat): vol.Coerce(float),
            vol.Required(CONF_LONGITUDE, default=default_lon): vol.Coerce(float),
            vol.Required(CONF_LOOKAHEAD_MINUTES, default=default_lookahead): vol.Coerce(int),
            vol.Optional(CONF_LOCATION_NAME, default=default_location_name): str,
            vol.Optional(CONF_MAP_STYLE, default=default_map_style): SelectSelector(
                SelectSelectorConfig(
                    options=list(_VALID_MAP_STYLES),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="map_style",
                )
            ),
        }
    )


def _build_options_schema(
    default_lookahead: int = DEFAULT_LOOKAHEAD_MINUTES,
    default_location_name: str = "",
    default_map_style: str = _DEFAULT_MAP_STYLE,
) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_LOOKAHEAD_MINUTES, default=default_lookahead): vol.Coerce(int),
            vol.Optional(CONF_LOCATION_NAME, default=default_location_name): str,
            vol.Optional(CONF_MAP_STYLE, default=default_map_style): SelectSelector(
                SelectSelectorConfig(
                    options=list(_VALID_MAP_STYLES),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="map_style",
                )
            ),
        }
    )


class RainIncomingConfigFlow(ConfigFlow, domain=DOMAIN):
    """Config flow for Rain Incoming."""

    VERSION = 2

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return RainIncomingOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_input(user_input)
            if not errors:
                # Normalise: ensure map_style is present with default
                if CONF_MAP_STYLE not in user_input:
                    user_input = {**user_input, CONF_MAP_STYLE: _DEFAULT_MAP_STYLE}
                lat = user_input[CONF_LATITUDE]
                lon = user_input[CONF_LONGITUDE]
                await self.async_set_unique_id(f"{lat:.4f}_{lon:.4f}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=_build_title(user_input),
                    data=user_input,
                )

        schema = _build_schema(
            default_lat=user_input.get(CONF_LATITUDE, self.hass.config.latitude) if user_input else self.hass.config.latitude,
            default_lon=user_input.get(CONF_LONGITUDE, self.hass.config.longitude) if user_input else self.hass.config.longitude,
            default_lookahead=user_input.get(CONF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES) if user_input else DEFAULT_LOOKAHEAD_MINUTES,
            default_location_name=user_input.get(CONF_LOCATION_NAME, "") if user_input else "",
            default_map_style=user_input.get(CONF_MAP_STYLE, _DEFAULT_MAP_STYLE) if user_input else _DEFAULT_MAP_STYLE,
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )


class RainIncomingOptionsFlow(OptionsFlow):
    """Options flow: change map_style, location_name, and lookahead post-install.

    Latitude and longitude are install-time properties and are NOT editable here.
    """

    def __init__(self, config_entry: ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate_options_input(user_input)
            if not errors:
                # Merge the new options into entry.data so the title reflects changes.
                merged_data = dict(self._config_entry.data)
                if CONF_LOCATION_NAME in user_input:
                    merged_data[CONF_LOCATION_NAME] = user_input[CONF_LOCATION_NAME]
                if CONF_LOOKAHEAD_MINUTES in user_input:
                    merged_data[CONF_LOOKAHEAD_MINUTES] = user_input[CONF_LOOKAHEAD_MINUTES]

                self.hass.config_entries.async_update_entry(
                    self._config_entry,
                    data=merged_data,
                    title=_build_title({**merged_data}),
                )
                # Trigger reload so the coordinator picks up the new style immediately.
                self.hass.config_entries.async_schedule_reload(self._config_entry.entry_id)
                return self.async_create_entry(data=user_input)

        # Pre-populate from current options (preferred) then data (fallback).
        # When re-rendering after a validation error, use what the user typed so
        # they don't have to retype values they already entered correctly.
        current_options = self._config_entry.options
        current_data = self._config_entry.data
        if user_input is not None:
            # Re-render after validation failure: restore what the user typed.
            default_lookahead = user_input.get(CONF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES)
            default_location_name = user_input.get(CONF_LOCATION_NAME, "")
            default_map_style = user_input.get(CONF_MAP_STYLE, _DEFAULT_MAP_STYLE)
        else:
            # First render: use persisted values.
            default_lookahead = current_options.get(
                CONF_LOOKAHEAD_MINUTES,
                current_data.get(CONF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES),
            )
            default_location_name = current_options.get(
                CONF_LOCATION_NAME,
                current_data.get(CONF_LOCATION_NAME, ""),
            )
            default_map_style = current_options.get(
                CONF_MAP_STYLE,
                current_data.get(CONF_MAP_STYLE, _DEFAULT_MAP_STYLE),
            )
        schema = _build_options_schema(
            default_lookahead=default_lookahead,
            default_location_name=default_location_name,
            default_map_style=default_map_style,
        )
        return self.async_show_form(
            step_id="init", data_schema=schema, errors=errors
        )
