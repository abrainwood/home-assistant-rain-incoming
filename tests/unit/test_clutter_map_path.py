"""Test that each config entry gets its own clutter map file.

I3: With a shared filename, multiple locations overwrite each other's
clutter maps. The filename must include the entry ID.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.rain_incoming.coordinator import RainDetectorCoordinator


def _make_hass():
    hass = MagicMock()
    hass.config.path.return_value = "/fake/.storage"
    hass.config.latitude = -33.7
    hass.config.longitude = 151.2
    return hass


def _make_entry(entry_id: str):
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.data = {"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 60}
    return entry


class TestClutterMapPath:
    def test_clutter_path_includes_entry_id(self):
        """The clutter map filename must include the entry ID so each
        location gets its own file."""
        hass = _make_hass()
        entry = _make_entry("abc123")
        coordinator = RainDetectorCoordinator(hass, entry)

        assert "abc123" in coordinator._clutter_path, (
            f"Clutter path must include entry ID. Got: {coordinator._clutter_path}"
        )

    def test_different_entries_get_different_paths(self):
        """Two coordinators with different entry IDs must use different files."""
        hass = _make_hass()
        coord_a = RainDetectorCoordinator(hass, _make_entry("location_a"))
        coord_b = RainDetectorCoordinator(hass, _make_entry("location_b"))

        assert coord_a._clutter_path != coord_b._clutter_path, (
            f"Different entries must use different clutter paths. "
            f"Both got: {coord_a._clutter_path}"
        )
