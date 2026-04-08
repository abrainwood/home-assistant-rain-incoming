"""
E2E tests for the incoming_rain integration.

These tests run against a real HA instance in Docker with a mock RainViewer
server. They verify the full pipeline: HTTP fetch -> tile decoding ->
detection -> sensor state updates.
"""
from __future__ import annotations

import time

import pytest


BINARY_SENSOR = "binary_sensor.incoming_rain_status"
ARRIVAL_SENSOR = "sensor.incoming_rain_arrival_time"
IMAGE_64 = "image.incoming_rain_radar_64km"
IMAGE_128 = "image.incoming_rain_radar_128km"
IMAGE_256 = "image.incoming_rain_radar_256km"


class TestIntegrationLoaded:
    def test_binary_sensor_exists(self, ha_client):
        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None, f"Entity {BINARY_SENSOR} not found"
        assert state["state"] in ("on", "off", "unavailable")

    def test_arrival_sensor_exists(self, ha_client):
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None, f"Entity {ARRIVAL_SENSOR} not found"

    @pytest.mark.parametrize("entity_id", [IMAGE_64, IMAGE_128, IMAGE_256])
    def test_image_entities_exist(self, ha_client, entity_id):
        state = ha_client.get_state(entity_id)
        assert state is not None, f"Entity {entity_id} not found"


class TestNoRainScenario:
    def test_no_rain_detected(self, ha_client):
        ha_client.set_mock_scenario("no_rain")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)  # let the coordinator fetch tiles + run detection

        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None
        assert state["state"] == "off", f"Expected off, got {state['state']}"

    def test_arrival_time_unknown(self, ha_client):
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None
        assert state["state"] in ("unknown", "unavailable", "None")


class TestRainApproachingScenario:
    def test_rain_approaching_detected(self, ha_client):
        """Rain cell moving east toward the location should trigger rain_incoming."""
        ha_client.set_mock_scenario("rain_approaching")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None
        assert state["state"] == "on", f"Expected on, got {state['state']}"

    def test_arrival_time_in_future(self, ha_client):
        """Approaching rain should have an arrival time set."""
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None
        assert state["state"] not in ("unknown", "unavailable", "None"), \
            f"Expected a timestamp, got {state['state']}"


class TestRainEverywhereScenario:
    def test_rain_detected_overhead(self, ha_client):
        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(5)

        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None
        assert state["state"] == "on", f"Expected on, got {state['state']}"

    def test_arrival_time_is_set(self, ha_client):
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None
        assert state["state"] not in ("unknown", "unavailable", "None"), \
            f"Expected a timestamp, got {state['state']}"


class TestSystemLogs:
    """Run last - checks HA system logs for unexpected warnings from our integration."""

    def test_no_unexpected_warnings_from_integration(self, ha_client):
        """Our integration should not produce WARNING or ERROR level logs."""
        log_text = ha_client.get_text("/api/error_log")

        our_lines = [
            line for line in log_text.split("\n")
            if "incoming_rain" in line
            and ("WARNING" in line or "ERROR" in line)
        ]

        # Exclude known/expected warnings we can't control
        expected_patterns = [
            "custom integration incoming_rain which has not been tested",  # HA standard warning
        ]

        unexpected = [
            line for line in our_lines
            if not any(pattern in line for pattern in expected_patterns)
        ]

        assert not unexpected, (
            f"Found {len(unexpected)} unexpected warning(s) from incoming_rain:\n"
            + "\n".join(unexpected)
        )
