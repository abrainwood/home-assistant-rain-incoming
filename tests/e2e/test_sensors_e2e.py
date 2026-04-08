"""
E2E tests for the incoming_rain integration.

These tests run against a real HA instance in Docker with a mock RainViewer
server. They verify the full pipeline: HTTP fetch -> tile decoding ->
detection -> sensor state updates.
"""
from __future__ import annotations

import time

import pytest


BINARY_SENSOR = "binary_sensor.incoming_rain_rain_incoming"
ARRIVAL_SENSOR = "sensor.incoming_rain_rain_arrival_time"


class TestIntegrationLoaded:
    def test_binary_sensor_exists(self, ha_client):
        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None, f"Entity {BINARY_SENSOR} not found"
        assert state["state"] in ("on", "off", "unavailable")

    def test_arrival_sensor_exists(self, ha_client):
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None, f"Entity {ARRIVAL_SENSOR} not found"


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
