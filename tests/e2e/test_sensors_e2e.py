"""
E2E tests for the incoming_rain integration.

These tests run against a real HA instance in Docker with a mock RainViewer
server. They verify the full pipeline: HTTP fetch -> tile decoding ->
detection -> sensor state updates.
"""
from __future__ import annotations

import time
from io import BytesIO

import numpy as np
import pytest
from PIL import Image


BINARY_SENSOR = "binary_sensor.incoming_rain_status"
ARRIVAL_SENSOR = "sensor.incoming_rain_arrival_time"
INTENSITY_SENSOR = "sensor.incoming_rain_intensity"
LAST_RAIN_SENSOR = "sensor.incoming_rain_last_rain"
IMAGE_64 = "image.incoming_rain_radar_64km"
IMAGE_128 = "image.incoming_rain_radar_128km"
IMAGE_256 = "image.incoming_rain_radar_256km"
IMAGE_512 = "image.incoming_rain_radar_512km"


class TestIntegrationLoaded:
    def test_binary_sensor_exists(self, ha_client):
        state = ha_client.get_state(BINARY_SENSOR)
        assert state is not None, f"Entity {BINARY_SENSOR} not found"
        assert state["state"] in ("on", "off", "unavailable")

    def test_arrival_sensor_exists(self, ha_client):
        state = ha_client.get_state(ARRIVAL_SENSOR)
        assert state is not None, f"Entity {ARRIVAL_SENSOR} not found"

    def test_intensity_sensor_exists(self, ha_client):
        state = ha_client.get_state(INTENSITY_SENSOR)
        assert state is not None, f"Entity {INTENSITY_SENSOR} not found"
        assert state["state"] in ("none", "light", "moderate", "heavy", "extreme", "unavailable")

    def test_last_rain_sensor_exists(self, ha_client):
        state = ha_client.get_state(LAST_RAIN_SENSOR)
        assert state is not None, f"Entity {LAST_RAIN_SENSOR} not found"

    @pytest.mark.parametrize("entity_id", [IMAGE_64, IMAGE_128, IMAGE_256, IMAGE_512])
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

    def test_intensity_shows_level_with_rain(self, ha_client):
        """When rain is everywhere, intensity should not be 'none'."""
        state = ha_client.get_state(INTENSITY_SENSOR)
        assert state is not None
        assert state["state"] != "none", \
            f"Expected a rain intensity level, got {state['state']}"
        assert state["state"] in ("light", "moderate", "heavy", "extreme"), \
            f"Unexpected intensity value: {state['state']}"


def _rain_fraction_from_gif(gif_bytes: bytes) -> float:
    """Analyze a radar GIF and return the fraction of rain-colored pixels.

    Rain pixels are identified by having a blue channel significantly higher
    than the red channel - the Voyager base map is predominantly beige/green/white,
    while precipitation overlay adds blue/cyan.
    """
    img = Image.open(BytesIO(gif_bytes))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    arr = np.array(img.convert("RGB"))

    blue_heavy = arr[:, :, 2] > arr[:, :, 0] + 30
    rain_pixel_count = int(blue_heavy.sum())
    total_pixels = arr.shape[0] * arr.shape[1]
    return rain_pixel_count / total_pixels


class TestIntegratedRainValidation:
    """Tests that tie sensor states to radar image content.

    If the binary sensor says rain, the image MUST show visible rain.
    If the binary sensor says dry, the image MUST NOT show rain.
    """

    def test_rain_scenario_sensors_and_images_consistent(self, ha_client):
        """When rain is present: sensor=on, image shows rain, intensity!=none."""
        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        # Check sensor state
        sensor_state = ha_client.get_state(BINARY_SENSOR)
        assert sensor_state["state"] == "on", (
            f"Sensor says {sensor_state['state']} but rain scenario is active"
        )

        # Check intensity is not "none"
        intensity = ha_client.get_state(INTENSITY_SENSOR)
        assert intensity["state"] != "none", (
            f"Intensity is {intensity['state']} but rain scenario is active"
        )

        # Download the actual GIF and verify it contains visible precipitation
        gif_bytes = ha_client.get_image(IMAGE_128)
        assert gif_bytes and len(gif_bytes) > 1000, "GIF is empty or too small"

        rain_fraction = _rain_fraction_from_gif(gif_bytes)

        assert rain_fraction > 0.02, (
            f"Rain scenario active, sensor={sensor_state['state']}, "
            f"but GIF has only {rain_fraction:.4f} rain-colored pixels. "
            f"Rain is not visible in the rendered image."
        )

    def test_no_rain_scenario_sensors_and_images_consistent(self, ha_client):
        """When no rain: sensor=off, image shows clean map, intensity=none."""
        ha_client.set_mock_scenario("no_rain")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        sensor_state = ha_client.get_state(BINARY_SENSOR)
        assert sensor_state["state"] == "off", (
            f"Sensor says {sensor_state['state']} but no-rain scenario"
        )

        intensity = ha_client.get_state(INTENSITY_SENSOR)
        assert intensity["state"] == "none", (
            f"Intensity is {intensity['state']} but no-rain scenario"
        )

        gif_bytes = ha_client.get_image(IMAGE_128)
        assert gif_bytes and len(gif_bytes) > 1000, "GIF is empty or too small"

        rain_fraction = _rain_fraction_from_gif(gif_bytes)

        assert rain_fraction < 0.01, (
            f"No-rain scenario but GIF has {rain_fraction:.4f} rain-colored pixels. "
            f"Image should be clean."
        )

    def test_approaching_rain_sensors_and_images_consistent(self, ha_client):
        """Rain approaching: sensor=on, image shows rain cells."""
        ha_client.set_mock_scenario("rain_approaching")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        sensor_state = ha_client.get_state(BINARY_SENSOR)
        assert sensor_state["state"] == "on", (
            f"Approaching rain but sensor says {sensor_state['state']}"
        )

        arrival = ha_client.get_state(ARRIVAL_SENSOR)
        assert arrival["state"] not in ("unknown", "unavailable"), (
            f"Approaching rain but arrival time is {arrival['state']}"
        )

        # The image should show rain cells (approaching from west in mock)
        gif_bytes = ha_client.get_image(IMAGE_128)
        assert gif_bytes and len(gif_bytes) > 1000, "GIF is empty or too small"

        rain_fraction = _rain_fraction_from_gif(gif_bytes)

        assert rain_fraction > 0.005, (
            f"Approaching rain scenario, sensor={sensor_state['state']}, "
            f"but GIF has only {rain_fraction:.4f} rain-colored pixels."
        )


class TestValidationCanFail:
    """Meta-tests: prove our validation tests would catch real bugs.

    Feed wrong scenario data and verify the assertions WOULD fail.
    """

    def test_rain_assertions_fail_on_no_rain_data(self, ha_client):
        """If we check for rain but the scenario is no_rain, assertions should fail."""
        ha_client.set_mock_scenario("no_rain")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        sensor_state = ha_client.get_state(BINARY_SENSOR)
        # This SHOULD be "off" - if it's "on", our mock is broken
        assert sensor_state["state"] == "off", (
            "Mock scenario broken - no_rain should produce off"
        )

        # Now verify that the rain assertions would FAIL on this data
        gif_bytes = ha_client.get_image(IMAGE_128)
        assert gif_bytes and len(gif_bytes) > 1000, "GIF is empty or too small"

        rain_fraction = _rain_fraction_from_gif(gif_bytes)

        # The rain threshold from our positive test is 0.02
        # With no-rain data, this MUST be below that threshold
        assert rain_fraction < 0.02, (
            f"No-rain data has {rain_fraction:.4f} rain fraction - "
            f"our rain detection threshold (0.02) would NOT catch a missing-rain bug!"
        )

    def test_no_rain_assertions_fail_on_rain_data(self, ha_client):
        """If we check for no-rain but scenario is rain_everywhere, assertions should fail."""
        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.update_entity(BINARY_SENSOR)
        time.sleep(15)

        sensor_state = ha_client.get_state(BINARY_SENSOR)
        assert sensor_state["state"] == "on", (
            "Mock scenario broken - rain should produce on"
        )

        gif_bytes = ha_client.get_image(IMAGE_128)
        assert gif_bytes and len(gif_bytes) > 1000, "GIF is empty or too small"

        rain_fraction = _rain_fraction_from_gif(gif_bytes)

        # The no-rain threshold from our negative test is 0.01
        # With rain data, this MUST be above that threshold
        assert rain_fraction > 0.01, (
            f"Rain data has only {rain_fraction:.4f} rain fraction - "
            f"our no-rain threshold (0.01) would NOT catch a visible-rain bug!"
        )


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
