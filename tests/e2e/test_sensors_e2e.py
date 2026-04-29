"""
E2E tests for the rain_incoming integration.

These tests run against a real HA instance in Docker with a mock RainViewer
server. They verify the full pipeline: HTTP fetch -> tile decoding ->
detection -> sensor state updates.

Note: entity existence and basic sensor-state assertions are covered at the
integration layer (tests/integration/test_sensors.py). The classes here
focus on what only E2E can verify: radar image content, sensor/image
coherence. Warning/error log cleanliness is covered at the integration
layer (tests/integration/test_no_unexpected_warnings.py).
"""
from __future__ import annotations

import os
import time

from tests.e2e.image_helpers import gif_has_precipitation_pixels, images_differ_significantly


BINARY_SENSOR = "binary_sensor.rain_incoming_imminent"
ARRIVAL_SENSOR = "sensor.rain_incoming_arrival_time"
INTENSITY_SENSOR = "sensor.rain_incoming_intensity"
IMAGE_128 = "image.rain_incoming_radar_128km"

# Timeout for polling until a rendered image changes after a scenario switch.
# The render pipeline in CI (coordinator cycle + tile fetches + composite_frames
# + GIF encoding) can take 60s+ on a shared runner under load. 90s gives
# headroom without being so long that a genuine failure takes forever to report.
_IMAGE_POLL_TIMEOUT = 90


class TestIntegratedRainValidation:
    """Tests that tie sensor states to radar image content.

    Uses differential comparison: download a no-rain baseline and a rain GIF,
    then compare pixel-by-pixel. The old heuristic (blue > red + 30) matched
    the ocean on the Voyager base map and could not distinguish rain from map.
    """

    def _download_gif(self, ha_client, scenario, entity_id=IMAGE_128, previous_gif=None):
        """Set scenario, wait for coordinator cycle, then download GIF.

        Captures the pre-switch image as baseline before changing scenario.
        Polls until:
        - The image changes at the byte level (new render completed), OR
        - The image is stable for several polls (render was skipped because frame
          content didn't change - same scenario re-applied or frames unchanged).

        Using images_differ_significantly as the exit condition is wrong here:
        the threshold (5%) is too high to catch a timestamp-only re-render (a new
        coordinator cycle produces the same tiles but with a shifted frame window,
        changing only the small timestamp text overlay). Byte-level != catches all
        real changes including timestamp-only differences.
        """
        # Capture current image BEFORE switching so we can detect a fresh render,
        # not just return whatever was cached from the previous test's scenario.
        baseline = previous_gif if previous_gif is not None else ha_client.get_image(entity_id)

        ha_client.set_mock_scenario(scenario)
        ha_client.wait_for_coordinator_cycle(BINARY_SENSOR, timeout=60)
        deadline = time.time() + _IMAGE_POLL_TIMEOUT
        stable_polls = 0  # consecutive polls that returned same bytes as baseline
        while time.time() < deadline:
            gif = ha_client.get_image(entity_id)
            if gif and len(gif) > 1000:
                if baseline is None:
                    return gif
                if gif != baseline:
                    return gif  # render completed and produced new content
                stable_polls += 1
                if stable_polls >= 5:
                    # Image has been stable for ~10s - render either completed
                    # (same-scenario, frame skipped) or is truly unchanged. Accept.
                    return gif
            else:
                stable_polls = 0
            time.sleep(2)
        raise AssertionError(
            f"Timed out after {_IMAGE_POLL_TIMEOUT}s waiting for '{scenario}' "
            f"image to be ready after scenario switch. "
            f"Last image: {len(gif) if gif else 0} bytes. "
            f"Background render may not have completed."
        )

    def test_rain_is_visible_on_radar_image(self, ha_client):
        """The radar image MUST look different with rain vs without rain."""
        no_rain_gif = self._download_gif(ha_client, "no_rain")
        rain_gif = self._download_gif(ha_client, "rain_everywhere", previous_gif=no_rain_gif)

        assert no_rain_gif and rain_gif, "Failed to download GIFs"

        differs, fraction = images_differ_significantly(rain_gif, no_rain_gif)

        # Save GIFs for manual inspection
        if os.environ.get("RAIN_TEST_DEBUG_DUMP"):
            with open("/tmp/e2e_rain.gif", "wb") as f:
                f.write(rain_gif)
            with open("/tmp/e2e_no_rain.gif", "wb") as f:
                f.write(no_rain_gif)

        assert differs, (
            f"Rain GIF is identical to no-rain GIF (diff fraction: {fraction:.4f}). "
            f"Rain is NOT visible in the rendered radar image."
        )

    def test_rain_sensors_match_image_when_raining(self, ha_client):
        """When rain scenario is active: sensor=on, intensity!=none, image differs from baseline."""
        # Get baseline first
        no_rain_gif = self._download_gif(ha_client, "no_rain")

        # Set rain scenario and wait for a fresh coordinator cycle under the new scenario
        ha_client.set_mock_scenario("rain_everywhere")
        sensor = ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)

        # Check all sensors agree it's raining
        assert sensor["state"] == "on", f"Rain active but sensor says {sensor['state']}"

        intensity = ha_client.get_state(INTENSITY_SENSOR)
        assert intensity["state"] != "none", f"Rain active but intensity is {intensity['state']}"

        # Wait for the image render to reflect the new scenario.
        # Poll until the image has actual precipitation pixels - byte-level
        # inequality isn't sufficient because the placeholder GIF satisfies
        # it while having zero precipitation content.
        rain_gif = None
        deadline = time.time() + _IMAGE_POLL_TIMEOUT
        while time.time() < deadline:
            candidate = ha_client.get_image(IMAGE_128)
            if candidate and len(candidate) > 1000:
                has_precip, _ = gif_has_precipitation_pixels(candidate)
                if has_precip:
                    rain_gif = candidate
                    break
            time.sleep(2)

        assert rain_gif is not None, "Rain image never changed from baseline after poll timeout"
        differs, fraction = images_differ_significantly(rain_gif, no_rain_gif)
        assert differs, (
            f"Sensors say rain (state={sensor['state']}, intensity={intensity['state']}) "
            f"but radar image is identical to no-rain baseline (diff: {fraction:.4f})"
        )

    def test_no_rain_sensors_match_image_when_dry(self, ha_client):
        """When no-rain: sensor=off, intensity=none."""
        ha_client.set_mock_scenario("no_rain")
        sensor = ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)

        assert sensor["state"] == "off", f"No rain but sensor says {sensor['state']}"

        intensity = ha_client.get_state(INTENSITY_SENSOR)
        assert intensity["state"] == "none", f"No rain but intensity is {intensity['state']}"

    def test_approaching_rain_sensors_match_image(self, ha_client):
        """Approaching rain: sensor=on, arrival_time set, image differs from baseline."""
        no_rain_gif = self._download_gif(ha_client, "no_rain")

        ha_client.set_mock_scenario("rain_approaching")
        sensor = ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)

        assert sensor["state"] == "on", f"Approaching rain but sensor says {sensor['state']}"

        arrival = ha_client.get_state(ARRIVAL_SENSOR)
        assert arrival["state"] not in ("unknown", "unavailable"), (
            f"Approaching rain but arrival is {arrival['state']}"
        )

        # Wait for the image render to reflect the new scenario.
        rain_gif = None
        deadline = time.time() + _IMAGE_POLL_TIMEOUT
        while time.time() < deadline:
            candidate = ha_client.get_image(IMAGE_128)
            if candidate and candidate != no_rain_gif:
                rain_gif = candidate
                break
            time.sleep(2)

        assert rain_gif is not None, "Approaching rain image never changed from baseline after poll timeout"
        differs, fraction = images_differ_significantly(rain_gif, no_rain_gif)
        assert differs, (
            f"Sensors say approaching rain but image identical to baseline (diff: {fraction:.4f})"
        )


class TestValidationCanFail:
    """Meta-tests: prove our validation would catch bugs."""

    def test_different_scenarios_produce_different_images(self, ha_client):
        """rain_everywhere and no_rain MUST produce visually different GIFs."""
        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)
        # Poll until image has actual precipitation pixels (placeholder satisfies
        # len > 1000 but has no precipitation content, which would fool the later
        # images_differ_significantly check).
        rain_gif = None
        deadline = time.time() + _IMAGE_POLL_TIMEOUT
        while time.time() < deadline:
            candidate = ha_client.get_image(IMAGE_128)
            if candidate and len(candidate) > 1000:
                has_precip, _ = gif_has_precipitation_pixels(candidate)
                if has_precip:
                    rain_gif = candidate
                    break
            time.sleep(2)
        if rain_gif is None:
            rain_gif = ha_client.get_image(IMAGE_128)

        ha_client.set_mock_scenario("no_rain")
        ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)
        # Poll until image changes from the rain version
        clean_gif = None
        deadline = time.time() + _IMAGE_POLL_TIMEOUT
        while time.time() < deadline:
            candidate = ha_client.get_image(IMAGE_128)
            if candidate and candidate != rain_gif:
                clean_gif = candidate
                break
            time.sleep(2)
        if clean_gif is None:
            clean_gif = ha_client.get_image(IMAGE_128)

        # Rain vs clean must differ
        differs_rain, frac_rain = images_differ_significantly(rain_gif, clean_gif)
        assert differs_rain, (
            f"rain_everywhere and no_rain produce identical images (diff: {frac_rain:.4f}). "
            f"The image comparison cannot detect rain."
        )

        # Same scenario fetched again should produce similar images
        clean_gif2 = ha_client.get_image(IMAGE_128)
        differs_clean, frac_clean = images_differ_significantly(clean_gif, clean_gif2)
        assert frac_clean < 0.05, (
            f"Two no_rain images differ by {frac_clean:.4f} - too much noise in comparison"
        )

    def test_inverse_dry_scenario_must_not_detect_rain(self, ha_client):
        """no_rain scenario must produce state='off'.

        Proves the dry assertion can fail: if the pipeline were broken
        and always returned 'on', this test would catch it.
        """
        ha_client.set_mock_scenario("no_rain")
        sensor = ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)

        assert sensor["state"] == "off", (
            f"Synthetic no_rain scenario but sensor says '{sensor['state']}' - "
            f"the pipeline is detecting rain in transparent tiles"
        )

    def test_inverse_rain_scenario_must_not_be_dry(self, ha_client):
        """rain_everywhere scenario must produce state='on'.

        Proves the rain assertion can fail: if the pipeline were broken
        and always returned 'off', this test would catch it.
        """
        ha_client.set_mock_scenario("rain_everywhere")
        sensor = ha_client.wait_for_coordinator_cycle(BINARY_SENSOR)

        assert sensor["state"] == "on", (
            f"Synthetic rain_everywhere scenario but sensor says '{sensor['state']}' - "
            f"the pipeline is not detecting rain in solid rain tiles"
        )
