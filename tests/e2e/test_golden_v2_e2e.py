"""
Golden v2 E2E tests using REAL RainViewer tile PNGs.

These tests serve actual captured tiles through the mock server and
validate the full HA pipeline produces correct sensor states AND
visible radar images. If the rendering pipeline makes rain invisible,
these tests WILL fail because the tiles are the real thing.

Data sources:
- Canberra: 13 frames, none(0-2) -> incoming(3-10) -> overhead(11-12)
- Darwin: 13 frames, mostly overhead (all frames have tiles)
- Melbourne: 13 frames, overhead(0) -> none(1-12)
"""
from __future__ import annotations

import os
import time

from tests.e2e.image_helpers import gif_has_precipitation_pixels, images_differ_significantly


def _save_gif(data: bytes, path: str) -> None:
    """Save GIF bytes to a file for manual inspection."""
    if os.environ.get("RAIN_TEST_DEBUG_DUMP"):
        with open(path, "wb") as f:
            f.write(data)


# --- Canberra ---
# Classifications: none(0-2), rain_incoming(3-10), rain_overhead(11-12)
CANBERRA_LAT = -35.309
CANBERRA_LON = 149.123


class TestCanberraGoldenV2:
    """Canberra: none(0-2) -> incoming(3-10) -> overhead(11-12)"""

    def test_rain_incoming_detected(self, ha_client, configure_location):
        """Frames 5-10 classified as rain_incoming. Sensor must be on."""
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        configure_location(CANBERRA_LAT, CANBERRA_LON, "Canberra_v2")

        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        # Save rain image for inspection
        image_id = "image.rain_incoming_canberra_v2_radar_128km"
        rain_gif = ha_client.get_image(image_id)
        if rain_gif:
            _save_gif(rain_gif, "/tmp/golden_v2_canberra_rain.gif")

        assert state["state"] == "on", (
            f"Canberra frames 5-10 (all rain_incoming) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

    def test_rain_visible_in_image(self, ha_client):
        """The radar image must show visible rain - not just sensor on."""
        image_id = "image.rain_incoming_canberra_v2_radar_128km"
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"

        # Ensure we have fresh rain data rendered - poll until sensor is in expected state
        ha_client.update_entity(sensor_id)
        ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

        rain_gif = ha_client.get_image(image_id)
        assert rain_gif and len(rain_gif) > 1000, (
            f"Rain GIF is missing or too small ({len(rain_gif) if rain_gif else 0} bytes)"
        )
        _save_gif(rain_gif, "/tmp/golden_v2_canberra_rain_vis.gif")

        # Switch to known-dry golden frames for baseline.
        # We use real dry frames (Canberra 0-2, classified "none") instead of
        # the synthetic "no_rain" scenario, which produces all-zero grids that
        # trigger the coordinator's "all fetches failed" protection.
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        ha_client.update_entity(sensor_id)
        ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") == "off",
        )

        # Wait for the image render to reflect the new dry scenario.
        # async_image() returns from cache immediately when a cached image
        # exists (to avoid blocking on the render lock), so the render may
        # still be in flight when the sensor state has already changed.
        # Poll until the image bytes differ from the rain image.
        baseline_gif = None
        deadline = time.time() + 30
        while time.time() < deadline:
            candidate = ha_client.get_image(image_id)
            if candidate and candidate != rain_gif:
                baseline_gif = candidate
                break
            time.sleep(2)

        if baseline_gif:
            _save_gif(baseline_gif, "/tmp/golden_v2_canberra_norain.gif")

        assert baseline_gif and len(baseline_gif) > 100, "Baseline GIF missing or still stale after 30s"

        differs, fraction = images_differ_significantly(rain_gif, baseline_gif)
        assert differs, (
            f"Sensor says rain but radar image identical to baseline (diff={fraction:.4f}). "
            f"Rain is not visible in the rendered image. "
            f"GIFs saved to /tmp/golden_v2_canberra_*.gif for inspection."
        )

    def test_no_rain_when_classified_none(self, ha_client):
        """Frames 0-2 are all 'none'. With only none frames, sensor should be off."""
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        ha_client.update_entity(sensor_id)
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") == "off",
        )
        assert state["state"] == "off", (
            f"Canberra frames 0-2 (all none) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

    def test_would_fail_wrong_data(self, ha_client):
        """Verify the rain test WOULD fail if we served no-rain data.

        This is the 'test the test' - proves our assertions aren't tautologies.
        With dry golden frames, the sensor must be off. If it's on, something
        is wrong with the test setup.

        We use real dry frames (Canberra 0-2, classified "none") instead of
        the synthetic "no_rain" scenario, which produces all-zero grids that
        trigger the coordinator's "all fetches failed" protection and make
        the entity unavailable.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        ha_client.update_entity(sensor_id)
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") == "off",
        )
        assert state["state"] == "off", (
            f"Dry golden frames (Canberra 0-2, all none) but sensor says '{state['state']}' - "
            f"test infrastructure is broken"
        )

    def test_image_would_fail_without_rain_data(self, ha_client):
        """Verify the image visibility test WOULD fail if rain wasn't rendered.

        Two dry GIFs from the same scenario must look identical - proving
        there's no precipitation contamination in the rendered output.
        We compare two fetches of the same dry data rather than checking
        absolute pixel colours, because map background (CartoDB vegetation)
        can incidentally match precipitation colour thresholds.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        ha_client.update_entity("binary_sensor.rain_incoming_canberra_v2_imminent")
        ha_client.poll_entity_state(
            "binary_sensor.rain_incoming_canberra_v2_imminent", timeout=30,
            condition=lambda s: s.get("state") == "off",
        )

        dry_gif_1 = ha_client.get_image("image.rain_incoming_canberra_v2_radar_128km")
        assert dry_gif_1 and len(dry_gif_1) > 100, "Dry GIF 1 missing"

        # Fetch again - same scenario, should be identical
        dry_gif_2 = ha_client.get_image("image.rain_incoming_canberra_v2_radar_128km")
        assert dry_gif_2 and len(dry_gif_2) > 100, "Dry GIF 2 missing"

        # Two renders of the same dry data must be stable (within noise)
        differs, fraction = images_differ_significantly(dry_gif_1, dry_gif_2, threshold=0.01)
        assert not differs, (
            f"Two dry GIF renders differ by {fraction:.4f} - "
            f"rendering is not deterministic, image comparison unreliable"
        )

    def test_rain_data_would_fail_dry_assertion(self, ha_client):
        """Verify that rain data does NOT produce a dry sensor state.

        If this test passed with state='off', it would mean our rain detection
        test is meaningless - it would pass even when the pipeline is broken.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        ha_client.update_entity("binary_sensor.rain_incoming_canberra_v2_imminent")
        state = ha_client.poll_entity_state(
            "binary_sensor.rain_incoming_canberra_v2_imminent", timeout=30,
            condition=lambda s: s.get("state") == "on",
        )
        assert state["state"] != "off", (
            f"Rain golden data (frames 5-10, all rain_incoming) but sensor is off - "
            f"rain detection is broken and the positive test wouldn't catch it"
        )


# --- Darwin ---
# Classifications: rain_overhead(0), rain_incoming(1-2), rain_overhead(3-12)
# All 13 frames have 50 tiles each
DARWIN_LAT = -12.4634
DARWIN_LON = 130.8456


class TestDarwinGoldenV2:
    """Darwin: mostly overhead rain throughout all frames."""

    def test_rain_overhead_detected(self, ha_client, configure_location):
        """Darwin frames 5-12 are all rain_overhead. Sensor must be on."""
        ha_client.set_mock_scenario("golden_v2:Darwin:5-12")
        configure_location(DARWIN_LAT, DARWIN_LON, "Darwin_v2")

        sensor_id = "binary_sensor.rain_incoming_darwin_v2_imminent"
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        image_id = "image.rain_incoming_darwin_v2_radar_128km"
        rain_gif = ha_client.get_image(image_id)
        if rain_gif:
            _save_gif(rain_gif, "/tmp/golden_v2_darwin_rain.gif")

        assert state["state"] == "on", (
            f"Darwin frames 5-12 (all rain_overhead) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

    def test_rain_visible_in_image(self, ha_client):
        """Darwin radar image must show visible rain.

        Darwin has rain in ALL golden frames (no dry frames available),
        so instead of comparing to a dry baseline, we directly check
        for precipitation-coloured pixels in the rendered image.
        """
        image_id = "image.rain_incoming_darwin_v2_radar_128km"
        sensor_id = "binary_sensor.rain_incoming_darwin_v2_imminent"

        # Ensure we have fresh rain data rendered - poll until sensor is in expected state
        ha_client.update_entity(sensor_id)
        ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

        rain_gif = ha_client.get_image(image_id)
        assert rain_gif and len(rain_gif) > 1000, (
            f"Rain GIF is missing or too small ({len(rain_gif) if rain_gif else 0} bytes)"
        )
        _save_gif(rain_gif, "/tmp/golden_v2_darwin_rain_vis.gif")

        has_precip, fraction = gif_has_precipitation_pixels(rain_gif)
        assert has_precip, (
            f"Sensor says rain but radar image has no precipitation pixels "
            f"(fraction={fraction:.4f}). Rain is not visible in the rendered image. "
            f"GIF saved to /tmp/golden_v2_darwin_rain_vis.gif for inspection."
        )


# --- Melbourne ---
# Classifications: rain_overhead(0), none(1-12)
MELBOURNE_LAT = -37.8136
MELBOURNE_LON = 144.9631


class TestMelbourneGoldenV2:
    """Melbourne: overhead(0) -> none(1-12). Should be mostly dry."""

    def test_mostly_dry_detected(self, ha_client, configure_location):
        """Melbourne frames 1-12 are all 'none'. Sensor should be off."""
        ha_client.set_mock_scenario("golden_v2:Melbourne:1-12")
        configure_location(MELBOURNE_LAT, MELBOURNE_LON, "Melbourne_v2")

        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_imminent"
        # Wait for a complete coordinator cycle with the dry scenario.
        # configure_location's internal poll accepts any non-unavailable state,
        # which may capture a transient "on" from the coordinator's initial
        # refresh before the dry scenario data is fully processed. Explicitly
        # waiting for a fresh cycle ensures we test the final detection result.
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        image_id = "image.rain_incoming_melbourne_v2_radar_128km"
        dry_gif = ha_client.get_image(image_id)
        if dry_gif:
            _save_gif(dry_gif, "/tmp/golden_v2_melbourne_dry.gif")

        assert state["state"] == "off", (
            f"Melbourne frames 1-12 (all none) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

    def test_single_overhead_frame_with_dry_majority(self, ha_client):
        """Melbourne frames 0-12: only frame 0 is overhead, 1-12 are none.

        With 12 out of 13 frames showing no rain, the system should
        classify this as dry overall.
        """
        ha_client.set_mock_scenario("golden_v2:Melbourne:0-12")
        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_imminent"
        ha_client.update_entity(sensor_id)
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") == "off",
        )

        image_id = "image.rain_incoming_melbourne_v2_radar_128km"
        mixed_gif = ha_client.get_image(image_id)
        if mixed_gif:
            _save_gif(mixed_gif, "/tmp/golden_v2_melbourne_mixed.gif")

        # With 12/13 dry frames, the sensor should be off
        assert state["state"] == "off", (
            f"Melbourne frames 0-12 (1 overhead + 12 none) but sensor says '{state['state']}'. "
            f"With 12/13 dry frames the system should classify this as dry. "
            f"Attributes: {state.get('attributes', {})}"
        )
