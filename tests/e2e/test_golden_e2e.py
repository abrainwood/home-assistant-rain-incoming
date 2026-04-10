"""
E2E tests using golden data through the full HA pipeline.

These tests configure the integration for real locations (Bright, Albury,
Shepparton) and serve golden data tile images through the mock server.
They validate that the full pipeline - tile fetch, colour decoding,
detection, sensor state - produces correct results for real-world data.
"""
from __future__ import annotations

import time
from io import BytesIO

import numpy as np
from PIL import Image


def _images_differ_significantly(
    rain_gif_bytes: bytes, no_rain_gif_bytes: bytes, threshold: float = 0.01
) -> tuple[bool, float]:
    """Compare two GIFs and return whether they differ significantly."""
    def last_frame_array(gif_bytes):
        img = Image.open(BytesIO(gif_bytes))
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(img.n_frames - 1)
        return np.array(img.convert("RGB")).astype(np.float32)

    rain_arr = last_frame_array(rain_gif_bytes)
    clean_arr = last_frame_array(no_rain_gif_bytes)

    diff = np.abs(rain_arr - clean_arr)
    pixel_differs = diff.max(axis=2) > 20
    fraction = pixel_differs.sum() / (rain_arr.shape[0] * rain_arr.shape[1])

    return fraction > threshold, float(fraction)


def _save_gif(data: bytes, path: str) -> None:
    """Save GIF bytes to a file for manual inspection."""
    with open(path, "wb") as f:
        f.write(data)


# Bright golden data: all frames 0-12 classified rain_overhead or rain_incoming
# Coordinates: -36.731, 146.960
BRIGHT_LAT = -36.731
BRIGHT_LON = 146.960

# Entity IDs for location-specific integrations
# HA generates entity IDs from device name "Rain Incoming - Bright"
# -> binary_sensor.rain_incoming_bright_status
# -> image.rain_incoming_bright_radar_128km
# etc.


class TestGoldenDataBright:
    """E2E test using Bright golden data - heavy rain throughout."""

    def test_rain_detected_with_golden_data(self, ha_client, configure_location):
        """Bright frames 0-12 are all rain_overhead/incoming - sensor must be on.

        Steps:
        1. Set mock to serve Bright golden data
        2. Configure integration for Bright's coordinates
        3. Wait for coordinator to poll, fetch tiles, and detect
        4. Assert sensor is 'on'
        5. Download radar image and save for inspection
        6. Switch to no_rain and verify sensor goes 'off'
        7. Compare images to verify rain was visible
        """
        # Step 1: Load Bright golden data (last 8 frames, since coordinator fetches 8)
        ha_client.set_mock_scenario("golden:bright:5-12")

        # Step 2: Configure Bright integration
        configure_location(BRIGHT_LAT, BRIGHT_LON, "Bright")

        # Step 3: Wait for coordinator
        time.sleep(15)

        # Step 4: Check sensor state
        sensor_id = "binary_sensor.rain_incoming_bright_status"
        state = ha_client.get_state(sensor_id)
        assert state is not None, (
            f"Entity {sensor_id} not found - integration may not have loaded"
        )

        # Step 5: Download rain image
        image_id = "image.rain_incoming_bright_radar_128km"
        rain_gif = ha_client.get_image(image_id)
        if rain_gif:
            _save_gif(rain_gif, "/tmp/e2e_golden_bright_rain.gif")
            _save_gif(rain_gif, "/tmp/golden_bright_rain.gif")

        # Record rain state for later assertion
        rain_state = state["state"]
        rain_attrs = state.get("attributes", {})

        # Step 6: Switch to no_rain and trigger update
        ha_client.set_mock_scenario("no_rain")
        ha_client.update_entity(sensor_id)
        time.sleep(15)

        no_rain_state = ha_client.get_state(sensor_id)
        no_rain_gif = ha_client.get_image(image_id)
        if no_rain_gif:
            _save_gif(no_rain_gif, "/tmp/e2e_golden_bright_no_rain.gif")
            _save_gif(no_rain_gif, "/tmp/golden_bright_norain.gif")

        # Step 7: Assertions
        assert rain_state == "on", (
            f"Bright golden data (all rain_overhead) but sensor says '{rain_state}'. "
            f"Attributes: {rain_attrs}"
        )

        assert no_rain_state is not None
        assert no_rain_state["state"] == "off", (
            f"After switching to no_rain, sensor should be 'off' "
            f"but got '{no_rain_state['state']}'"
        )

        # Image comparison - rain vs no-rain must look different
        if rain_gif and no_rain_gif:
            differs, fraction = _images_differ_significantly(rain_gif, no_rain_gif)
            assert differs, (
                f"Bright rain_overhead: sensor={rain_state} but image identical "
                f"to no-rain baseline (diff fraction: {fraction:.4f}). "
                f"GIFs saved to /tmp/e2e_golden_bright_*.gif"
            )

        # Also verify intensity is reported correctly
        intensity_id = "sensor.rain_incoming_bright_intensity"
        intensity_state = ha_client.get_state(intensity_id)
        if intensity_state is not None:
            # Restore golden scenario to check intensity in rain state
            ha_client.set_mock_scenario("golden:bright:5-12")
            ha_client.update_entity(sensor_id)
            time.sleep(15)
            intensity_state = ha_client.get_state(intensity_id)
            assert intensity_state["state"] != "none", (
                f"Bright golden data should have non-none intensity, "
                f"got '{intensity_state['state']}'"
            )
            assert intensity_state["state"] in (
                "light", "moderate", "heavy", "extreme",
            ), f"Unexpected intensity: '{intensity_state['state']}'"


# Albury golden data: frames 0-1 rain_receeding, 2-3 none, 4+ rain_incoming
# Coordinates: -36.081, 146.916
ALBURY_LAT = -36.081
ALBURY_LON = 146.916


class TestGoldenDataAlbury:
    """E2E test using Albury golden data - mixed classifications."""

    def test_rain_incoming_detected_with_later_frames(self, ha_client, configure_location):
        """Albury frames 5-12 show rain_incoming/overhead - sensor should be on."""
        ha_client.set_mock_scenario("golden:albury:5-12")
        configure_location(ALBURY_LAT, ALBURY_LON, "Albury")
        time.sleep(15)

        sensor_id = "binary_sensor.rain_incoming_albury_status"
        state = ha_client.get_state(sensor_id)
        assert state is not None, f"Entity {sensor_id} not found"

        rain_gif = ha_client.get_image("image.rain_incoming_albury_radar_128km")
        if rain_gif:
            _save_gif(rain_gif, "/tmp/e2e_golden_albury_rain.gif")

        assert state["state"] == "on", (
            f"Albury frames 5-12 (rain_incoming/overhead) but sensor says "
            f"'{state['state']}'. Attributes: {state.get('attributes', {})}"
        )


# Shepparton golden data: all frames rain_overhead or rain_incoming
# Coordinates: -36.381, 145.399
SHEPPARTON_LAT = -36.381
SHEPPARTON_LON = 145.399


class TestGoldenDataShepparton:
    """E2E test using Shepparton golden data - consistent rain overhead.

    NOTE: The mock server serves the same golden tile for ALL tile positions.
    This creates a tiled repetition in the stitched grid. The detector may
    not detect rain if the tiled pattern doesn't form trackable cells.
    This is a known limitation of the E2E golden data approach.
    """

    def test_rain_overhead_detected(self, ha_client, configure_location):
        """Shepparton frames 0-12 are all rain_overhead - sensor should be on.

        KNOWN BUG: tiled golden data doesn't form trackable cells for
        Shepparton. This test fails honestly - see BACKLOG.md for tracking.
        The test must remain and must fail visibly until the pipeline is fixed.
        """
        ha_client.set_mock_scenario("golden:shepparton:5-12")
        configure_location(SHEPPARTON_LAT, SHEPPARTON_LON, "Shepparton")
        time.sleep(15)

        sensor_id = "binary_sensor.rain_incoming_shepparton_status"
        state = ha_client.get_state(sensor_id)
        assert state is not None, f"Entity {sensor_id} not found"

        rain_gif = ha_client.get_image("image.rain_incoming_shepparton_radar_128km")
        if rain_gif:
            _save_gif(rain_gif, "/tmp/e2e_golden_shepparton_rain.gif")

        assert state["state"] == "on", (
            f"Shepparton golden data (all rain_overhead) but sensor says "
            f"'{state['state']}'. Known bug: tiled golden data doesn't form "
            f"trackable cells for this location - see BACKLOG.md. "
            f"Attributes: {state.get('attributes', {})}"
        )

    def test_no_rain_data_correctly_shows_dry(self, ha_client, configure_location):
        """Verify test catches missing rain: no-rain data must produce sensor=off."""
        ha_client.set_mock_scenario("no_rain")
        configure_location(SHEPPARTON_LAT, SHEPPARTON_LON, "Shepparton")
        time.sleep(15)

        sensor_id = "binary_sensor.rain_incoming_shepparton_status"
        state = ha_client.get_state(sensor_id)
        assert state is not None, f"Entity {sensor_id} not found"
        assert state["state"] == "off", (
            f"No-rain scenario should produce off but got '{state['state']}'"
        )
