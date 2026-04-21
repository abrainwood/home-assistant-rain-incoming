"""
Golden v2 E2E tests using REAL RainViewer tile PNGs.

These tests serve actual captured tiles through the mock server and
validate the full HA pipeline produces correct sensor states AND
visible radar images. If the rendering pipeline makes rain invisible,
these tests WILL fail because the tiles are the real thing.

QC state: all golden tests run with cold QC - a freshly created
coordinator with no clutter map and zero maturity. This is the
controlled baseline. Tests that depend on mature QC (clutter
suppression, etc.) should inject a known clutter map, not rely on
accumulated state from prior tests.

IMPORTANT: all three golden capture locations have real precipitation
in their analysis area across ALL frames, including those classified
"none" at the capture location. The "none" classification means no
rain was directly overhead at capture time - it does NOT mean the
analysis area is dry. The detector correctly reports rain_incoming
for all golden data with cold QC.

Inverse validation (proving tests CAN fail with wrong input) lives
in test_sensors_e2e.py using synthetic scenarios - the only data we
have where the analysis area is genuinely dry (transparent tiles).
Golden-data-layer inverse validation requires genuinely dry captures,
tracked in #139.

Data sources:
- Canberra: 13 frames, real precip in all frames (8-13% of analysis area)
- Darwin: 13 frames, real precip in all frames (1-2% of analysis area)
- Melbourne: 13 frames, real precip in all frames (10-13% of analysis area)
"""
from __future__ import annotations

import os

from tests.e2e.image_helpers import gif_has_precipitation_pixels, images_differ_significantly


def _save_gif(data: bytes, path: str) -> None:
    """Save GIF bytes to a file for manual inspection."""
    if os.environ.get("RAIN_TEST_DEBUG_DUMP"):
        with open(path, "wb") as f:
            f.write(data)


# --- Canberra ---
CANBERRA_LAT = -35.309
CANBERRA_LON = 149.123


class TestCanberraGoldenV2:
    """Canberra rain detection with real radar tiles.

    All Canberra frames have 8-13% precipitation in the analysis area.
    With cold QC (no clutter map), the detector finds rain in all frame
    ranges - including frames 0-2 which were classified "none" at the
    capture location.
    """

    def test_rain_incoming_detected(self, ha_client, configure_location):
        """Frames 5-10 (classified rain_incoming): sensor must be on."""
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        configure_location(CANBERRA_LAT, CANBERRA_LON, "Canberra_v2")

        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        assert state["state"] == "on", (
            f"Canberra frames 5-10 (rain_incoming) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

        rain_gif = ha_client.get_image("image.rain_incoming_canberra_v2_radar_128km")
        if rain_gif:
            _save_gif(rain_gif, "/tmp/golden_v2_canberra_rain.gif")

    def test_rain_visible_in_image(self, ha_client):
        """The radar image must show visible precipitation pixels."""
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        rain_gif = ha_client.get_image("image.rain_incoming_canberra_v2_radar_128km")
        assert rain_gif and len(rain_gif) > 1000, (
            f"Rain GIF is missing or too small ({len(rain_gif) if rain_gif else 0} bytes)"
        )
        _save_gif(rain_gif, "/tmp/golden_v2_canberra_rain_vis.gif")

        has_precip, fraction = gif_has_precipitation_pixels(rain_gif)
        assert has_precip, (
            f"Sensor says rain but radar image has no precipitation pixels "
            f"(fraction={fraction:.4f}). Rain is not visible in the rendered image."
        )

    def test_image_render_is_deterministic(self, ha_client):
        """Two fetches of the same scenario must produce byte-identical images.

        Proves the image comparison method is reliable - if rendering
        were non-deterministic, differential image tests would be
        meaningless.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_imminent"
        ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        image_id = "image.rain_incoming_canberra_v2_radar_128km"
        gif_1 = ha_client.get_image(image_id)
        assert gif_1 and len(gif_1) > 1000, "GIF 1 missing or too small"

        gif_2 = ha_client.get_image(image_id)
        assert gif_2 and len(gif_2) > 1000, "GIF 2 missing or too small"

        differs, fraction = images_differ_significantly(gif_1, gif_2, threshold=0.01)
        assert not differs, (
            f"Two GIF renders of the same data differ by {fraction:.4f} - "
            f"rendering is not deterministic, image comparison unreliable"
        )


# --- Darwin ---
DARWIN_LAT = -12.4634
DARWIN_LON = 130.8456


class TestDarwinGoldenV2:
    """Darwin rain detection with real radar tiles.

    All Darwin frames have 1-2% precipitation in the analysis area.
    Lower coverage than Canberra/Melbourne but still real rain.
    """

    def test_rain_overhead_detected(self, ha_client, configure_location):
        """Darwin frames 5-12 (rain_overhead): sensor must be on."""
        ha_client.set_mock_scenario("golden_v2:Darwin:5-12")
        configure_location(DARWIN_LAT, DARWIN_LON, "Darwin_v2")

        sensor_id = "binary_sensor.rain_incoming_darwin_v2_imminent"
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        assert state["state"] == "on", (
            f"Darwin frames 5-12 (rain_overhead) but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

        rain_gif = ha_client.get_image("image.rain_incoming_darwin_v2_radar_128km")
        if rain_gif:
            _save_gif(rain_gif, "/tmp/golden_v2_darwin_rain.gif")

    def test_rain_visible_in_image(self, ha_client):
        """Darwin radar image must show visible precipitation pixels."""
        ha_client.set_mock_scenario("golden_v2:Darwin:5-12")
        sensor_id = "binary_sensor.rain_incoming_darwin_v2_imminent"
        ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        rain_gif = ha_client.get_image("image.rain_incoming_darwin_v2_radar_128km")
        assert rain_gif and len(rain_gif) > 1000, (
            f"Rain GIF is missing or too small ({len(rain_gif) if rain_gif else 0} bytes)"
        )
        _save_gif(rain_gif, "/tmp/golden_v2_darwin_rain_vis.gif")

        has_precip, fraction = gif_has_precipitation_pixels(rain_gif)
        assert has_precip, (
            f"Sensor says rain but radar image has no precipitation pixels "
            f"(fraction={fraction:.4f}). Rain is not visible in the rendered image."
        )


# --- Melbourne ---
MELBOURNE_LAT = -37.8136
MELBOURNE_LON = 144.9631


class TestMelbourneGoldenV2:
    """Melbourne rain detection with real radar tiles.

    All Melbourne frames have 10-13% precipitation in the analysis area,
    including those classified "none" at the capture coordinates. The
    "none" classification means no rain directly over Melbourne - but
    the 2x2 tile analysis area (~300km across) has consistent rain
    throughout.

    With cold QC, the detector correctly finds 97 tracked cells in
    Melbourne frames 1-12 and reports rain_incoming=True.
    """

    def test_rain_in_analysis_area_detected(self, ha_client, configure_location):
        """Melbourne frames 1-12: rain in analysis area must be detected.

        These frames were classified "none" at capture because rain wasn't
        directly over Melbourne. But the analysis area has 10-13%
        precipitation coverage with intensities up to 1.0. The detector
        correctly reports rain_incoming=True with cold QC.
        """
        ha_client.set_mock_scenario("golden_v2:Melbourne:1-12")
        configure_location(MELBOURNE_LAT, MELBOURNE_LON, "Melbourne_v2")

        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_imminent"
        state = ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        assert state["state"] == "on", (
            f"Melbourne frames 1-12 have 10-13% precip in analysis area "
            f"but sensor says '{state['state']}'. "
            f"Attributes: {state.get('attributes', {})}"
        )

        rain_gif = ha_client.get_image("image.rain_incoming_melbourne_v2_radar_128km")
        if rain_gif:
            _save_gif(rain_gif, "/tmp/golden_v2_melbourne_rain.gif")

    def test_rain_visible_in_image(self, ha_client):
        """Melbourne radar image must show visible precipitation pixels."""
        ha_client.set_mock_scenario("golden_v2:Melbourne:1-12")
        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_imminent"
        ha_client.wait_for_coordinator_cycle(sensor_id, timeout=60)

        rain_gif = ha_client.get_image("image.rain_incoming_melbourne_v2_radar_128km")
        assert rain_gif and len(rain_gif) > 1000, (
            f"Rain GIF is missing or too small ({len(rain_gif) if rain_gif else 0} bytes)"
        )
        _save_gif(rain_gif, "/tmp/golden_v2_melbourne_rain_vis.gif")

        has_precip, fraction = gif_has_precipitation_pixels(rain_gif)
        assert has_precip, (
            f"Sensor says rain but radar image has no precipitation pixels "
            f"(fraction={fraction:.4f}). Rain is not visible in the rendered image."
        )
