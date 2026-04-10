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


# RainViewer Universal Blue scheme 2 precipitation colours (RGB).
# These are the distinctive blues/cyans/yellows/reds that indicate rain.
_PRECIP_COLOURS_RGB = np.array([
    [0, 154, 213],   # light rain (cyan-blue)
    [0, 130, 202],
    [0, 105, 191],
    [22, 170, 0],    # moderate (greens)
    [31, 190, 0],
    [255, 240, 0],   # heavy (yellows)
    [255, 200, 0],
    [255, 140, 0],   # very heavy (oranges)
    [255, 80, 0],
    [255, 0, 0],     # extreme (reds)
    [200, 0, 0],
], dtype=np.float32)


def _gif_has_precipitation_pixels(
    gif_bytes: bytes, threshold: float = 0.005, max_colour_dist: float = 40.0
) -> tuple[bool, float]:
    """Check if a GIF contains visible precipitation-coloured pixels.

    Matches pixels against known RainViewer precipitation colours using
    L2 distance. Returns (has_precip, fraction_of_pixels).
    """
    img = Image.open(BytesIO(gif_bytes))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    arr = np.array(img.convert("RGB")).astype(np.float32)

    # Compute L2 distance from each pixel to each precipitation colour
    diff = arr[:, :, np.newaxis, :] - _PRECIP_COLOURS_RGB[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W, N)
    best_dist = distances.min(axis=-1)  # (H, W)

    precip_mask = best_dist < max_colour_dist
    fraction = precip_mask.sum() / (arr.shape[0] * arr.shape[1])
    return fraction > threshold, float(fraction)


def _save_gif(data: bytes, path: str) -> None:
    """Save GIF bytes to a file for manual inspection."""
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

        sensor_id = "binary_sensor.rain_incoming_canberra_v2_status"
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

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
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_status"

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

        # Force image re-render by fetching after coordinator update
        baseline_gif = ha_client.get_image(image_id)
        if baseline_gif:
            _save_gif(baseline_gif, "/tmp/golden_v2_canberra_norain.gif")

        assert baseline_gif and len(baseline_gif) > 100, "Baseline GIF missing"

        differs, fraction = _images_differ_significantly(rain_gif, baseline_gif)
        assert differs, (
            f"Sensor says rain but radar image identical to baseline (diff={fraction:.4f}). "
            f"Rain is not visible in the rendered image. "
            f"GIFs saved to /tmp/golden_v2_canberra_*.gif for inspection."
        )

    def test_no_rain_when_classified_none(self, ha_client):
        """Frames 0-2 are all 'none'. With only none frames, sensor should be off."""
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_status"
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
        sensor_id = "binary_sensor.rain_incoming_canberra_v2_status"
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

        With dry golden data, the image should have no precipitation pixels.
        If it does, our image inspection can't distinguish rain from no-rain.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:0-2")
        ha_client.update_entity("binary_sensor.rain_incoming_canberra_v2_status")
        ha_client.poll_entity_state(
            "binary_sensor.rain_incoming_canberra_v2_status", timeout=30,
            condition=lambda s: s.get("state") == "off",
        )

        dry_gif = ha_client.get_image("image.rain_incoming_canberra_v2_radar_128km")
        assert dry_gif and len(dry_gif) > 100, "Dry GIF missing"

        has_precip, fraction = _gif_has_precipitation_pixels(dry_gif)
        assert not has_precip, (
            f"Dry golden data has precipitation pixels (fraction={fraction:.4f}) - "
            f"image inspection too sensitive, would produce false positives"
        )

    def test_rain_data_would_fail_dry_assertion(self, ha_client):
        """Verify that rain data does NOT produce a dry sensor state.

        If this test passed with state='off', it would mean our rain detection
        test is meaningless - it would pass even when the pipeline is broken.
        """
        ha_client.set_mock_scenario("golden_v2:Canberra:5-10")
        ha_client.update_entity("binary_sensor.rain_incoming_canberra_v2_status")
        state = ha_client.poll_entity_state(
            "binary_sensor.rain_incoming_canberra_v2_status", timeout=30,
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

        sensor_id = "binary_sensor.rain_incoming_darwin_v2_status"
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

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
        sensor_id = "binary_sensor.rain_incoming_darwin_v2_status"

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

        has_precip, fraction = _gif_has_precipitation_pixels(rain_gif)
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

        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_status"
        state = ha_client.poll_entity_state(
            sensor_id, timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

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
        sensor_id = "binary_sensor.rain_incoming_melbourne_v2_status"
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
