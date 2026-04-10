"""
Playwright smoke tests for the HA dashboard UI.

Only test_dashboard_loads_with_integration uses the browser - it validates
that HA's frontend loads and we can authenticate. The remaining tests use
ha_client (REST API) since they're asserting entity state and image data,
not browser rendering.
"""
from __future__ import annotations

from io import BytesIO

import pytest

HA_URL = "http://localhost:18123"

BINARY_SENSOR = "binary_sensor.rain_incoming_status"
IMAGE_128 = "image.rain_incoming_radar_128km"


class TestDashboardUI:
    """UI smoke test - only the dashboard load test needs a browser."""

    def test_dashboard_loads_with_integration(self, page, ha_client):
        """HA frontend loads, login works, and app shell renders."""
        ha_client.set_mock_scenario("rain_everywhere")

        page.goto(HA_URL, wait_until="networkidle", timeout=30_000)

        # Fill login form - try multiple selector strategies for HA version compat
        page.locator("input[name='username']").fill("dev")
        page.locator("input[name='password']").fill("devdevdev")

        # HA's submit button varies across versions - click whatever's visible
        page.locator("text=Log in").or_(
            page.locator("mwc-button")
        ).or_(
            page.locator("button[type='submit']")
        ).first.click()

        # Wait for the app shell - confirms auth succeeded and frontend loaded
        page.wait_for_selector("home-assistant", timeout=30_000)
        assert "/auth/authorize" not in page.url, "Redirected back to login"
        assert "Home Assistant" in page.title()


class TestEntityState:
    """Entity state assertions via REST API - no browser needed."""

    def test_sensor_has_value_with_rain(self, ha_client):
        """With rain_everywhere scenario, binary sensor should be on."""
        ha_client.set_mock_scenario("rain_everywhere")
        state = ha_client.poll_entity_state(
            BINARY_SENSOR, timeout=60,
            condition=lambda s: s.get("state") == "on",
        )
        assert state["state"] == "on"

    def test_sensor_off_with_no_rain(self, ha_client):
        """With no_rain scenario, binary sensor should be off (not unavailable)."""
        ha_client.set_mock_scenario("no_rain")
        state = ha_client.poll_entity_state(
            BINARY_SENSOR, timeout=60,
            condition=lambda s: s.get("state") in ("on", "off"),
        )
        assert state["state"] == "off"


class TestRadarImage:
    """Radar image assertions via REST API - no browser needed."""

    def test_image_returns_valid_gif(self, ha_client):
        """Radar image entity returns valid GIF bytes."""
        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.poll_entity_state(
            IMAGE_128, timeout=60,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

        image_bytes = ha_client.get_image(IMAGE_128)
        assert image_bytes is not None, f"No image data for {IMAGE_128}"
        assert len(image_bytes) > 500, f"Image too small ({len(image_bytes)} bytes)"
        assert image_bytes[:3] == b"GIF", f"Not a GIF (magic: {image_bytes[:6]!r})"

    def test_gif_has_multiple_frames(self, ha_client):
        """Radar GIF is animated (more than one frame)."""
        from PIL import Image

        ha_client.set_mock_scenario("rain_everywhere")
        ha_client.poll_entity_state(
            IMAGE_128, timeout=60,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

        image_bytes = ha_client.get_image(IMAGE_128)
        assert image_bytes is not None

        gif = Image.open(BytesIO(image_bytes))
        assert getattr(gif, "n_frames", 1) > 1, (
            f"GIF has only {getattr(gif, 'n_frames', 1)} frame(s)"
        )
