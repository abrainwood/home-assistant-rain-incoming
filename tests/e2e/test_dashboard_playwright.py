"""Playwright UI test for the Home Assistant dashboard.

Validates the radar image renders end-to-end through HA's frontend:
1. Login via the browser (with "Keep me logged in" - HA bounces clicks
   back to login otherwise)
2. Find the Rain Incoming radar card on the dashboard
3. Click into the more-info dialog where the actual <img> renders
4. Fetch the image bytes via the browser's authenticated session
5. Verify it's an animated GIF with multiple frames (not a placeholder)

The test manages its own browser inside the test function (no pytest-playwright
fixtures) to avoid event loop conflicts with the async E2E conftest.

Built interactively using the Playwright MCP - see SESSION_HANDOFF.md.
Stable selectors come from accessibility roles, image discovery uses a
shadow DOM walk, and binary data is transported as base64 from
page.evaluate().
"""
from __future__ import annotations

import base64
import re
from io import BytesIO

import pytest
from PIL import Image

from tests.e2e.conftest import HA_URL

# JavaScript run inside the browser to find the radar img element,
# fetch its bytes via the authenticated session, and return as base64.
# Filters on alt to skip the unlabelled thumbnail variant of the same image.
_FETCH_RADAR_GIF_JS = """
async () => {
  function findDeep(root, predicate, found = []) {
    if (root.querySelectorAll) {
      root.querySelectorAll('*').forEach(el => {
        if (predicate(el)) found.push(el);
        if (el.shadowRoot) findDeep(el.shadowRoot, predicate, found);
      });
    }
    return found;
  }
  const imgs = findDeep(document, el =>
    el.tagName === 'IMG'
    && el.src
    && el.src.includes('rain_incoming_radar')
    && el.alt
  );
  if (imgs.length === 0) return { error: 'no radar img element found' };
  const img = imgs[0];
  if (!img.complete) return { error: 'img not loaded yet' };
  const resp = await fetch(img.src);
  if (!resp.ok) return { error: 'fetch failed: ' + resp.status };
  const buf = await resp.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return {
    alt: img.alt,
    src: img.src,
    sizeBytes: bytes.length,
    base64: btoa(binary),
    naturalWidth: img.naturalWidth,
  };
}
"""


def _wait_for_radar_image(page, attempts: int = 40, interval_ms: int = 500) -> dict:
    """Poll until the radar image element is loaded in the dialog."""
    last_error = "polling timeout"
    for _ in range(attempts):
        page.wait_for_timeout(interval_ms)
        result = page.evaluate(_FETCH_RADAR_GIF_JS)
        if not result.get("error"):
            return result
        last_error = result["error"]
    raise AssertionError(f"Radar image never loaded: {last_error}")


class TestDashboardRadarRendering:
    """End-to-end browser test for radar image rendering on the HA dashboard."""

    def test_radar_gif_renders_as_animated_image(self, ha_client):
        """The dashboard renders our radar entity as an animated GIF.

        Proves end-to-end that:
        - HA loads our integration and registers the image entity
        - The dashboard shows the entity card with a real value
        - Clicking the card opens the more-info dialog with an <img>
        - HA's image proxy serves a real animated GIF (not a placeholder)
        - The GIF has multiple frames (real radar animation, not single-frame fallback)
        """
        from playwright.sync_api import sync_playwright

        # Make sure the entity has data before opening the browser
        ha_client.poll_entity_state(
            "image.rain_incoming_radar_128km",
            timeout=120,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            try:
                # Login. Note: must check "Keep me logged in" - without it, the
                # next click will bounce back to /auth/authorize.
                page.goto(HA_URL, wait_until="networkidle", timeout=30_000)
                page.get_by_role("textbox", name="Username").fill("dev")
                page.get_by_role("textbox", name="Password").fill("devdevdev")
                page.get_by_role("checkbox", name="Keep me logged in").check()
                page.get_by_role("button", name="Log in").click()

                # Wait for the dashboard
                page.wait_for_url(re.compile(r"/(home|lovelace)"), timeout=30_000)

                # Find and click the Rain Incoming radar card.
                # Accessible name includes the timestamp, so we use a regex.
                radar_card = page.get_by_role(
                    "button", name=re.compile(r"Rain Incoming Radar 128km")
                ).first
                radar_card.wait_for(state="visible", timeout=15_000)
                radar_card.click()

                # Poll for the more-info dialog to open and the image to load
                result = _wait_for_radar_image(page)

                # Validate it's a real animated radar GIF
                assert result["sizeBytes"] > 50_000, (
                    f"Image too small ({result['sizeBytes']} bytes) - "
                    f"likely a placeholder, not a real radar render"
                )
                assert result["naturalWidth"] >= 600, (
                    f"Unexpected image width: {result['naturalWidth']}"
                )

                # Parse with PIL and verify multi-frame animation
                gif = Image.open(BytesIO(base64.b64decode(result["base64"])))
                assert gif.format == "GIF", f"Expected GIF, got {gif.format}"
                assert getattr(gif, "is_animated", False), (
                    "Image is not animated - likely a placeholder"
                )
                assert gif.n_frames > 1, (
                    f"GIF has only {gif.n_frames} frame(s) - expected animated radar"
                )
            finally:
                context.close()
                browser.close()
