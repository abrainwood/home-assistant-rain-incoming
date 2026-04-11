"""Playwright UI test for the Home Assistant dashboard.

Two complementary E2E tests:

1. test_radar_entity_visible_in_dashboard_ui (Playwright/browser):
   Login, navigate to Developer Tools > States, verify our entity ID and
   entity_picture URL are present in the rendered UI, and verify a fake
   entity does NOT appear (false-positive guard). Proves the integration
   is actually visible to a user.

2. test_radar_image_proxy_serves_animated_gif (REST/PIL):
   Fetch image bytes via the authenticated REST API, parse with PIL, verify
   it's an animated GIF with multiple frames. Proves the image is a real
   radar render, not a placeholder or broken image.

Each test is self-contained - both poll for entity readiness independently.
The Docker container is the expensive part; a second Playwright session or
REST call is cheap.

The browser test manages its own browser inside the test function (no
pytest-playwright fixtures) to avoid event loop conflicts with the async
E2E conftest.

Built interactively using the Playwright MCP. The Developer Tools > States
page is more reliable than the dashboard for browser tests because it always
lists all entities regardless of dashboard customisation.
"""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from tests.e2e.conftest import HA_URL

ENTITY_ID = "image.rain_incoming_radar_128km"


def test_radar_entity_visible_in_dashboard_ui(ha_client):
    """The radar entity is visible in HA's Developer Tools > States UI.

    Browser-side checks (Playwright):
    - HA loads our integration
    - The entity appears in Developer Tools > States with an entity_picture URL
    - A fake entity does NOT appear (false-positive guard)

    These four assertions share the same DOM walk and form one coherent check:
    "our entity is visible in the HA UI with a working picture URL".
    """
    from playwright.sync_api import sync_playwright

    # Wait for the entity to be ready before opening the browser
    ha_client.poll_entity_state(
        ENTITY_ID,
        timeout=120,
        condition=lambda s: s.get("state") not in (None, "unavailable"),
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            # Login. "Keep me logged in" prevents subsequent navigations
            # from bouncing back to /auth/authorize.
            page.goto(HA_URL, wait_until="networkidle", timeout=30_000)
            page.get_by_role("textbox", name="Username").fill("dev")
            page.get_by_role("textbox", name="Password").fill("devdevdev")
            page.get_by_role("checkbox", name="Keep me logged in").check()
            page.get_by_role("button", name="Log in").click()

            # Wait for any non-auth page to load (HA may go to /home or /onboarding)
            page.wait_for_function(
                "() => !window.location.pathname.startsWith('/auth')",
                timeout=30_000,
            )

            # Navigate to Developer Tools > States - always shows all entities
            page.goto(
                f"{HA_URL}/developer-tools/state",
                wait_until="networkidle",
                timeout=30_000,
            )

            # Wait for the entity rows to render
            page.wait_for_function(
                f"""() => {{
                    function findDeep(root, found = []) {{
                        if (root.querySelectorAll) {{
                            root.querySelectorAll('*').forEach(el => {{
                                if (el.children.length === 0 && el.textContent
                                    && el.textContent.includes('{ENTITY_ID}')) {{
                                    found.push(el);
                                }}
                                if (el.shadowRoot) findDeep(el.shadowRoot, found);
                            }});
                        }}
                        return found;
                    }}
                    return findDeep(document).length > 0;
                }}""",
                timeout=30_000,
            )

            # Verify entity is present and a fake entity is not
            checks = page.evaluate(f"""
                () => {{
                    function findDeep(root, predicate, found = []) {{
                        if (root.querySelectorAll) {{
                            root.querySelectorAll('*').forEach(el => {{
                                if (predicate(el)) found.push(el);
                                if (el.shadowRoot) findDeep(el.shadowRoot, predicate, found);
                            }});
                        }}
                        return found;
                    }}
                    const realEntity = findDeep(document, el =>
                        el.children.length === 0
                        && el.textContent
                        && el.textContent.includes('{ENTITY_ID}')
                    );
                    const fakeEntity = findDeep(document, el =>
                        el.children.length === 0
                        && el.textContent
                        && el.textContent.includes('image.fake_nonexistent_xyz_789')
                    );
                    // Pull entity_picture URL from any element that has it
                    let entityPictureUrl = null;
                    const withUrl = findDeep(document, el =>
                        el.children.length === 0
                        && el.textContent
                        && el.textContent.includes('entity_picture:')
                        && el.textContent.includes('rain_incoming_radar_128km')
                    );
                    if (withUrl.length > 0) {{
                        const m = withUrl[0].textContent.match(/entity_picture:\\s*(\\/api\\/image_proxy\\/[^\\s]+)/);
                        if (m) entityPictureUrl = m[1];
                    }}
                    return {{
                        realEntityCount: realEntity.length,
                        fakeEntityCount: fakeEntity.length,
                        entityPictureUrl,
                    }};
                }}
            """)

            assert checks["realEntityCount"] > 0, (
                f"{ENTITY_ID} not found in Developer Tools > States - "
                f"integration may not be loaded"
            )
            assert checks["fakeEntityCount"] == 0, (
                "Fake entity found - false positive in element search"
            )
            assert checks["entityPictureUrl"], (
                "entity_picture URL not found in entity attributes"
            )
            assert "rain_incoming_radar_128km" in checks["entityPictureUrl"]

        finally:
            context.close()
            browser.close()


def test_radar_image_proxy_serves_animated_gif(ha_client):
    """The image proxy serves a real animated GIF for our radar entity.

    REST-side checks (HAClient + PIL):
    - Image bytes are returned (not None)
    - GIF magic bytes are present
    - PIL parses it as GIF format
    - The GIF is animated (is_animated=True)
    - The GIF has multiple frames (n_frames > 1) - rules out a placeholder

    These assertions form a ladder: bytes first, then parse, then frame count.
    Splitting them would require duplicating the get_image + PIL parse in each
    test, so they stay together as one compound assertion.
    """
    # Wait for the entity to be ready before fetching the image
    ha_client.poll_entity_state(
        ENTITY_ID,
        timeout=120,
        condition=lambda s: s.get("state") not in (None, "unavailable"),
    )

    image_bytes = ha_client.get_image(ENTITY_ID)
    assert image_bytes is not None, f"No image data for {ENTITY_ID}"
    assert image_bytes[:3] == b"GIF", (
        f"Image is not a GIF (magic: {image_bytes[:6]!r}) - "
        f"placeholder or error response"
    )

    gif = Image.open(BytesIO(image_bytes))
    assert gif.format == "GIF", f"Expected GIF, got {gif.format}"
    assert getattr(gif, "is_animated", False), (
        "Image is not animated - likely a placeholder"
    )
    assert gif.n_frames > 1, (
        f"GIF has only {gif.n_frames} frame(s) - expected animated radar"
    )
