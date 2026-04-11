"""Playwright UI test for the Home Assistant dashboard.

One E2E test proves the full chain end-to-end:

    test_dashboard_links_to_real_animated_radar_gif

The test:
  1. Opens a browser, logs in, navigates to Developer Tools > States.
  2. Walks the shadow DOM to find our entity and extract the entity_picture URL
     that the UI actually exposes.
  3. Asserts the URL looks right (starts with /api/image_proxy/ and contains
     the entity ID).
  4. Fetches **that exact URL** from inside the browser (same session/cookies),
     base64-encodes the response, and returns it to Python.
  5. Decodes and validates with PIL: GIF magic bytes, animated, multi-frame.

Why one test: the image fetch URL comes from the browser extraction - you can't
split the two halves without either passing state between tests (bad coupling)
or re-doing the browser extraction in each test (wasteful). One test, one story:
"the dashboard links to a real animated radar GIF".

The test manages its own browser inside the test function (no pytest-playwright
fixtures) to avoid event loop conflicts with the async E2E conftest.

Built interactively using the Playwright MCP. The Developer Tools > States page
is more reliable than the dashboard for browser tests because it always lists
all entities regardless of dashboard customisation.
"""
from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from tests.e2e.conftest import HA_URL

ENTITY_ID = "image.rain_incoming_radar_128km"

# JS that fetches a URL using the browser's own authenticated session,
# base64-encodes the response body, and returns it. This exercises exactly
# the same code path a real browser user would trigger.
_FETCH_AS_BASE64_JS = """
async (url) => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error('fetch failed: ' + response.status + ' ' + url);
    }
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}
"""


def test_dashboard_links_to_real_animated_radar_gif(ha_client):
    """UI renders entity -> with working entity_picture URL -> URL serves real animated GIF.

    Steps:
    1. Poll entity readiness via REST (so we don't open a browser to an empty state page).
    2. Browser: login, navigate to Developer Tools > States.
    3. Browser: shadow-DOM walk to find the entity and extract the entity_picture URL.
    4. Assert URL shape: must start with /api/image_proxy/ and contain the entity ID.
    5. Browser: fetch that exact URL via the authenticated browser session.
    6. Python: decode base64, parse with PIL, assert GIF magic + animated + multi-frame.

    False-positive guards:
    - Assert realEntityCount > 0 (entity IS present).
    - Assert fakeEntityCount == 0 (our element search doesn't match random text).
    - Assert n_frames > 1 (rules out a single-frame placeholder).
    """
    from playwright.sync_api import sync_playwright

    # Wait for entity to be ready before opening the browser
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
            # --- Step 1: Login ---
            # "Keep me logged in" prevents subsequent navigations from
            # bouncing back to /auth/authorize.
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

            # --- Step 2: Navigate to Developer Tools > States ---
            page.goto(
                f"{HA_URL}/developer-tools/state",
                wait_until="networkidle",
                timeout=30_000,
            )

            # Wait for entity rows to render in the shadow DOM
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

            # --- Step 3: Extract entity_picture URL from the shadow DOM ---
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
                    // Pull entity_picture URL from any element that contains it
                    let entityPictureUrl = null;
                    const withUrl = findDeep(document, el =>
                        el.children.length === 0
                        && el.textContent
                        && el.textContent.includes('entity_picture:')
                        && el.textContent.includes('rain_incoming_radar_128km')
                    );
                    if (withUrl.length > 0) {{
                        const m = withUrl[0].textContent.match(
                            /entity_picture:\\s*(\\/api\\/image_proxy\\/[^\\s]+)/
                        );
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
                "integration may not be loaded"
            )
            assert checks["fakeEntityCount"] == 0, (
                "Fake entity found - false positive in element search"
            )
            assert checks["entityPictureUrl"], (
                "entity_picture URL not found in entity attributes"
            )

            entity_picture_url: str = checks["entityPictureUrl"]

            # --- Step 4: Assert URL shape ---
            assert entity_picture_url.startswith("/api/image_proxy/"), (
                f"entity_picture URL has unexpected prefix: {entity_picture_url!r}"
            )
            assert "rain_incoming_radar_128km" in entity_picture_url, (
                f"entity_picture URL doesn't contain entity ID: {entity_picture_url!r}"
            )

            # --- Step 5: Fetch that exact URL via the browser's authenticated session ---
            # Using page.evaluate() means we use the same cookies/auth as a real browser,
            # not a separate HTTP client. This proves the link the UI shows actually works.
            full_url = f"{HA_URL}{entity_picture_url}"
            image_b64: str = page.evaluate(_FETCH_AS_BASE64_JS, full_url)

        finally:
            context.close()
            browser.close()

    # --- Step 6: Validate the image bytes ---
    image_bytes = base64.b64decode(image_b64)

    assert image_bytes[:3] == b"GIF", (
        f"Image at {entity_picture_url!r} is not a GIF "
        f"(magic: {image_bytes[:6]!r}) - placeholder or error response"
    )

    gif = Image.open(BytesIO(image_bytes))
    assert gif.format == "GIF", f"Expected GIF, got {gif.format}"
    assert getattr(gif, "is_animated", False), (
        f"Image at {entity_picture_url!r} is not animated - likely a placeholder"
    )
    assert gif.n_frames > 1, (
        f"GIF at {entity_picture_url!r} has only {gif.n_frames} frame(s) - "
        "expected animated radar"
    )
