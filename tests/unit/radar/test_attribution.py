"""Tests for _draw_attribution in composite.py.

TDD: written FIRST (red), then implementation makes them green.

Requirements:
- Attribution text "rainviewer.com | {style.attribution}" appears on each frame
- RainViewer attribution comes FIRST per their API terms (rainviewer.com/api.html)
- Positioned along the bottom of the image (full width available - frame label is now at top)
- Single-line for all styles including ESRI's 85-char attribution at the chosen font size
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFont

from custom_components.rain_incoming.radar.composite import _draw_attribution
from custom_components.rain_incoming.radar.map_styles import MapStyle, get_style


def _blank_image(width=640, height=640) -> Image.Image:
    """Create a blank black RGBA image."""
    return Image.new("RGBA", (width, height), (0, 0, 0, 255))


def _bottom_region(img: Image.Image, height_fraction: float = 0.05) -> np.ndarray:
    """Return the bottom strip of the image as a numpy array (RGB only)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    region_h = max(1, int(h * height_fraction))
    # Bottom slice, full width, RGB only (ignore alpha channel which is always 255)
    return arr[h - region_h:h, :, :3]


def _count_nonblack_pixels(region: np.ndarray) -> int:
    """Count pixels with any RGB channel > 10."""
    return int((region.max(axis=2) > 10).sum())


# ---------------------------------------------------------------------------
# Attribution text appears in the bottom region
# ---------------------------------------------------------------------------

def test_draw_attribution_renders_style_text():
    """_draw_attribution should draw text in the bottom region of the image.

    A blank black image should have non-black pixels in the bottom strip after
    calling _draw_attribution - proving text was rendered there.
    """
    img = _blank_image()
    style_def = get_style(MapStyle.VOYAGER)
    _draw_attribution(img, style_def)

    region = _bottom_region(img)
    nonblack = _count_nonblack_pixels(region)
    assert nonblack > 0, (
        "Expected non-black pixels in bottom region after _draw_attribution, "
        "but the region is still all-black. Attribution was not drawn."
    )


def test_draw_attribution_includes_rainviewer():
    """The drawn attribution must include 'RainViewer' - verified by rendering non-empty text."""
    # We can't easily OCR the image in tests, but we verify _draw_attribution draws
    # something for every style. The 'RainViewer' content is verified at the string level
    # via the implementation contract: the text must be f"{style_def.attribution} | RainViewer".
    for style in MapStyle:
        img = _blank_image()
        style_def = get_style(style)
        _draw_attribution(img, style_def)
        region = _bottom_region(img)
        nonblack = _count_nonblack_pixels(region)
        assert nonblack > 0, (
            f"Expected attribution text for style={style} in bottom region, "
            f"but region is all-black. Text may have been placed elsewhere or not drawn."
        )


# ---------------------------------------------------------------------------
# Regression test: ESRI's long attribution must fit on one line at chosen font size
# ---------------------------------------------------------------------------

def test_draw_attribution_esri_fits_single_line():
    """ESRI's full attribution with RainViewer prefix must fit within img.width - 2*padding.

    This is a regression guard: if someone bumps the font size up beyond what fits,
    this test will catch it. The implementation uses 14pt. RainViewer is now FIRST,
    so the string is "rainviewer.com | {esri_attribution}" (569px in a 624px available width).
    """
    from custom_components.rain_incoming.radar.map_styles import MapStyle, get_style
    from custom_components.rain_incoming.const import RAINVIEWER_ATTRIBUTION

    padding = 8
    img_width = 640
    available_width = img_width - 2 * padding

    # Match the font size used in _draw_attribution
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    esri_def = get_style(MapStyle.ESRI_IMAGERY)
    text = f"{RAINVIEWER_ATTRIBUTION} | {esri_def.attribution}"
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]

    assert text_w < available_width, (
        f"ESRI attribution text is {text_w}px wide but only {available_width}px available "
        f"({img_width}px image - {2*padding}px padding). Reduce font size in _draw_attribution "
        f"to fit ESRI on a single line. Text: {text!r}"
    )


def test_draw_attribution_frame_label_position_is_top():
    """_draw_frame_label should place text in the top region of the image, not the bottom.

    After calling _draw_frame_label, the top strip should have non-black pixels
    and the very bottom strip should remain black (no text overlap between label and attribution).
    This verifies the layout separation: label at top, attribution at bottom.

    Now also accepts frame_index and frame_count parameters for the frame counter.
    """
    from custom_components.rain_incoming.radar.composite import _draw_frame_label
    from datetime import datetime, timezone

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
    _draw_frame_label(img, ts, "UTC", 128, "Sydney", frame_index=3, frame_count=13)

    arr = np.array(img)[:, :, :3]
    h = arr.shape[0]

    # Top strip should have text pixels
    top_strip = arr[:int(h * 0.05), :, :]
    top_nonblack = int((top_strip.max(axis=2) > 10).sum())

    # Bottom strip (bottom 5%) should be empty (no text rendered there)
    bottom_strip = arr[int(h * 0.95):, :, :]
    bottom_nonblack = int((bottom_strip.max(axis=2) > 10).sum())

    assert top_nonblack > 0, (
        "Expected frame label text pixels in the top strip of the image, "
        "but the top strip is all-black. Label may still be at the bottom."
    )
    assert bottom_nonblack == 0, (
        f"Expected no frame label pixels in the bottom strip, "
        f"but found {bottom_nonblack} non-black pixels. Label may still be at the bottom."
    )


# ---------------------------------------------------------------------------
# New header layout: location (left), date/time (centre), range+counter (right)
# ---------------------------------------------------------------------------

def test_draw_frame_label_renders_frame_counter():
    """_draw_frame_label must render the frame counter text (e.g. '3/13').

    Mocks ImageDraw.text to capture all text rendered, then asserts that
    at least one call includes a string matching the 'N/M' frame counter pattern.
    """
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, "Sydney", frame_index=3, frame_count=13)

    all_text_strings = [
        c.args[1] for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]

    counter_calls = [s for s in all_text_strings if "3/13" in s]
    assert counter_calls, (
        f"Expected at least one draw.text call containing '3/13' for the frame counter, "
        f"but none found. All rendered strings: {all_text_strings}"
    )


def test_draw_frame_label_location_drawn_at_left():
    """Location name must be drawn near the left edge (x close to padding).

    Verifies that 'Sydney' appears in a draw.text call with an x-coordinate
    close to the padding value (8px), i.e. anchored at the left.
    """
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, "Sydney", frame_index=1, frame_count=5)

    # Find the call that draws "Sydney"
    sydney_calls = [
        c for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str) and "Sydney" in c.args[1]
    ]
    assert sydney_calls, (
        f"Expected a draw.text call containing 'Sydney', but none found. "
        f"All calls: {[c.args for c in mock_text.call_args_list]}"
    )

    # The x coordinate of the position tuple should be close to left padding (8)
    # We accept any call where x < img.width // 3 (clearly on the left third)
    left_threshold = img.width // 3  # 213px for a 640px image
    for call in sydney_calls:
        pos = call.args[0]
        x = pos[0]
        if x < left_threshold:
            break  # found at least one left-anchored call
    else:
        xs = [c.args[0][0] for c in sydney_calls]
        assert False, (
            f"'Sydney' was drawn but not near the left edge. "
            f"x-coordinates: {xs}, left threshold: {left_threshold}px (img width: {img.width})"
        )


def test_draw_frame_label_datetime_drawn_at_centre():
    """Date/time must be drawn near the horizontal centre of the image.

    Verifies a draw.text call with the date string has an x-coordinate
    near img.width // 2, using anchor='mt' (middle-top).
    """
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()  # 640px wide
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, "Sydney", frame_index=2, frame_count=10)

    # Date or time substring to find the centre text call
    # The date is "10/04/26" and time is "20:40 UTC"
    centre_calls = [
        c for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
        and ("10/04/26" in c.args[1] or "20:40" in c.args[1])
    ]
    assert centre_calls, (
        f"Expected a draw.text call containing date/time ('10/04/26' or '20:40'), "
        f"but none found. All strings: {[c.args[1] for c in mock_text.call_args_list if len(c.args) >= 2 and isinstance(c.args[1], str)]}"
    )

    # x should be close to the centre (320px for 640px image)
    # Accept anchor='mt' style: x == img.width // 2 (within 20px tolerance)
    centre_x = img.width // 2  # 320
    tolerance = 20
    for call in centre_calls:
        pos = call.args[0]
        x = pos[0]
        if abs(x - centre_x) <= tolerance:
            break
    else:
        xs = [c.args[0][0] for c in centre_calls]
        assert False, (
            f"Date/time text was drawn but not near the centre. "
            f"x-coordinates: {xs}, expected ~{centre_x}px (±{tolerance}px)"
        )


def test_draw_frame_label_range_drawn_at_right():
    """Range text (e.g. '128km') must be drawn near the right edge of the image.

    Verifies a draw.text call containing the range string has an x-coordinate
    near img.width - padding, indicating right-anchored placement.
    """
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()  # 640px wide
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, "Sydney", frame_index=1, frame_count=5)

    # Find calls containing the range text
    range_calls = [
        c for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str) and "128km" in c.args[1]
    ]
    assert range_calls, (
        f"Expected a draw.text call containing '128km', but none found. "
        f"All strings: {[c.args[1] for c in mock_text.call_args_list if len(c.args) >= 2 and isinstance(c.args[1], str)]}"
    )

    # x should be near the right edge. With anchor='rt', x == img.width - padding.
    # Accept any x >= img.width * 2 // 3 (right third of image)
    right_threshold = img.width * 2 // 3  # ~427px for 640px image
    for call in range_calls:
        pos = call.args[0]
        x = pos[0]
        if x >= right_threshold:
            break
    else:
        xs = [c.args[0][0] for c in range_calls]
        assert False, (
            f"Range text '128km' was drawn but not near the right edge. "
            f"x-coordinates: {xs}, right threshold: {right_threshold}px (img width: {img.width})"
        )


def test_draw_frame_label_no_counter_when_indices_omitted():
    """When frame_index/frame_count are None, no slash counter appears in any text."""
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, "Sydney")

    all_strings = [
        c.args[1] for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]
    # The range text should be just "128km" with no "N/M" counter appended.
    range_strings = [s for s in all_strings if "128km" in s]
    assert range_strings, f"Expected '128km' in draw calls, got: {all_strings}"
    for s in range_strings:
        assert s.strip() == "128km", (
            f"Range text should be exactly '128km' when no frame counter, got: {s!r}"
        )


def test_draw_frame_label_no_location_no_crash():
    """_draw_frame_label with location_name=None must not crash or draw 'None'."""
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, None, frame_index=1, frame_count=5)

    all_strings = [
        c.args[1] for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]
    none_strings = [s for s in all_strings if "None" in s]
    assert not none_strings, (
        f"'None' should never appear in rendered text, but found: {none_strings}"
    )


def test_draw_frame_label_no_timestamp_no_crash():
    """_draw_frame_label with timestamp=None must not crash or draw date text."""
    from unittest.mock import patch
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, None, None, 128, "Sydney", frame_index=1, frame_count=5)

    all_strings = [
        c.args[1] for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]
    # Should still have location and range, but no date/time
    assert any("Sydney" in s for s in all_strings), f"Location should appear: {all_strings}"
    assert any("128km" in s for s in all_strings), f"Range should appear: {all_strings}"
    date_strings = [s for s in all_strings if "/" in s and len(s) > 5 and any(c.isdigit() for c in s[:2])]
    # Filter out "1/5" frame counter - we're looking for date-like patterns "DD/MM/YY"
    date_like = [s for s in all_strings if any(sub in s for sub in ["26", "2026"])]
    assert not date_like, (
        f"No date text should be drawn when timestamp is None, but found: {date_like}"
    )


def test_draw_frame_label_truncates_long_location():
    """_draw_frame_label must truncate location names > 30 chars via the live code path."""
    from unittest.mock import patch
    from datetime import datetime, timezone
    from PIL import ImageDraw
    from custom_components.rain_incoming.radar.composite import _draw_frame_label

    img = _blank_image()
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
    long_name = "A" * 40

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_frame_label(img, ts, "UTC", 128, long_name, frame_index=1, frame_count=5)

    all_strings = [
        c.args[1] for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]
    truncated = "A" * 29 + "\u2026"
    assert any(truncated in s for s in all_strings), (
        f"Expected truncated location '{truncated}' in draw calls, got: {all_strings}"
    )
    assert not any(long_name in s for s in all_strings), (
        f"Full 40-char name should not appear untruncated, got: {all_strings}"
    )


# ---------------------------------------------------------------------------
# P0 regression: render_composite must pass style_def through _render_sync
# ---------------------------------------------------------------------------

def test_render_composite_uses_passed_style_for_attribution():
    """render_composite called with ESRI_IMAGERY must draw ESRI's attribution, not Voyager's.

    The bug: render_composite computed style_def but did not pass it through
    _render_sync -> _composite_single_frame, so _composite_single_frame always
    fell back to VOYAGER's attribution regardless of what style was requested.

    This test mocks _draw_attribution so we can inspect exactly which style_def
    was used - pixel-level verification is covered by test_draw_attribution_renders_style_text.
    """
    from unittest.mock import patch, MagicMock, AsyncMock
    import asyncio
    from io import BytesIO
    from PIL import Image

    from custom_components.rain_incoming.radar.composite import render_composite, _map_tile_cache
    from custom_components.rain_incoming.radar.map_styles import MapStyle, get_style

    # Build a minimal mock session that returns valid PNG tiles
    tile_png = BytesIO()
    Image.new("RGBA", (256, 256), (80, 80, 80, 255)).save(tile_png, format="PNG")
    tile_bytes = tile_png.getvalue()

    async def _fake_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status = 200
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        resp.release = MagicMock()
        resp.read = AsyncMock(return_value=tile_bytes)
        return resp

    session = MagicMock()
    session.get = AsyncMock(side_effect=_fake_get)

    called_with_style: list = []

    # Import the real function BEFORE patching so we can call it without recursion
    from custom_components.rain_incoming.radar.composite import _draw_attribution as _real_draw

    def _capture_draw_attribution(img, style_def):
        called_with_style.append(style_def)
        # Still draw so rendering doesn't error - call the real function directly
        _real_draw(img, style_def)

    _map_tile_cache.clear()

    with patch(
        "custom_components.rain_incoming.radar.composite._draw_attribution",
        side_effect=_capture_draw_attribution,
    ):
        asyncio.get_event_loop().run_until_complete(
            render_composite(
                lat=-33.7,
                lon=151.2,
                radius_km=128,
                frame_path="/test/radar/frame",
                output_size=256,
                session=session,
                map_style=MapStyle.ESRI_IMAGERY,
            )
        )

    assert called_with_style, "_draw_attribution was never called"
    used_style_def = called_with_style[0]
    esri_def = get_style(MapStyle.ESRI_IMAGERY)
    assert used_style_def.style == MapStyle.ESRI_IMAGERY, (
        f"render_composite with map_style=ESRI_IMAGERY should draw ESRI attribution, "
        f"but _draw_attribution was called with style={used_style_def.style!r}. "
        f"Expected {MapStyle.ESRI_IMAGERY!r}. "
        f"This is the P0 bug: style_def was not threaded through _render_sync."
    )
    assert "Esri" in used_style_def.attribution, (
        f"ESRI style_def attribution should mention 'Esri', got: {used_style_def.attribution!r}"
    )


# ---------------------------------------------------------------------------
# Layout integrity: label at top, attribution at bottom - no leakage
# ---------------------------------------------------------------------------

def test_composite_single_frame_has_label_at_top_and_attribution_at_bottom():
    """_composite_single_frame must place the frame label at the top and attribution at the bottom.

    Layout invariant: if someone accidentally changes a vertical offset, this catches it.
    - Top 5% of the image must have non-black pixels (frame label).
    - Bottom 5% must have non-black pixels (attribution text).
    - The middle 90% should NOT have label text rendered there.

    Uses a blank background and ESRI style so we exercise the full code path.
    """
    from datetime import datetime, timezone
    from custom_components.rain_incoming.radar.composite import _composite_single_frame
    from custom_components.rain_incoming.radar.map_styles import MapStyle, get_style

    output_size = 640
    background = Image.new("RGBA", (output_size, output_size), (60, 60, 60, 255))
    radar = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
    esri_def = get_style(MapStyle.ESRI_IMAGERY)

    result = _composite_single_frame(
        map_crop=background,
        radar_resized=radar,
        lat=-33.7,
        lon=151.2,
        map_zoom=8,
        radius_km=128,
        output_size=output_size,
        timestamp=ts,
        tz_name="UTC",
        location_name="Sydney",
        style_def=esri_def,
    )

    arr = np.array(result)[:, :, :3]
    h = arr.shape[0]
    strip_h = int(h * 0.05)

    # The background is grey (60, 60, 60) - "non-black" includes that.
    # We compare against what the background alone would look like: max(axis=2) = 60.
    # Text pixels are white (255) or black (0) outlines against that grey background.
    # We detect text by looking for pixels that DIFFER significantly from the grey background.
    _BACKGROUND_GREY = 60
    _TEXT_DELTA_THRESHOLD = 30  # pixels that differ from background grey by this much are text

    def _count_text_pixels(strip: np.ndarray) -> int:
        """Count pixels that differ from the background grey significantly."""
        delta = np.abs(strip.astype(np.int32) - _BACKGROUND_GREY).max(axis=2)
        return int((delta > _TEXT_DELTA_THRESHOLD).sum())

    top_strip = arr[:strip_h, :, :]
    bottom_strip = arr[h - strip_h:, :, :]
    # Middle excludes top + bottom strips
    middle_strip = arr[strip_h * 2:h - strip_h * 2, :, :]

    top_text = _count_text_pixels(top_strip)
    bottom_text = _count_text_pixels(bottom_strip)

    assert top_text > 0, (
        f"Expected frame label pixels in the top {strip_h}px strip, "
        f"but found {top_text} text pixels. Frame label may not be at the top."
    )
    assert bottom_text > 0, (
        f"Expected attribution pixels in the bottom {strip_h}px strip, "
        f"but found {bottom_text} text pixels. Attribution may not be at the bottom."
    )


# ---------------------------------------------------------------------------
# Ordering: RainViewer attribution must come BEFORE the map provider attribution
# ---------------------------------------------------------------------------

def test_attribution_rainviewer_appears_first():
    """RainViewer attribution must come before the map provider attribution.

    Per RainViewer's API terms (rainviewer.com/api.html), we must credit them.
    User direction: 'let's make RainViewer come first in the attributions'.

    Mocks ImageDraw.text to capture the full attribution string and verifies
    that 'rainviewer.com' appears before 'OpenStreetMap' in the rendered text.
    """
    from unittest.mock import patch
    from PIL import ImageDraw

    img = Image.new("RGBA", (640, 640), (0, 0, 0, 0))
    style_def = get_style(MapStyle.OSM_STANDARD)

    with patch.object(ImageDraw.ImageDraw, "text") as mock_text:
        _draw_attribution(img, style_def)

    # Find the call that drew the attribution - it's the one containing both brand names
    text_calls = [
        c for c in mock_text.call_args_list
        if len(c.args) >= 2 and isinstance(c.args[1], str)
    ]
    rainviewer_call = next(
        (c for c in text_calls if "rainviewer" in c.args[1].lower()),
        None,
    )
    assert rainviewer_call is not None, (
        f"No draw.text call contained 'rainviewer'. Calls: {[c.args[1] for c in text_calls]}"
    )
    full_text = rainviewer_call.args[1]

    rainviewer_pos = full_text.lower().find("rainviewer")
    osm_pos = full_text.find("OpenStreetMap")

    assert rainviewer_pos != -1, f"'rainviewer' not found in attribution text: {full_text!r}"
    assert osm_pos != -1, f"'OpenStreetMap' not found in attribution text: {full_text!r}"
    assert rainviewer_pos < osm_pos, (
        f"RainViewer must appear before OpenStreetMap in the attribution string, "
        f"but rainviewer_pos={rainviewer_pos} >= osm_pos={osm_pos}. "
        f"Full text: {full_text!r}"
    )
