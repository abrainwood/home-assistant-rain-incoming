"""Unit tests for composable rendering pipeline.

The rendering pipeline exposes independent building blocks:
- fetch_map_crop(): fetches CartoDB map tiles for a location/radius
- fetch_radar_overlays(): fetches RainViewer radar tiles for frame paths
- composite_frames(): composites radar over a background image
- render_gif(): encodes PIL frames to animated GIF bytes

Tests can compose these directly for clean, fast assertions without
needing the full E2E stack.
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.radar.composite import (
    composite_frames,
    fetch_map_crop,
    fetch_radar_overlays,
    render_animated_composite,
    render_gif,
)
from tests.e2e.image_helpers import PRECIP_COLOURS_RGB


def _make_mock_session():
    """Create a mock aiohttp session that returns valid tile PNGs."""
    session = MagicMock()

    async def _mock_get(url, **kwargs):
        resp = MagicMock()
        if "tilecache.rainviewer.com" in url:
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        else:
            img = Image.new("RGBA", (256, 256), (128, 128, 128, 255))
        buf = BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        resp.read = AsyncMock(return_value=buf.getvalue())
        resp.status = 200
        resp.release = MagicMock()
        return resp

    session.get = AsyncMock(side_effect=_mock_get)
    return session


def _count_non_black_pixels(gif_bytes: bytes) -> int:
    """Count pixels that aren't black in the last frame of a GIF."""
    img = Image.open(BytesIO(gif_bytes))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    arr = np.array(img.convert("RGB"))
    return int((arr.max(axis=2) > 10).sum())


def _count_precipitation_pixels(gif_bytes: bytes, max_colour_dist: float = 40.0) -> tuple[int, float]:
    """Count pixels matching precipitation colours in the last frame of a GIF.

    Returns (count, fraction_of_total).
    """
    img = Image.open(BytesIO(gif_bytes))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    arr = np.array(img.convert("RGB")).astype(np.float32)

    diff = arr[:, :, np.newaxis, :] - np.array(PRECIP_COLOURS_RGB, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    best_dist = distances.min(axis=-1)

    precip_mask = best_dist < max_colour_dist
    count = int(precip_mask.sum())
    total = arr.shape[0] * arr.shape[1]
    return count, count / total


class TestComposableRenderingPipeline:
    """Test the composable building blocks of the rendering pipeline."""

    @pytest.mark.asyncio
    async def test_radar_only_dry_data_is_mostly_black(self):
        """Radar overlays with no precipitation composited over black
        should produce a mostly-black image (only annotations visible)."""
        session = _make_mock_session()

        overlays = await fetch_radar_overlays(
            session=session, lat=-33.701, lon=151.209, radius_km=128,
            frame_paths=["/v2/radar/test_frame"],
        )
        assert len(overlays) == 1

        background = Image.new("RGBA", (640, 640), (0, 0, 0, 255))
        frames = composite_frames(
            background=background, radar_overlays=overlays,
            lat=-33.701, lon=151.209, radius_km=128,
        )
        gif_bytes = render_gif(frames)

        non_black = _count_non_black_pixels(gif_bytes)
        total = 640 * 640
        assert non_black / total < 0.05, (
            f"Radar-only dry render has {non_black / total:.1%} non-black pixels"
        )

    @pytest.mark.asyncio
    async def test_map_background_is_mostly_non_black(self):
        """Map crop should contain visible map content."""
        session = _make_mock_session()
        map_crop = await fetch_map_crop(
            session=session, lat=-33.701, lon=151.209, radius_km=128,
        )
        arr = np.array(map_crop.convert("RGB"))
        non_black = (arr.max(axis=2) > 10).sum()
        total = arr.shape[0] * arr.shape[1]
        assert non_black / total > 0.5, "Map background should be mostly non-black"

    @pytest.mark.asyncio
    async def test_full_composite_includes_map(self):
        """render_animated_composite produces a full map+radar image."""
        session = _make_mock_session()
        gif_bytes = await render_animated_composite(
            lat=-33.701, lon=151.209, radius_km=128,
            frame_paths=["/v2/radar/test_frame"],
            session=session,
        )
        non_black = _count_non_black_pixels(gif_bytes)
        total = 640 * 640
        assert non_black / total > 0.5, "Full composite should include map"

    @pytest.mark.asyncio
    async def test_render_gif_produces_valid_gif(self):
        """render_gif returns valid GIF bytes."""
        frame = Image.new("RGB", (100, 100), (255, 0, 0))
        gif_bytes = render_gif([frame])
        assert gif_bytes[:6] in (b"GIF87a", b"GIF89a")

    @pytest.mark.asyncio
    async def test_composite_frames_preserves_frame_count(self):
        """composite_frames returns one frame per radar overlay."""
        session = _make_mock_session()
        overlays = await fetch_radar_overlays(
            session=session, lat=-33.701, lon=151.209, radius_km=128,
            frame_paths=["/v2/radar/f1", "/v2/radar/f2", "/v2/radar/f3"],
        )
        background = Image.new("RGBA", (640, 640), (0, 0, 0, 255))
        frames = composite_frames(
            background=background, radar_overlays=overlays,
            lat=-33.701, lon=151.209, radius_km=128,
        )
        assert len(frames) == 3

    def test_filter_keeps_production_palette_pixels(self):
        """filter_precipitation_pixels must preserve pixels matching PRECIP_COLOURS.

        Regression test for a hardcoded palette divergence bug: a test helper
        invented 11 RGB values labelled as the production palette, but only one
        overlapped with the real palette. This test imports the production palette
        directly and verifies the renderer filter keeps pixels that match it.
        """
        from custom_components.rain_incoming.providers.rainviewer import PRECIP_COLOURS
        from custom_components.rain_incoming.radar.composite import filter_precipitation_pixels

        # Build a 4x4 RGBA image: row 0 has non-precipitation grey pixels,
        # row 1 has the first three production palette colours.
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = [128, 128, 128, 255]  # grey - not a precipitation colour, must be filtered OUT
        for i, (r, g, b, _) in enumerate(PRECIP_COLOURS[:3]):
            arr[1, i] = [r, g, b, 255]  # production palette colour, must be KEPT

        result = filter_precipitation_pixels(arr)

        # The grey pixel is not a precipitation colour - its alpha must be zeroed
        assert result[0, 0, 3] == 0, (
            f"Non-precipitation grey pixel should be filtered out, got alpha={result[0, 0, 3]}"
        )
        # Each of the three production palette pixels must survive with non-zero alpha
        for i, (r, g, b, _) in enumerate(PRECIP_COLOURS[:3]):
            assert result[1, i, 3] > 0, (
                f"Production palette pixel PRECIP_COLOURS[{i}] ({r},{g},{b}) "
                f"should survive filter, got alpha={result[1, i, 3]}"
            )

    def test_palette_roundtrip_parser_filter_render(self):
        """Parser-to-render roundtrip guard against palette drift.

        Uses HARDCODED production palette values rather than reading from
        PRECIP_COLOURS at runtime, so a silent palette change can't regenerate
        a matching synthetic tile. The sanity check catches the "palette changed,
        test not updated" case with a descriptive message.
        """
        from custom_components.rain_incoming.providers.rainviewer import (
            PRECIP_COLOURS,
            TILE_SIZE,
            _tile_to_intensity_array,
        )
        from custom_components.rain_incoming.radar.composite import (
            filter_precipitation_pixels,
            render_gif,
        )
        from tests.e2e.image_helpers import gif_has_precipitation_pixels

        # PRECIP_COLOURS[3] - moderate rain. Must be updated in both files together.
        _TILE_R, _TILE_G, _TILE_B = 81, 197, 232
        _EXPECTED_INTENSITY = 0.38

        tile_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (_TILE_R, _TILE_G, _TILE_B, 255))
        buf = BytesIO()
        tile_img.save(buf, format="PNG")
        tile_bytes = buf.getvalue()

        matching = [
            (r, g, b, intensity)
            for r, g, b, intensity in PRECIP_COLOURS
            if abs(r - _TILE_R) < 2 and abs(g - _TILE_G) < 2 and abs(b - _TILE_B) < 2
        ]
        assert matching, (
            f"Production PRECIP_COLOURS no longer contains the hardcoded tile colour "
            f"({_TILE_R},{_TILE_G},{_TILE_B}) - palette has diverged. "
            f"Update the hardcoded constants in this test AND in PRECIP_COLOURS "
            f"to keep them in sync."
        )

        intensity_array = _tile_to_intensity_array(tile_bytes)

        assert intensity_array.shape == (TILE_SIZE, TILE_SIZE), (
            f"Expected intensity array shape ({TILE_SIZE}, {TILE_SIZE}), "
            f"got {intensity_array.shape}"
        )
        assert np.allclose(intensity_array, _EXPECTED_INTENSITY, atol=0.01), (
            f"Parser should produce intensity ~{_EXPECTED_INTENSITY} for "
            f"the hardcoded production colour ({_TILE_R},{_TILE_G},{_TILE_B}), "
            f"got mean={intensity_array.mean():.4f}. "
            f"Check PRECIP_COLOURS in rainviewer.py still contains this entry."
        )

        rgba_array = np.array(Image.open(BytesIO(tile_bytes)), dtype=np.uint8)
        filtered = filter_precipitation_pixels(rgba_array)

        assert (filtered[:, :, 3] > 0).all(), (
            f"filter_precipitation_pixels should preserve every pixel since the "
            f"tile is filled with production colour ({_TILE_R},{_TILE_G},{_TILE_B}). "
            f"Check _PRECIP_RGB in composite.py is derived from PRECIP_COLOURS."
        )

        filtered_img = Image.fromarray(filtered, mode="RGBA")
        gif_bytes = render_gif([filtered_img])

        has_precip, fraction = gif_has_precipitation_pixels(gif_bytes)
        assert has_precip, (
            f"Rendered GIF from production colour ({_TILE_R},{_TILE_G},{_TILE_B}) "
            f"should be detected as precipitation by gif_has_precipitation_pixels, "
            f"got has_precip={has_precip}, fraction={fraction:.4f}"
        )
        assert fraction > 0.9, (
            f"GIF filled entirely with a production palette colour should have "
            f"~100% precipitation pixels (>0.9 after GIF quantisation fringe), "
            f"got fraction={fraction:.4f}"
        )

    def test_renderer_filter_at_least_as_strict_as_parser(self):
        """The renderer filter threshold must be <= the parser threshold.

        Both thresholds gate pixel matching against the same PRECIP_COLOURS
        palette. An inverted asymmetry would make rendered images show
        precipitation pixels the detector never assigned intensity to.
        """
        from custom_components.rain_incoming.providers.rainviewer import MAX_COLOUR_DISTANCE
        from custom_components.rain_incoming.radar.composite import _FILTER_MAX_COLOUR_DISTANCE

        assert _FILTER_MAX_COLOUR_DISTANCE <= MAX_COLOUR_DISTANCE, (
            f"Renderer filter threshold ({_FILTER_MAX_COLOUR_DISTANCE}) must be <= "
            f"parser threshold ({MAX_COLOUR_DISTANCE}). An inverted asymmetry would "
            f"make rendered images show precipitation pixels the detector never "
            f"recognised. See composite.py and rainviewer.py."
        )

    def test_parser_accepts_filter_rejects_in_threshold_gap(self):
        """A pixel inside the (renderer, parser) tolerance gap must be parser-accepted and filter-rejected.

        Proves both thresholds are active. If someone tightens parser or loosens
        filter such that the gap collapses, this test fails.
        """
        from custom_components.rain_incoming.providers.rainviewer import (
            MAX_COLOUR_DISTANCE,
            PRECIP_COLOURS,
            TILE_SIZE,
            _tile_to_intensity_array,
        )
        from custom_components.rain_incoming.radar.composite import (
            _FILTER_MAX_COLOUR_DISTANCE,
            filter_precipitation_pixels,
        )

        # Offset PRECIP_COLOURS[0] red channel by +40 - lands at L2=40 from
        # the source entry, between the filter (30) and parser (60) thresholds.
        # If palette changes such that the offset no longer lands in the gap,
        # the sanity assertion below will tell you to update this offset.
        base_r, base_g, base_b, _ = PRECIP_COLOURS[0]
        offset = 40
        gap_r, gap_g, gap_b = base_r + offset, base_g, base_b

        min_dist = min(
            ((gap_r - r) ** 2 + (gap_g - g) ** 2 + (gap_b - b) ** 2) ** 0.5
            for r, g, b, _ in PRECIP_COLOURS
        )
        assert _FILTER_MAX_COLOUR_DISTANCE < min_dist <= MAX_COLOUR_DISTANCE, (
            f"Synthetic pixel ({gap_r},{gap_g},{gap_b}) has min palette distance "
            f"{min_dist:.2f}, which is not in the gap "
            f"({_FILTER_MAX_COLOUR_DISTANCE}, {MAX_COLOUR_DISTANCE}]. "
            f"Update the offset in this test to keep the gap property."
        )

        tile_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (gap_r, gap_g, gap_b, 255))
        buf = BytesIO()
        tile_img.save(buf, format="PNG")
        intensity_array = _tile_to_intensity_array(buf.getvalue())

        assert (intensity_array > 0).all(), (
            f"Parser must accept gap pixel ({gap_r},{gap_g},{gap_b}) at distance "
            f"{min_dist:.2f} from the nearest palette entry - within "
            f"MAX_COLOUR_DISTANCE ({MAX_COLOUR_DISTANCE})."
        )

        rgba_array = np.array(Image.open(BytesIO(buf.getvalue())), dtype=np.uint8)
        filtered = filter_precipitation_pixels(rgba_array)

        assert (filtered[:, :, 3] == 0).all(), (
            f"Renderer filter must reject gap pixel ({gap_r},{gap_g},{gap_b}) at "
            f"distance {min_dist:.2f} from the nearest palette entry - exceeds "
            f"_FILTER_MAX_COLOUR_DISTANCE ({_FILTER_MAX_COLOUR_DISTANCE})."
        )

    def test_parser_rejects_beyond_max_colour_distance(self):
        """Parser must return intensity 0 for pixels beyond MAX_COLOUR_DISTANCE.

        Complement to test_parser_accepts_filter_rejects_in_threshold_gap:
        proves the parser reads the threshold, not just that it accepts
        in-range pixels.
        """
        from custom_components.rain_incoming.providers.rainviewer import (
            MAX_COLOUR_DISTANCE,
            PRECIP_COLOURS,
            TILE_SIZE,
            _tile_to_intensity_array,
        )

        # Offset PRECIP_COLOURS[0] red channel by +100 - lands at L2=100 from
        # the source entry, well beyond the parser threshold (60). If palette
        # changes push a nearer entry within 60 of this pixel, the sanity
        # assertion below will tell you to update the offset.
        base_r, base_g, base_b, _ = PRECIP_COLOURS[0]
        offset = 100
        far_r, far_g, far_b = base_r + offset, base_g, base_b

        min_dist = min(
            ((far_r - r) ** 2 + (far_g - g) ** 2 + (far_b - b) ** 2) ** 0.5
            for r, g, b, _ in PRECIP_COLOURS
        )
        assert min_dist > MAX_COLOUR_DISTANCE, (
            f"Synthetic pixel ({far_r},{far_g},{far_b}) has min palette distance "
            f"{min_dist:.2f}, which is not beyond MAX_COLOUR_DISTANCE ({MAX_COLOUR_DISTANCE}). "
            f"Update the offset in this test to keep the out-of-range property."
        )

        tile_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (far_r, far_g, far_b, 255))
        buf = BytesIO()
        tile_img.save(buf, format="PNG")
        intensity_array = _tile_to_intensity_array(buf.getvalue())

        assert (intensity_array == 0).all(), (
            f"Parser must reject pixel ({far_r},{far_g},{far_b}) at distance "
            f"{min_dist:.2f} from the nearest palette entry - exceeds "
            f"MAX_COLOUR_DISTANCE ({MAX_COLOUR_DISTANCE})."
        )

    @pytest.mark.asyncio
    async def test_dry_radar_over_black_has_zero_precipitation_pixels(self):
        """Dry radar (transparent tiles) composited over black must have
        zero precipitation-coloured pixels.

        This is the unit-level equivalent of the E2E dry-frame check,
        but without the CartoDB map background that causes false positives
        from vegetation greens matching precipitation colour thresholds.
        """
        session = _make_mock_session()

        overlays = await fetch_radar_overlays(
            session=session, lat=-33.701, lon=151.209, radius_km=128,
            frame_paths=["/v2/radar/dry_frame"],
        )
        background = Image.new("RGBA", (640, 640), (0, 0, 0, 255))
        frames = composite_frames(
            background=background, radar_overlays=overlays,
            lat=-33.701, lon=151.209, radius_km=128,
        )
        gif_bytes = render_gif(frames)

        count, fraction = _count_precipitation_pixels(gif_bytes)
        assert fraction < 0.005, (
            f"Dry radar over black has {fraction:.2%} precipitation pixels ({count} px) - "
            f"annotations or rendering artifacts are matching precipitation colours"
        )
