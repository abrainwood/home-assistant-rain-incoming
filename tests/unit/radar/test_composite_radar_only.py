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
