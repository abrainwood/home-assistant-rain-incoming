"""
Contract tests against the real RainViewer API.

These tests hit the live API to verify the contract hasn't changed.
They do NOT test our integration logic - just that the upstream API
still returns data in the shape we expect.

Run separately: make test-contract
Scheduled daily in CI so the build goes red if the API changes.
"""
from __future__ import annotations

from io import BytesIO

import aiohttp
import aiohttp.resolver
import numpy as np
import pytest
from PIL import Image

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_BASE_URL = "https://tilecache.rainviewer.com"
ZOOM = 7
TILE_X, TILE_Y = 117, 76  # Terry Hills
COLOUR_SCHEME = 6
TILE_SIZE = 256


def _session() -> aiohttp.ClientSession:
    """Create a session using the threaded resolver (avoids aiodns compat issues)."""
    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    return aiohttp.ClientSession(connector=connector)


@pytest.mark.asyncio
async def test_manifest_returns_valid_json():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            assert resp.status == 200
            data = await resp.json(content_type=None)
            assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_manifest_has_radar_past_frames():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            data = await resp.json(content_type=None)

    radar = data.get("radar")
    assert radar is not None, "Missing 'radar' key in manifest"

    past = radar.get("past")
    assert isinstance(past, list), f"radar.past should be a list, got {type(past)}"
    assert len(past) >= 2, f"Expected at least 2 past frames, got {len(past)}"


@pytest.mark.asyncio
async def test_frame_has_required_fields():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            data = await resp.json(content_type=None)

    frame = data["radar"]["past"][-1]
    assert "time" in frame, f"Frame missing 'time': {frame}"
    assert "path" in frame, f"Frame missing 'path': {frame}"
    assert isinstance(frame["time"], int)
    assert isinstance(frame["path"], str)


@pytest.mark.asyncio
async def test_tile_returns_valid_png():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            data = await resp.json(content_type=None)
        path = data["radar"]["past"][-1]["path"]

        tile_url = f"{TILE_BASE_URL}{path}/{TILE_SIZE}/{ZOOM}/{TILE_X}/{TILE_Y}/{COLOUR_SCHEME}/0.png"
        async with session.get(tile_url) as resp:
            assert resp.status == 200
            tile_bytes = await resp.read()

    img = Image.open(BytesIO(tile_bytes))
    assert img.size == (TILE_SIZE, TILE_SIZE)


@pytest.mark.asyncio
async def test_tile_is_rgba_decodable():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            data = await resp.json(content_type=None)
        path = data["radar"]["past"][-1]["path"]

        tile_url = f"{TILE_BASE_URL}{path}/{TILE_SIZE}/{ZOOM}/{TILE_X}/{TILE_Y}/{COLOUR_SCHEME}/0.png"
        async with session.get(tile_url) as resp:
            tile_bytes = await resp.read()

    img = Image.open(BytesIO(tile_bytes)).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    assert arr.shape == (TILE_SIZE, TILE_SIZE, 4)


@pytest.mark.asyncio
async def test_zoom_7_is_supported():
    async with _session() as session:
        async with session.get(MANIFEST_URL) as resp:
            data = await resp.json(content_type=None)
        path = data["radar"]["past"][-1]["path"]

        url = f"{TILE_BASE_URL}{path}/{TILE_SIZE}/{ZOOM}/{TILE_X}/{TILE_Y}/{COLOUR_SCHEME}/0.png"
        async with session.get(url) as resp:
            assert resp.status == 200
