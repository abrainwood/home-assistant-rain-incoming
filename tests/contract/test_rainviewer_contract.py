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

import numpy as np
import pytest
from PIL import Image

from ._session import make_rainviewer_session

TILE_BASE_URL = "https://tilecache.rainviewer.com"
ZOOM = 7
TILE_X, TILE_Y = 117, 76  # Terry Hills
COLOUR_SCHEME = 6
TILE_SIZE = 256


@pytest.mark.asyncio
async def test_manifest_returns_valid_json(manifest):
    assert isinstance(manifest, dict)


@pytest.mark.asyncio
async def test_manifest_has_radar_past_frames(manifest):
    radar = manifest.get("radar")
    assert radar is not None, "Missing 'radar' key in manifest"

    past = radar.get("past")
    assert isinstance(past, list), f"radar.past should be a list, got {type(past)}"
    assert len(past) >= 2, f"Expected at least 2 past frames, got {len(past)}"


@pytest.mark.asyncio
async def test_frame_has_required_fields(manifest):
    frame = manifest["radar"]["past"][-1]
    assert "time" in frame, f"Frame missing 'time': {frame}"
    assert "path" in frame, f"Frame missing 'path': {frame}"
    assert isinstance(frame["time"], int)
    assert isinstance(frame["path"], str)


@pytest.mark.asyncio
async def test_tile_returns_valid_png(manifest):
    path = manifest["radar"]["past"][-1]["path"]
    tile_url = f"{TILE_BASE_URL}{path}/{TILE_SIZE}/{ZOOM}/{TILE_X}/{TILE_Y}/{COLOUR_SCHEME}/0.png"

    async with make_rainviewer_session() as session:
        async with session.get(tile_url) as resp:
            assert resp.status == 200
            tile_bytes = await resp.read()

    img = Image.open(BytesIO(tile_bytes))
    assert img.size == (TILE_SIZE, TILE_SIZE)


@pytest.mark.asyncio
async def test_tile_is_rgba_decodable(manifest):
    path = manifest["radar"]["past"][-1]["path"]
    tile_url = f"{TILE_BASE_URL}{path}/{TILE_SIZE}/{ZOOM}/{TILE_X}/{TILE_Y}/{COLOUR_SCHEME}/0.png"

    async with make_rainviewer_session() as session:
        async with session.get(tile_url) as resp:
            tile_bytes = await resp.read()

    img = Image.open(BytesIO(tile_bytes)).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    assert arr.shape == (TILE_SIZE, TILE_SIZE, 4)
