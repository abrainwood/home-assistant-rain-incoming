"""
Mock RainViewer server for E2E testing.

Serves radar manifests and tile PNGs. The active scenario controls what tiles
look like - transparent (no rain) or coloured (rain).

Scenarios:
  no_rain                      - all tiles transparent
  rain_everywhere              - all tiles filled with light-moderate precipitation
  rain_approaching             - rain cell moving east toward the location (tile 117,76)
  golden:<location>:<start>-<end>       - serve golden data tiles (numpy grids) for a location/frame range
  golden_v2:<location>:<start>-<end>    - serve REAL bronze tile PNGs for a location/frame range
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web
from PIL import Image

from tests.e2e.tile_converter import grid_to_tile_png

_logger = logging.getLogger(__name__)

GOLDEN_DATA_DIR = Path(__file__).parent.parent / "fixtures" / "golden_data"
GOLDEN_V2_DIR = Path(__file__).parent.parent / "fixtures" / "golden_v2"

# RainViewer scheme 6 light-moderate rain colour (intensity ~0.28)
_RAIN_COLOUR = (0, 154, 213, 255)

# Location tile at zoom 7 (Terry Hills)
_LOCATION_TILE_X = 117

_current_scenario = "no_rain"
_manifest_timestamps: list[int] = []

# Golden data state (v1 - numpy grids)
_golden_tiles: dict[int, bytes] = {}  # timestamp -> tile PNG bytes
_golden_manifest_frames: list[dict] = []
_golden_no_rain_tile: bytes | None = None

# Golden v2 state - real bronze tile PNGs
# Key: (timestamp, zoom, x, y, scheme) -> PNG bytes
_golden_v2_tiles: dict[tuple[int, int, int, int, int], bytes] = {}
_golden_v2_manifest_frames: list[dict] = []
_golden_v2_no_rain_tile: bytes | None = None


def _load_golden_scenario(location: str, start: int, end: int) -> None:
    """Load golden data grids and convert to tile PNGs for serving."""
    global _golden_tiles, _golden_manifest_frames, _golden_no_rain_tile

    grids_path = GOLDEN_DATA_DIR / f"grids_20260409_{location}_2hr.npz"
    manifest_path = GOLDEN_DATA_DIR / f"manifest_20260409_{location}_2hr.json"

    if not grids_path.exists():
        _logger.error("Golden data not found: %s", grids_path)
        return
    if not manifest_path.exists():
        _logger.error("Golden manifest not found: %s", manifest_path)
        return

    data = np.load(grids_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    _golden_tiles = {}
    _golden_manifest_frames = []
    for idx in range(start, end + 1):
        frame_key = f"frame_{idx}"
        if frame_key not in data:
            _logger.warning("Frame %s not found in %s", frame_key, grids_path)
            continue
        grid = data[frame_key]
        _golden_tiles[manifest["frames"][idx]["time"]] = grid_to_tile_png(grid)
        _golden_manifest_frames.append(manifest["frames"][idx])

    # Pre-build an empty tile for non-golden requests
    _golden_no_rain_tile = _make_tile(None)
    _logger.info(
        "Loaded golden scenario: %s frames %d-%d (%d tiles)",
        location, start, end, len(_golden_tiles),
    )


def _load_golden_v2_scenario(location: str, start: int, end: int) -> None:
    """Load real bronze tile PNGs for serving."""
    global _golden_v2_tiles, _golden_v2_manifest_frames, _golden_v2_no_rain_tile
    import re

    location_dir = GOLDEN_V2_DIR / location
    manifest_path = location_dir / "bronze" / "manifest.json"
    capture_info_path = location_dir / "bronze" / "capture_info.json"
    tiles_dir = location_dir / "bronze" / "tiles"

    if not manifest_path.exists():
        _logger.error("Golden v2 manifest not found: %s", manifest_path)
        return
    if not capture_info_path.exists():
        _logger.error("Golden v2 capture_info not found: %s", capture_info_path)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(capture_info_path) as f:
        capture_info = json.load(f)

    past_frames = manifest.get("radar", {}).get("past", [])

    _golden_v2_tiles = {}
    _golden_v2_manifest_frames = []

    for idx in range(start, end + 1):
        if idx >= len(past_frames):
            _logger.warning("Frame index %d out of range (max %d)", idx, len(past_frames) - 1)
            continue
        frame = past_frames[idx]
        ts = frame["time"]
        ts_dir = tiles_dir / str(ts)
        if not ts_dir.exists():
            _logger.warning("No tile dir for timestamp %d at %s", ts, ts_dir)
            continue

        # Load all tile PNGs from this timestamp directory
        # Filename pattern: {zoom}_{x}_{y}_s{scheme}.png
        tile_pattern = re.compile(r"(\d+)_(\d+)_(\d+)_s(\d+)\.png")
        tile_count = 0
        for tile_file in ts_dir.iterdir():
            m = tile_pattern.match(tile_file.name)
            if m:
                zoom = int(m.group(1))
                x = int(m.group(2))
                y = int(m.group(3))
                scheme = int(m.group(4))
                _golden_v2_tiles[(ts, zoom, x, y, scheme)] = tile_file.read_bytes()
                tile_count += 1

        _golden_v2_manifest_frames.append(frame)
        _logger.debug("Loaded %d tiles for frame %d (ts=%d)", tile_count, idx, ts)

    # Pre-build an empty tile for non-matching requests
    _golden_v2_no_rain_tile = _make_tile(None)
    _logger.info(
        "Loaded golden_v2 scenario: %s frames %d-%d (%d frames, %d total tiles)",
        location, start, end, len(_golden_v2_manifest_frames), len(_golden_v2_tiles),
    )


def _make_tile(rgba: tuple[int, int, int, int] | None) -> bytes:
    if rgba is None:
        arr = np.zeros((256, 256, 4), dtype=np.uint8)
    else:
        arr = np.full((256, 256, 4), rgba, dtype=np.uint8)
    img = Image.fromarray(arr, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


_NO_RAIN_TILE = _make_tile(None)
_RAIN_TILE = _make_tile(_RAIN_COLOUR)


def _parse_tile_url(tail: str) -> tuple[int, int]:
    """Extract (timestamp, tile_x) from a tile request path.

    Path format: v2/radar/{ts}/{tile_size}/{z}/{x}/{y}/{color}/{smooth}.png
    """
    parts = tail.rstrip(".png").split("/")
    # parts: ['v2', 'radar', ts, tile_size, z, x, y, color, smooth]
    ts = int(parts[2])
    x = int(parts[5])
    return ts, x


def _parse_tile_url_full(tail: str) -> tuple[str, int, int, int, int]:
    """Extract (frame_path_segment, zoom, x, y, scheme) from tile URL.

    Path format: v2/radar/{path_hash}/{tile_size}/{z}/{x}/{y}/{color}/{smooth}.png
    Returns frame_path_segment like "v2/radar/{path_hash}", plus tile params.
    """
    parts = tail.rstrip(".png").split("/")
    # parts: ['v2', 'radar', path_hash, tile_size, z, x, y, color, smooth]
    frame_path_segment = "/".join(parts[:3])  # "v2/radar/{path_hash}"
    z = int(parts[4])
    x = int(parts[5])
    y = int(parts[6])
    scheme = int(parts[7])
    return frame_path_segment, z, x, y, scheme


def _approaching_tile(ts: int, tile_x: int) -> bytes:
    """Determine whether a tile has rain in the 'approaching' scenario.

    Simulates a rain cell moving east toward the location:
    - Frame 0 (oldest): rain in tiles x <= location - 1
    - Frame 1:          rain in tiles x <= location (at location)
    - Frame 2 (newest): rain in tiles x <= location (at location)

    The cell front advances one tile between frames 0 and 1, reaching
    the location tile. Rain is visible in the radar image and detected
    as approaching/overhead.
    """
    if ts not in _manifest_timestamps:
        return _NO_RAIN_TILE

    frame_idx = _manifest_timestamps.index(ts)
    rain_max_x = _LOCATION_TILE_X - 1 + min(frame_idx, 1)  # -1, 0, 0

    return _RAIN_TILE if tile_x <= rain_max_x else _NO_RAIN_TILE


async def handle_manifest(request: web.Request) -> web.Response:
    global _manifest_timestamps

    if _current_scenario.startswith("golden_v2:") and _golden_v2_manifest_frames:
        # Serve golden v2 manifest with original timestamps and paths
        frames = [
            {"time": f["time"], "path": f["path"]}
            for f in _golden_v2_manifest_frames
        ]
        _manifest_timestamps = [f["time"] for f in _golden_v2_manifest_frames]
        manifest = {
            "version": "2.0",
            "generated": int(time.time()),
            "host": "",
            "radar": {"past": frames, "nowcast": []},
        }
        return web.json_response(manifest)

    if _current_scenario.startswith("golden:") and _golden_manifest_frames:
        # Serve golden data manifest with original timestamps
        frames = [
            {"time": f["time"], "path": f["path"]}
            for f in _golden_manifest_frames
        ]
        _manifest_timestamps = [f["time"] for f in _golden_manifest_frames]
        manifest = {
            "version": "2.0",
            "generated": int(time.time()),
            "host": "",
            "radar": {"past": frames, "nowcast": []},
        }
        return web.json_response(manifest)

    now = int(time.time())
    _manifest_timestamps = [now - 1200, now - 600, now]
    frames = [
        {"time": ts, "path": f"/v2/radar/{ts}"}
        for ts in _manifest_timestamps
    ]
    manifest = {
        "version": "2.0",
        "generated": now,
        "host": "",
        "radar": {"past": frames, "nowcast": []},
    }
    return web.json_response(manifest)


async def handle_tile(request: web.Request) -> web.Response:
    if _current_scenario.startswith("golden_v2:") and _golden_v2_manifest_frames:
        tail = request.match_info["tail"]
        # Parse the full tile URL to get frame path, zoom, x, y, scheme
        try:
            frame_path_seg, zoom, x, y, scheme = _parse_tile_url_full(tail)
        except (ValueError, IndexError):
            _logger.warning("Failed to parse golden_v2 tile URL: %s", tail)
            return web.Response(
                body=_golden_v2_no_rain_tile or _NO_RAIN_TILE,
                content_type="image/png",
            )

        # Find the matching frame by path
        for frame in _golden_v2_manifest_frames:
            frame_path_part = frame["path"].lstrip("/")
            if frame_path_seg == frame_path_part:
                ts = frame["time"]
                tile_data = _golden_v2_tiles.get((ts, zoom, x, y, scheme))
                if tile_data:
                    return web.Response(body=tile_data, content_type="image/png")
                # Tile coordinates not in bronze data - serve transparent
                _logger.debug(
                    "No bronze tile for ts=%d z=%d x=%d y=%d s=%d",
                    ts, zoom, x, y, scheme,
                )
                break
        return web.Response(
            body=_golden_v2_no_rain_tile or _NO_RAIN_TILE,
            content_type="image/png",
        )

    if _current_scenario.startswith("golden:") and _golden_tiles:
        tail = request.match_info["tail"]
        # Match the tile request path against golden data frame paths
        # tail is like "v2/radar/44f727a8d236/256/7/117/76/6/0.png"
        # frame["path"] is like "/v2/radar/44f727a8d236"
        for frame in _golden_manifest_frames:
            frame_path_part = frame["path"].lstrip("/")
            if tail.startswith(frame_path_part):
                tile_data = _golden_tiles.get(frame["time"])
                if tile_data:
                    return web.Response(body=tile_data, content_type="image/png")
                break
        # No matching golden frame - serve empty tile
        return web.Response(body=_golden_no_rain_tile or _NO_RAIN_TILE, content_type="image/png")

    if _current_scenario == "rain_everywhere":
        return web.Response(body=_RAIN_TILE, content_type="image/png")

    if _current_scenario == "rain_approaching":
        ts, tile_x = _parse_tile_url(request.match_info["tail"])
        tile = _approaching_tile(ts, tile_x)
        return web.Response(body=tile, content_type="image/png")

    return web.Response(body=_NO_RAIN_TILE, content_type="image/png")


async def handle_set_scenario(request: web.Request) -> web.Response:
    global _current_scenario
    data = await request.json()
    scenario = data["scenario"]
    _current_scenario = scenario

    # Parse golden v2 scenario: "golden_v2:<location>:<start>-<end>"
    if scenario.startswith("golden_v2:"):
        parts = scenario.split(":")
        if len(parts) == 3:
            location = parts[1]
            frame_range = parts[2]
            start, end = frame_range.split("-")
            _load_golden_v2_scenario(location, int(start), int(end))

    # Parse golden data scenario: "golden:<location>:<start>-<end>"
    elif scenario.startswith("golden:"):
        parts = scenario.split(":")
        if len(parts) == 3:
            location = parts[1]
            frame_range = parts[2]
            start, end = frame_range.split("-")
            _load_golden_scenario(location, int(start), int(end))

    return web.json_response({"status": "ok", "scenario": _current_scenario})


async def handle_get_scenario(request: web.Request) -> web.Response:
    return web.json_response({"scenario": _current_scenario})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/public/weather-maps.json", handle_manifest)
    app.router.add_post("/__scenario", handle_set_scenario)
    app.router.add_get("/__scenario", handle_get_scenario)
    app.router.add_get("/{tail:.+\\.png}", handle_tile)
    return app


def start_in_background(port: int = 9876) -> threading.Thread:
    """Start the mock server in a daemon thread. Returns immediately."""
    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = create_app()
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "0.0.0.0", port)
        loop.run_until_complete(site.start())
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    web.run_app(create_app(), port=9876)
