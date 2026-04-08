"""
Mock RainViewer server for E2E testing.

Serves radar manifests and tile PNGs. The active scenario controls what tiles
look like - transparent (no rain) or coloured (rain).

Scenarios:
  no_rain          - all tiles transparent
  rain_everywhere  - all tiles filled with light-moderate precipitation
  rain_approaching - rain cell moving east toward the location (tile 117,76)
"""
from __future__ import annotations

import asyncio
import io
import threading
import time

import numpy as np
from aiohttp import web
from PIL import Image

# RainViewer scheme 6 light-moderate rain colour (intensity ~0.28)
_RAIN_COLOUR = (0, 154, 213, 255)

# Location tile at zoom 7 (Terry Hills)
_LOCATION_TILE_X = 117

_current_scenario = "no_rain"
_manifest_timestamps: list[int] = []


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
    _current_scenario = data["scenario"]
    return web.json_response({"status": "ok", "scenario": _current_scenario})


async def handle_get_scenario(request: web.Request) -> web.Response:
    return web.json_response({"scenario": _current_scenario})


async def handle_open_meteo(request: web.Request) -> web.Response:
    """Serve fake Open-Meteo current weather response."""
    if _current_scenario in ("rain_everywhere", "rain_approaching"):
        precip = 2.5
    else:
        precip = 0.0

    return web.json_response({
        "current": {"precipitation": precip},
        "current_units": {"precipitation": "mm"},
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/public/weather-maps.json", handle_manifest)
    app.router.add_post("/__scenario", handle_set_scenario)
    app.router.add_get("/__scenario", handle_get_scenario)
    app.router.add_get("/v1/forecast", handle_open_meteo)
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
