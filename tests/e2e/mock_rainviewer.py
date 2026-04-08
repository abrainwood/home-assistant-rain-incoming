"""
Mock RainViewer server for E2E testing.

Serves radar manifests and tile PNGs. The active scenario controls what tiles
look like - transparent (no rain) or coloured (rain).

Scenarios:
  no_rain          - all tiles transparent
  rain_everywhere  - all tiles filled with light-moderate precipitation
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

_current_scenario = "no_rain"


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


async def handle_manifest(request: web.Request) -> web.Response:
    now = int(time.time())
    # Serve 3 frames at 10-minute intervals (matching real RainViewer cadence)
    frames = [
        {"time": now - 1200, "path": f"/v2/radar/{now - 1200}"},
        {"time": now - 600, "path": f"/v2/radar/{now - 600}"},
        {"time": now, "path": f"/v2/radar/{now}"},
    ]
    manifest = {
        "version": "2.0",
        "generated": now,
        "host": "",
        "radar": {"past": frames, "nowcast": []},
    }
    return web.json_response(manifest)


async def handle_tile(request: web.Request) -> web.Response:
    tile = _RAIN_TILE if _current_scenario == "rain_everywhere" else _NO_RAIN_TILE
    return web.Response(body=tile, content_type="image/png")


async def handle_set_scenario(request: web.Request) -> web.Response:
    global _current_scenario
    data = await request.json()
    _current_scenario = data["scenario"]
    return web.json_response({"status": "ok", "scenario": _current_scenario})


async def handle_get_scenario(request: web.Request) -> web.Response:
    return web.json_response({"scenario": _current_scenario})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/public/weather-maps.json", handle_manifest)
    app.router.add_post("/__scenario", handle_set_scenario)
    app.router.add_get("/__scenario", handle_get_scenario)
    # Catch-all for tile requests
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
