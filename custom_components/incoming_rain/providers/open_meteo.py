from __future__ import annotations

import os

import aiohttp

OPEN_METEO_URL = os.environ.get(
    "OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast"
)


async def fetch_precipitation_now(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> float | None:
    """Fetch current precipitation in mm from Open-Meteo. Returns None on failure."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "precipitation",
    }
    try:
        async with session.get(
            OPEN_METEO_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("current", {}).get("precipitation")
    except Exception:
        return None  # fail open - don't penalize if API is unreachable
