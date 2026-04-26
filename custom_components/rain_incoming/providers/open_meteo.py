from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import aiohttp

_LOGGER = logging.getLogger(__name__)

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
    except aiohttp.ClientResponseError as e:
        _LOGGER.warning(
            "Open-Meteo precipitation fetch failed: HTTP %d for %s",
            e.status, OPEN_METEO_URL,
        )
        return None  # fail open - don't penalize if API is unreachable
    except Exception as e:
        _LOGGER.warning(
            "Open-Meteo precipitation fetch failed: %s: %s",
            type(e).__name__, e,
        )
        return None  # fail open - don't penalize if API is unreachable


async def fetch_hourly_pop(
    lat: float, lon: float, session: aiohttp.ClientSession
) -> list[tuple[datetime, float]] | None:
    """Fetch hourly precipitation probability from Open-Meteo. Returns None on failure."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation_probability",
    }
    try:
        async with session.get(
            OPEN_METEO_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            times = data["hourly"]["time"]
            probs = data["hourly"]["precipitation_probability"]
            return [
                (datetime.fromisoformat(t).replace(tzinfo=timezone.utc), float(p))
                for t, p in zip(times, probs)
            ]
    except aiohttp.ClientResponseError as e:
        _LOGGER.warning(
            "Open-Meteo hourly PoP fetch failed: HTTP %d for %s",
            e.status, OPEN_METEO_URL,
        )
        return None
    except Exception as e:
        _LOGGER.warning(
            "Open-Meteo hourly PoP fetch failed: %s: %s",
            type(e).__name__, e,
        )
        return None


def max_pop_in_window(
    forecast: list[tuple[datetime, float]],
    start: datetime,
    end: datetime,
) -> float | None:
    """Return max PoP from forecast entries within [start, end] inclusive."""
    return max(p for dt, p in forecast if start <= dt <= end)
