from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.rain_incoming.providers.open_meteo import (
    fetch_precipitation_now,
)


def _make_mock_session(response_json=None, status=200, raise_exc=None):
    """Build a mock aiohttp.ClientSession with a configurable GET response."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.raise_for_status = MagicMock()
    if status >= 400:
        mock_resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=status
        )
    if raise_exc is not None:
        mock_resp.__aenter__ = AsyncMock(side_effect=raise_exc)
    else:
        mock_resp.json = AsyncMock(return_value=response_json)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock(spec=aiohttp.ClientSession)
    session.get = MagicMock(return_value=mock_resp)
    return session


class TestFetchPrecipitationNow:
    def test_returns_precipitation_value(self):
        """Should extract precipitation from a valid API response."""
        session = _make_mock_session(
            response_json={"current": {"precipitation": 0.5, "time": "2026-04-08T10:00"}}
        )
        result = asyncio.get_event_loop().run_until_complete(
            fetch_precipitation_now(-33.7, 151.2, session)
        )
        assert result == 0.5

    def test_returns_none_on_network_error(self):
        """Network timeout should return None (fail open)."""
        session = _make_mock_session(raise_exc=asyncio.TimeoutError())
        result = asyncio.get_event_loop().run_until_complete(
            fetch_precipitation_now(-33.7, 151.2, session)
        )
        assert result is None

    def test_returns_none_on_invalid_json(self):
        """Garbage response (missing expected keys) should return None."""
        session = _make_mock_session(response_json={"garbage": True})
        result = asyncio.get_event_loop().run_until_complete(
            fetch_precipitation_now(-33.7, 151.2, session)
        )
        # "current" key is missing, so .get("current", {}).get("precipitation") -> None
        assert result is None
