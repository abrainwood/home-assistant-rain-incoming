from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.rain_incoming.providers.open_meteo import (
    fetch_hourly_pop,
    fetch_precipitation_now,
    max_pop_in_window,
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
    @pytest.mark.asyncio
    async def test_returns_precipitation_value(self):
        """Should extract precipitation from a valid API response."""
        session = _make_mock_session(
            response_json={"current": {"precipitation": 0.5, "time": "2026-04-08T10:00"}}
        )
        result = await fetch_precipitation_now(-33.7, 151.2, session)
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self):
        """Network timeout should return None (fail open)."""
        session = _make_mock_session(raise_exc=asyncio.TimeoutError())
        result = await fetch_precipitation_now(-33.7, 151.2, session)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        """Garbage response (missing expected keys) should return None."""
        session = _make_mock_session(response_json={"garbage": True})
        result = await fetch_precipitation_now(-33.7, 151.2, session)
        # "current" key is missing, so .get("current", {}).get("precipitation") -> None
        assert result is None


class TestFetchHourlyPop:
    @pytest.mark.asyncio
    async def test_returns_list_of_datetime_float_tuples(self):
        """Valid Open-Meteo hourly response is parsed into (datetime, float) tuples."""
        session = _make_mock_session(
            response_json={
                "hourly": {
                    "time": ["2026-04-26T18:00", "2026-04-26T19:00", "2026-04-26T20:00"],
                    "precipitation_probability": [10, 25, 60],
                }
            }
        )
        result = await fetch_hourly_pop(-33.7, 151.2, session)

        assert result is not None
        assert len(result) == 3
        dt0, pop0 = result[0]
        dt1, pop1 = result[1]
        dt2, pop2 = result[2]
        assert isinstance(dt0, datetime)
        assert isinstance(pop0, float)
        assert dt0 == datetime(2026, 4, 26, 18, 0, tzinfo=timezone.utc)
        assert dt1 == datetime(2026, 4, 26, 19, 0, tzinfo=timezone.utc)
        assert dt2 == datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
        assert pop0 == 10.0
        assert pop1 == 25.0
        assert pop2 == 60.0

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, caplog):
        """HTTP 5xx error should return None and log a warning."""
        session = _make_mock_session(status=503)
        result = await fetch_hourly_pop(-33.7, 151.2, session)
        assert result is None
        assert any(
            "Open-Meteo hourly PoP" in record.message and "503" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_hourly_key(self, caplog):
        """200 OK with body missing 'hourly' key should return None and log a warning."""
        session = _make_mock_session(response_json={"garbage": True})
        result = await fetch_hourly_pop(-33.7, 151.2, session)
        assert result is None
        assert any(
            "Open-Meteo hourly PoP" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )


class TestMaxPopInWindow:
    def test_returns_max_pop_in_window(self):
        """Should return max PoP from entries within start/end window (inclusive)."""
        forecast = [
            (datetime(2026, 4, 26, 18, 0, tzinfo=timezone.utc), 10.0),
            (datetime(2026, 4, 26, 19, 0, tzinfo=timezone.utc), 25.0),
            (datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc), 60.0),
            (datetime(2026, 4, 26, 21, 0, tzinfo=timezone.utc), 35.0),
            (datetime(2026, 4, 26, 22, 0, tzinfo=timezone.utc), 5.0),
        ]
        start = datetime(2026, 4, 26, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 26, 21, 0, tzinfo=timezone.utc)

        result = max_pop_in_window(forecast, start, end)

        assert result == 60.0
