"""Tests for the shared HTTP retry helper."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.incoming_rain.http_retry import fetch_with_retry


def _make_response(status: int, headers: dict | None = None) -> MagicMock:
    """Create a mock aiohttp response with the given status."""
    resp = MagicMock(spec=aiohttp.ClientResponse)
    resp.status = status
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        request_info = MagicMock()
        request_info.url = "http://example.com/test"
        resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=request_info,
            history=(),
            status=status,
            message=f"HTTP {status}",
        )
    return resp


class TestFetchWithRetry:
    """Tests for fetch_with_retry."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        ok_resp = _make_response(200)
        session.get = AsyncMock(return_value=ok_resp)

        result = await fetch_with_retry(session, "http://example.com/test")
        assert result is ok_resp
        assert session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        rate_limited = _make_response(429)
        ok_resp = _make_response(200)

        session.get = AsyncMock(side_effect=[rate_limited, rate_limited, ok_resp])

        with patch("custom_components.incoming_rain.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_with_retry(
                session, "http://example.com/test", max_retries=3, base_delay=0.01,
            )
        assert result is ok_resp
        assert session.get.call_count == 3

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        rate_limited = _make_response(429, headers={"Retry-After": "2.5"})
        ok_resp = _make_response(200)

        session.get = AsyncMock(side_effect=[rate_limited, ok_resp])

        sleep_delays = []

        async def mock_sleep(delay):
            sleep_delays.append(delay)

        with patch("custom_components.incoming_rain.http_retry.asyncio.sleep", side_effect=mock_sleep):
            result = await fetch_with_retry(
                session, "http://example.com/test", max_retries=3, base_delay=1.0,
            )
        assert result is ok_resp
        assert len(sleep_delays) == 1
        assert sleep_delays[0] == 2.5

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_succeeds(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        server_error = _make_response(503)
        ok_resp = _make_response(200)

        session.get = AsyncMock(side_effect=[server_error, ok_resp])

        with patch("custom_components.incoming_rain.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_with_retry(
                session, "http://example.com/test", max_retries=3, base_delay=0.01,
            )
        assert result is ok_resp
        assert session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        server_error = _make_response(500)

        # max_retries=2 means 3 attempts total (initial + 2 retries)
        session.get = AsyncMock(return_value=server_error)

        with patch("custom_components.incoming_rain.http_retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await fetch_with_retry(
                    session, "http://example.com/test", max_retries=2, base_delay=0.01,
                )
            assert exc_info.value.status == 500

    @pytest.mark.asyncio
    async def test_does_not_retry_on_4xx_other_than_429(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        not_found = _make_response(404)

        session.get = AsyncMock(return_value=not_found)

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await fetch_with_retry(
                session, "http://example.com/test", max_retries=3, base_delay=0.01,
            )
        assert exc_info.value.status == 404
        # Should only have called get once - no retries
        assert session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        server_error = _make_response(500)

        session.get = AsyncMock(return_value=server_error)

        sleep_delays = []

        async def mock_sleep(delay):
            sleep_delays.append(delay)

        with patch("custom_components.incoming_rain.http_retry.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(aiohttp.ClientResponseError):
                await fetch_with_retry(
                    session, "http://example.com/test", max_retries=3, base_delay=1.0,
                )

        # 3 retries = 3 sleep calls with exponential backoff: 1, 2, 4
        assert len(sleep_delays) == 3
        assert sleep_delays[0] == 1.0
        assert sleep_delays[1] == 2.0
        assert sleep_delays[2] == 4.0

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Verify the rate-limited fetch respects the semaphore."""
        from custom_components.incoming_rain.http_retry import rate_limited_fetch

        session = AsyncMock(spec=aiohttp.ClientSession)
        ok_resp = _make_response(200)
        session.get = AsyncMock(return_value=ok_resp)

        semaphore = asyncio.Semaphore(2)
        max_concurrent = 0
        current_concurrent = 0

        original_acquire = semaphore.acquire
        original_release = semaphore.release

        async def tracking_acquire():
            nonlocal current_concurrent, max_concurrent
            result = await original_acquire()
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            return result

        def tracking_release():
            nonlocal current_concurrent
            current_concurrent -= 1
            return original_release()

        semaphore.acquire = tracking_acquire
        semaphore.release = tracking_release

        tasks = [
            rate_limited_fetch(session, f"http://example.com/{i}", semaphore=semaphore)
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert max_concurrent <= 2
