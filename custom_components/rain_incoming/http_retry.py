"""Shared HTTP retry helper with backoff and rate-limit awareness."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

_LOGGER = logging.getLogger(__name__)

# Default concurrency limit for tile fetches
DEFAULT_CONCURRENT_LIMIT = 10

_UTILIZATION_THRESHOLD = 0.80


class RateLimitBudget:
    """Tracks RainViewer rate limit state from response headers."""

    def __init__(self) -> None:
        self._limit: int = 0
        self._used: int = 0
        self._window_seconds: float = 0.0
        self._burst_limit: int = 0
        self._burst_used: int = 0

    def update_from_headers(self, headers) -> None:
        """Parse x-ratelimit-* headers. Missing/malformed headers: no-op."""
        try:
            limit = int(headers["x-ratelimit-limit"])
            used = int(headers["x-ratelimit-used"])
            window = float(headers["x-ratelimit-window"])
        except (KeyError, ValueError, TypeError):
            return

        self._limit = limit
        self._used = used
        self._window_seconds = window

        try:
            self._burst_limit = int(headers["x-ratelimit-burst-limit"])
        except (KeyError, ValueError, TypeError):
            pass

        try:
            self._burst_used = int(headers["x-ratelimit-burst-used"])
        except (KeyError, ValueError, TypeError):
            pass

    @property
    def utilization(self) -> float:
        """used / limit, 0.0 if no data yet."""
        if self._limit == 0:
            return 0.0
        return self._used / self._limit

    @property
    def remaining(self) -> int:
        """limit - used, 0 if no data yet."""
        if self._limit == 0:
            return 0
        return max(0, self._limit - self._used)

    def suggested_delay(self) -> float:
        """Inter-request delay in seconds.

        - Under 80% utilization: 0.0
        - 80-100%: window_seconds / remaining (spread remaining budget evenly)
        - At limit (remaining=0): window_seconds (wait for reset)
        - No data yet (limit=0): 0.0
        """
        if self._limit == 0:
            return 0.0
        if self.utilization < _UTILIZATION_THRESHOLD:
            return 0.0
        remaining = self.remaining
        if remaining == 0:
            return self._window_seconds
        return self._window_seconds / remaining


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    budget: RateLimitBudget | None = None,
    timeout_total: float = 30.0,
) -> aiohttp.ClientResponse:
    """Fetch a URL with retry on 429/5xx, respecting Retry-After headers.

    Returns the successful response. Raises aiohttp.ClientResponseError if all
    retries are exhausted or a non-retryable error status is returned.

    timeout_total: per-request timeout in seconds. Default 30s. Tile fetches
        use 15s to fail fast when the tile server is slow.
    """
    request_timeout = aiohttp.ClientTimeout(total=timeout_total)
    for attempt in range(max_retries + 1):
        if budget is not None:
            delay = budget.suggested_delay()
            if delay > 0:
                _LOGGER.info(
                    "Rate limit pacing: sleeping %.2fs before request (utilization %.0f%%)",
                    delay, budget.utilization * 100,
                )
                await asyncio.sleep(delay)

        try:
            resp = await session.get(url, timeout=request_timeout)
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                _LOGGER.warning(
                    "Timeout on %s, retrying in %.1fs (attempt %d/%d)",
                    url, delay, attempt + 1, max_retries,
                )
                await asyncio.sleep(delay)
                continue
            _LOGGER.warning("All %d retries exhausted (timeout) for %s", max_retries, url)
            raise

        if budget is not None:
            budget.update_from_headers(resp.headers)

        if resp.status == 429:
            if attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                delay = base_delay * (2 ** attempt)
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        _LOGGER.debug(
                            "Non-numeric Retry-After header '%s', using exponential backoff",
                            retry_after,
                        )
                _LOGGER.warning(
                    "Rate limited (429) on %s, retrying in %.1fs (attempt %d/%d)",
                    url, delay, attempt + 1, max_retries,
                )
                resp.release()
                await asyncio.sleep(delay)
                continue
            # All retries exhausted on 429
            _LOGGER.warning(
                "All %d retries exhausted (429) for %s", max_retries, url,
            )
            resp.raise_for_status()

        if resp.status >= 500:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                _LOGGER.warning(
                    "Server error (%d) on %s, retrying in %.1fs (attempt %d/%d)",
                    resp.status, url, delay, attempt + 1, max_retries,
                )
                resp.release()
                await asyncio.sleep(delay)
                continue
            # All retries exhausted on 5xx
            _LOGGER.warning(
                "All %d retries exhausted (%d) for %s",
                max_retries, resp.status, url,
            )
            resp.raise_for_status()

        # Non-retryable status (success or 4xx other than 429)
        resp.raise_for_status()
        return resp

    # Should not reach here, but satisfy type checker
    raise RuntimeError("Unreachable: retry loop exited without return or raise")


async def rate_limited_fetch(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore | None = None,
    **kwargs,
) -> aiohttp.ClientResponse:
    """Fetch with retry, gated by a semaphore for concurrency control."""
    if semaphore is None:
        return await fetch_with_retry(session, url, **kwargs)
    async with semaphore:
        return await fetch_with_retry(session, url, **kwargs)
