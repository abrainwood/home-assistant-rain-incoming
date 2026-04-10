"""Shared HTTP retry helper with backoff and rate-limit awareness."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

_LOGGER = logging.getLogger(__name__)

# Default concurrency limit for tile fetches
DEFAULT_CONCURRENT_LIMIT = 10


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> aiohttp.ClientResponse:
    """Fetch a URL with retry on 429/5xx, respecting Retry-After headers.

    Returns the successful response. Raises aiohttp.ClientResponseError if all
    retries are exhausted or a non-retryable error status is returned.
    """
    request_timeout = aiohttp.ClientTimeout(total=30)
    for attempt in range(max_retries + 1):
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
