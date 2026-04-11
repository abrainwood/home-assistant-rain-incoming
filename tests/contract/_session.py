"""Shared aiohttp session factory for contract tests."""
from __future__ import annotations

import aiohttp
import aiohttp.resolver


def make_rainviewer_session() -> aiohttp.ClientSession:
    """Create a session using the threaded resolver (avoids aiodns compat issues)."""
    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    return aiohttp.ClientSession(connector=connector)
