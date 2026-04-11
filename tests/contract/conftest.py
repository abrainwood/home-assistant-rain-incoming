"""Contract tests hit real APIs - override HA test framework fixtures."""
from __future__ import annotations

import copy

import pytest
import pytest_asyncio
import pytest_socket

from ._session import make_rainviewer_session

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations():
    """Override root conftest - contract tests don't use HA."""
    import socket as _socket
    pytest_socket.enable_socket()
    if hasattr(pytest_socket, "_true_connect"):
        _socket.socket.connect = pytest_socket._true_connect
    yield


@pytest.fixture
def verify_cleanup():
    """Override HA's verify_cleanup fixture - not applicable here."""
    yield


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def manifest() -> dict:
    """Fetch the RainViewer manifest once per test session.

    All contract tests that need manifest data share this single fetch,
    reducing live-API calls and rate-limit exposure.

    Enables real sockets here because this session fixture runs before the
    function-scoped auto_enable_custom_integrations fixture.
    """
    import socket as _socket
    pytest_socket.enable_socket()
    if hasattr(pytest_socket, "_true_connect"):
        _socket.socket.connect = pytest_socket._true_connect

    async with make_rainviewer_session() as session:
        async with session.get(MANIFEST_URL) as resp:
            assert resp.status == 200, (
                f"Manifest fetch failed: HTTP {resp.status} from {MANIFEST_URL}"
            )
            data = await resp.json(content_type=None)
            return copy.deepcopy(data)
