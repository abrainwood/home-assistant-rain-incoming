"""Contract tests hit real APIs - override HA test framework fixtures."""
from __future__ import annotations

import pytest
import pytest_socket


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
