"""
E2E test fixtures.

Manages the mock RainViewer server, HA container lifecycle, and provides
a REST API client for interacting with the running HA instance.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest
import pytest_socket

# Add project root to path so we can import scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.e2e.mock_rainviewer import start_in_background


# --- Override root conftest fixtures that don't apply to E2E tests ---

def _restore_sockets():
    """Fully disable pytest-socket guards."""
    import socket as _socket
    pytest_socket.enable_socket()
    if hasattr(pytest_socket, "_true_connect"):
        _socket.socket.connect = pytest_socket._true_connect


@pytest.fixture(scope="session", autouse=True)
def _allow_sockets_session():
    """Enable sockets at session scope so session fixtures can use the network."""
    _restore_sockets()
    yield


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations():
    """
    Override root conftest (E2E tests don't use in-process HA) and re-enable
    sockets each test since pytest-socket re-patches between test functions.
    """
    _restore_sockets()
    yield

E2E_HA_PORT = 18123
HA_URL = f"http://localhost:{E2E_HA_PORT}"
MOCK_PORT = 9876
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".e2e-token")

# Docker env for E2E - isolated container, volume, and port to avoid clobbering dev state
_DOCKER_ENV = {
    "HA_CONTAINER": "ha-e2e",
    "HA_VOLUME": "ha-e2e-config",
    "HA_PORT": str(E2E_HA_PORT),
}


class HAClient:
    """Simple REST client for the HA instance."""

    def __init__(self, token: str) -> None:
        self.token = token

    def request(self, method: str, path: str, data: dict | None = None) -> dict | None:
        url = f"{HA_URL}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            raise RuntimeError(f"HTTP {e.code} on {method} {path}: {raw[:300]}") from e

    def get_state(self, entity_id: str) -> dict | None:
        try:
            return self.request("GET", f"/api/states/{entity_id}")
        except RuntimeError:
            return None

    def update_entity(self, entity_id: str) -> None:
        self.request("POST", "/api/services/homeassistant/update_entity",
                      {"entity_id": entity_id})

    def get_text(self, path: str) -> str:
        """Fetch a plain-text endpoint (e.g. /api/error_log)."""
        url = f"{HA_URL}{path}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()

    def get_image(self, entity_id: str) -> bytes | None:
        """Download an image entity's content via the HA image proxy."""
        url = f"{HA_URL}/api/image_proxy/{entity_id}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            return None

    def set_mock_scenario(self, scenario: str) -> None:
        """Switch the mock RainViewer server's active scenario."""
        body = json.dumps({"scenario": scenario}).encode()
        req = urllib.request.Request(
            f"http://localhost:{MOCK_PORT}/__scenario",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)


def _wait_for(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=2)
            return
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return  # server is up, just needs auth
            time.sleep(1)  # other errors - keep waiting
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {url}")


@pytest.fixture(scope="session")
def mock_server():
    """Start mock RainViewer server for the test session."""
    start_in_background(port=MOCK_PORT)
    _wait_for(f"http://localhost:{MOCK_PORT}/__scenario")
    yield


@pytest.fixture(scope="session")
def ha_container(mock_server):
    """Ensure HA container is running with mock RainViewer URLs."""
    env = {
        **os.environ,
        **_DOCKER_ENV,
        "RAINVIEWER_API_URL": f"http://host.docker.internal:{MOCK_PORT}",
        "RAINVIEWER_TILE_URL": f"http://host.docker.internal:{MOCK_PORT}",
    }

    # Always start fresh - remove old E2E container and volume for a clean slate
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.dev.yml", "down", "-v"],
        capture_output=True, env=env,
    )
    # Remove stale token
    try:
        os.remove(TOKEN_FILE)
    except FileNotFoundError:
        pass

    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.dev.yml", "up", "-d"],
        capture_output=True, env=env, check=True,
    )
    _wait_for(f"{HA_URL}/api/", timeout=120)
    # Give the integration time to load after HA startup
    time.sleep(10)
    yield

    # Don't tear down - leave container running for manual inspection


@pytest.fixture(scope="session")
def ha_client(ha_container) -> HAClient:
    """Onboard HA and return a REST client with a valid auth token."""
    # Try existing token first
    try:
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
        client = HAClient(token)
        client.request("GET", "/api/")
        return client
    except (FileNotFoundError, RuntimeError):
        pass

    # Need to onboard or log in
    def _raw_request(method, path, data=None, token=None, form=False):
        url = f"{HA_URL}{path}"
        headers = {}
        body = None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if data is not None:
            if form:
                body = urllib.parse.urlencode(data).encode()
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            else:
                body = json.dumps(data).encode()
                headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}

    def _get_token_via_login():
        """Log in with known credentials when onboarding is already done."""
        flow = _raw_request("POST", "/auth/login_flow", {
            "client_id": f"{HA_URL}/",
            "handler": ["homeassistant", None],
            "redirect_uri": f"{HA_URL}/",
        })
        result = _raw_request("POST", f"/auth/login_flow/{flow['flow_id']}", {
            "username": "dev", "password": "devdevdev",
            "client_id": f"{HA_URL}/",
        })
        token_resp = _raw_request("POST", "/auth/token", {
            "client_id": f"{HA_URL}/",
            "grant_type": "authorization_code",
            "code": result["result"],
        }, form=True)
        return token_resp["access_token"]

    # Check if onboarding is needed
    try:
        onboarding = _raw_request("GET", "/api/onboarding")
    except Exception:
        onboarding = None

    if onboarding and "user" in [s["step"] for s in onboarding]:
        # Fresh instance - create user
        resp = _raw_request("POST", "/api/onboarding/users", {
            "client_id": f"{HA_URL}/",
            "name": "E2E Test", "username": "dev",
            "password": "devdevdev", "language": "en",
        })
        token_resp = _raw_request("POST", "/auth/token", {
            "client_id": f"{HA_URL}/",
            "grant_type": "authorization_code",
            "code": resp["auth_code"],
        }, form=True)
        token = token_resp["access_token"]
        _raw_request("POST", "/api/onboarding/core_config", {
            "latitude": -33.701,
            "longitude": 151.209,
            "country": "AU",
            "time_zone": "Australia/Sydney",
            "elevation": 200,
            "unit_system": "metric",
            "currency": "AUD",
            "language": "en",
        }, token=token)
        _raw_request("POST", "/api/onboarding/analytics", {}, token=token)
    else:
        # Already onboarded - log in
        token = _get_token_via_login()

    with open(TOKEN_FILE, "w") as f:
        f.write(token)

    # Add integration
    flow = _raw_request("POST", "/api/config/config_entries/flow",
                         {"handler": "incoming_rain"}, token=token)
    _raw_request("POST", f"/api/config/config_entries/flow/{flow['flow_id']}", {
        "latitude": -33.701,
        "longitude": 151.209,
        "lookahead_minutes": 60,
    }, token=token)

    # Wait for coordinator to do its first update
    time.sleep(3)

    return HAClient(token)
