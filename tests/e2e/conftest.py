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

# docker compose prefixes named volumes with the project name (derived from the repo
# directory, "incoming_rain"). We need the full name for `docker volume rm` since that
# command is not compose-aware and won't apply the prefix automatically.
_E2E_DOCKER_VOLUME = f"incoming_rain_{_DOCKER_ENV['HA_VOLUME']}"


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
            with urllib.request.urlopen(req, timeout=120) as resp:
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

    def get_image(self, entity_id: str, max_retries: int = 5, delay: float = 5.0) -> bytes | None:
        """Download an image, retrying if the entity isn't ready yet."""
        for attempt in range(max_retries):
            try:
                url = f"{HA_URL}/api/image_proxy/{entity_id}"
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                    if data and len(data) > 100:
                        return data
                    print(f"  get_image attempt {attempt+1}: got {len(data)} bytes (too small)")
            except Exception as e:
                print(f"  get_image attempt {attempt+1}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
        return None

    def wait_for_coordinator_cycle(
        self,
        entity_id: str,
        timeout: float = 60.0,
        interval: float = 2.0,
    ) -> dict:
        """Trigger an entity update and wait for the coordinator to run a fresh cycle.

        Handles HA's update_entity debounce by repeatedly nudging the entity and
        polling the last_update_success attribute until it advances past the
        pre-trigger value. Returns the new state dict once the cycle completes.
        """
        start = time.time()
        deadline = start + timeout

        pre_state = self.get_state(entity_id)
        pre = pre_state.get("attributes", {}).get("last_update_success") if pre_state else None

        while time.time() < deadline:
            self.update_entity(entity_id)
            time.sleep(interval)
            state = self.get_state(entity_id)
            if state is not None:
                current = state.get("attributes", {}).get("last_update_success")
                if current is not None and current != pre:
                    return state

        elapsed = time.time() - start
        last_state = self.get_state(entity_id)
        last_value = last_state.get("attributes", {}).get("last_update_success") if last_state else None
        raise TimeoutError(
            f"wait_for_coordinator_cycle: entity {entity_id!r} did not advance "
            f"last_update_success past {pre!r} within {timeout}s (elapsed {elapsed:.1f}s). "
            f"Last seen value: {last_value!r}."
        )

    def poll_entity_state(
        self,
        entity_id: str,
        *,
        timeout: float = 60,
        interval: float = 2.0,
        condition=None,
    ) -> dict:
        """Poll entity until it exists and optionally meets a condition.

        Args:
            entity_id: The entity to poll.
            timeout: Max seconds to wait.
            interval: Seconds between polls.
            condition: Optional callable(state_dict) -> bool. If provided,
                       polls until it returns True.

        Returns the state dict when ready.
        Raises TimeoutError if timeout expires.
        """
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            state = self.get_state(entity_id)
            if state is not None:
                if condition is None or condition(state):
                    return state
                last_state = state
            time.sleep(interval)
        if last_state is not None:
            raise TimeoutError(
                f"Entity {entity_id} exists but condition not met within {timeout}s. "
                f"Last state: {last_state.get('state')}, "
                f"attrs: {last_state.get('attributes', {})}"
            )
        raise TimeoutError(f"Entity {entity_id} not found within {timeout}s")

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

    # Always start fresh - remove old E2E container and volume for a clean slate.
    # We deliberately do NOT use `down -v` here: that flag is project-scoped and
    # would remove every named volume declared in docker-compose.dev.yml - including
    # ha-dev-config, which belongs to the developer's manual dev instance.
    # Instead, stop the container without touching volumes, then remove just the
    # E2E volume by its full docker name (compose prefixes named volumes with the
    # project name, so the raw HA_VOLUME value won't match what docker sees).
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.dev.yml", "down"],
        capture_output=True, env=env,
    )
    subprocess.run(
        ["docker", "volume", "rm", "-f", _E2E_DOCKER_VOLUME],
        capture_output=True,
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
        # Mark integration step done - without this, browser logins are
        # redirected to /onboarding.html to finish setup.
        _raw_request("POST", "/api/onboarding/integration", {
            "client_id": f"{HA_URL}/",
            "redirect_uri": f"{HA_URL}/?auth_callback=1",
        }, token=token)
    else:
        # Already onboarded - log in
        token = _get_token_via_login()

    with open(TOKEN_FILE, "w") as f:
        f.write(token)

    # Add integration
    flow = _raw_request("POST", "/api/config/config_entries/flow",
                         {"handler": "rain_incoming"}, token=token)
    _raw_request("POST", f"/api/config/config_entries/flow/{flow['flow_id']}", {
        "latitude": -33.701,
        "longitude": 151.209,
        "lookahead_minutes": 60,
    }, token=token)

    # Wait for coordinator to do its first update
    client = HAClient(token)
    try:
        client.poll_entity_state(
            "binary_sensor.rain_incoming_status",
            timeout=30,
            condition=lambda s: s.get("state") not in (None, "unavailable"),
        )
    except TimeoutError:
        pass  # best-effort; tests will fail with clear messages if entity isn't ready

    return client


@pytest.fixture(scope="class")
def configure_location(ha_client):
    """Factory fixture to add an integration for a specific location.

    Class-scoped so all tests in a class share the same configured
    integration - subsequent tests can switch scenarios without
    re-creating the config entry.

    Returns a function that configures a new rain_incoming integration
    entry and waits for its first coordinator update.
    """
    entry_ids: list[str] = []

    def _configure(lat: float, lon: float, name: str = "Test") -> None:
        flow = ha_client.request(
            "POST",
            "/api/config/config_entries/flow",
            {"handler": "rain_incoming"},
        )
        result = ha_client.request(
            "POST",
            f"/api/config/config_entries/flow/{flow['flow_id']}",
            {
                "latitude": lat,
                "longitude": lon,
                "lookahead_minutes": 60,
                "location_name": name,
            },
        )
        if result and "result" in result:
            entry_id = result.get("result")
            if isinstance(entry_id, dict):
                entry_id = entry_id.get("entry_id")
            if entry_id:
                entry_ids.append(entry_id)
        # Poll until the binary sensor for this location is ready
        sensor_slug = name.lower().replace(" ", "_")
        sensor_id = f"binary_sensor.rain_incoming_{sensor_slug}_status"
        try:
            ha_client.poll_entity_state(
                sensor_id,
                timeout=30,
                condition=lambda s: s.get("state") not in (None, "unavailable"),
            )
        except TimeoutError:
            pass  # best-effort; test will fail with a clear message if entity isn't ready

    yield _configure

    # Teardown: remove the config entries we added
    for entry_id in entry_ids:
        try:
            ha_client.request(
                "DELETE",
                f"/api/config/config_entries/entry/{entry_id}",
            )
        except Exception:
            pass  # best-effort cleanup
