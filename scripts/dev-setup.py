#!/usr/bin/env python3
"""
Automated Home Assistant dev instance setup.

Usage:
    python scripts/dev-setup.py          # normal dev (real RainViewer)
    python scripts/dev-setup.py --mock   # E2E mode (mock RainViewer on port 9876)

Creates the HA account, completes onboarding, and adds the rain_incoming
integration. Saves the auth token to .dev-token for E2E tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HA_URL = "http://localhost:8123"
TOKEN_FILE = ".dev-token"
DEV_USER = {"name": "Developer", "username": "dev", "password": "devdevdev", "language": "en"}
HOME_LOCATION = {
    "latitude": -33.701,
    "longitude": 151.209,
    "country": "AU",
    "time_zone": "Australia/Sydney",
    "elevation": 200,
    "unit_system": "metric",
    "currency": "AUD",
    "language": "en",
}
INTEGRATION_CONFIG = {
    "location": {"latitude": -33.701, "longitude": 151.209},
    "lookahead_minutes": 30,
    "location_name": "Sydney",
}


def _request(method: str, path: str, data: dict | None = None,
             token: str | None = None, form: bool = False) -> dict | None:
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
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        print(f"  HTTP {e.code} on {method} {path}: {raw[:200]}")
        return None


def wait_for_ha(timeout: int = 120) -> None:
    print("Waiting for HA to start...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{HA_URL}/api/")
            urllib.request.urlopen(req, timeout=2)
            print(" ready")
            return
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                # HA is up but needs auth - that's fine
                print(" ready")
                return
            # Other HTTP errors (5xx etc) - keep waiting
            print(".", end="", flush=True)
            time.sleep(2)
        except (urllib.error.URLError, OSError):
            print(".", end="", flush=True)
            time.sleep(2)
    print("\nTimed out waiting for HA")
    sys.exit(1)


def _login(token_file: str = TOKEN_FILE) -> str:
    """Log in with existing credentials and return a fresh auth token."""
    print("Logging in with existing credentials...")
    # Start login flow
    flow = _request("POST", "/auth/login_flow", {
        "client_id": f"{HA_URL}/",
        "handler": ["homeassistant", None],
        "redirect_uri": f"{HA_URL}/",
    })
    if not flow or "flow_id" not in flow:
        print("Failed to start login flow")
        sys.exit(1)

    # Submit credentials
    result = _request("POST", f"/auth/login_flow/{flow['flow_id']}", {
        "username": DEV_USER["username"],
        "password": DEV_USER["password"],
        "client_id": f"{HA_URL}/",
    })
    if not result or result.get("type") != "create_entry":
        print(f"Login failed: {result}")
        sys.exit(1)

    auth_code = result["result"]

    # Exchange for token
    token_resp = _request("POST", "/auth/token", {
        "client_id": f"{HA_URL}/",
        "grant_type": "authorization_code",
        "code": auth_code,
    }, form=True)
    if not token_resp or "access_token" not in token_resp:
        print("Failed to get token")
        sys.exit(1)

    token = token_resp["access_token"]
    with open(token_file, "w") as f:
        f.write(token)
    print(f"Token saved to {token_file}")
    return token


def onboard(token_file: str = TOKEN_FILE) -> str:
    """Create account, get token, complete onboarding. Returns auth token."""
    # Try existing token first
    try:
        with open(token_file) as f:
            token = f.read().strip()
        # Verify it still works
        resp = _request("GET", "/api/", token=token)
        if resp is not None:
            print("Using saved token")
            return token
    except FileNotFoundError:
        pass

    # Check if onboarding is needed
    result = _request("GET", "/api/onboarding")
    if not result:
        # Already onboarded, no valid token - log in
        return _login(token_file)

    remaining = [step["step"] for step in result if not step.get("done", False)]
    if "user" not in remaining:
        # User already created - log in
        return _login(token_file)

    print("Creating dev account...")
    payload = {**DEV_USER, "client_id": f"{HA_URL}/"}
    resp = _request("POST", "/api/onboarding/users", payload)
    if not resp or "auth_code" not in resp:
        print("Failed to create user")
        sys.exit(1)

    auth_code = resp["auth_code"]

    print("Getting auth token...")
    token_resp = _request("POST", "/auth/token", {
        "client_id": f"{HA_URL}/",
        "grant_type": "authorization_code",
        "code": auth_code,
    }, form=True)
    if not token_resp or "access_token" not in token_resp:
        print("Failed to get token")
        sys.exit(1)

    token = token_resp["access_token"]

    # Complete remaining onboarding steps
    print("Completing onboarding...")
    _request("POST", "/api/onboarding/core_config", HOME_LOCATION, token=token)
    _request("POST", "/api/onboarding/analytics", {}, token=token)

    # Save token
    with open(token_file, "w") as f:
        f.write(token)
    print(f"Token saved to {token_file}")

    return token


def add_integration(token: str) -> bool:
    """Add the rain_incoming integration via config flow."""
    # Check if already configured
    entries = _request("GET", "/api/config/config_entries/entry", token=token)
    if entries:
        for entry in entries:
            if entry.get("domain") == "rain_incoming":
                print("Integration already configured")
                return True

    print("Adding rain_incoming integration...")
    flow = _request("POST", "/api/config/config_entries/flow",
                     {"handler": "rain_incoming"}, token=token)
    if not flow or "flow_id" not in flow:
        print("Failed to init config flow")
        return False

    result = _request("POST", f"/api/config/config_entries/flow/{flow['flow_id']}",
                       INTEGRATION_CONFIG, token=token)
    if not result:
        print("Failed to configure integration")
        return False

    if result.get("type") == "create_entry":
        print(f"Integration added: {result.get('title', 'rain_incoming')}")
        return True

    print(f"Unexpected flow result: {result}")
    return False


def start_docker(mock: bool = False) -> None:
    """Start the HA dev container."""
    env_args = []
    if mock:
        env_args = [
            "-e", "RAINVIEWER_API_URL=http://host.docker.internal:9876",
            "-e", "RAINVIEWER_TILE_URL=http://host.docker.internal:9876",
        ]

    # Check if already running
    result = subprocess.run(["docker", "ps", "-q", "-f", "name=ha-dev"],
                            capture_output=True, text=True)
    if result.stdout.strip():
        if mock:
            print("Restarting HA container with mock URLs...")
            subprocess.run(["docker", "compose", "-f", "docker-compose.dev.yml", "down"],
                           capture_output=True)
        else:
            print("HA container already running")
            return

    print(f"Starting HA container ({'mock' if mock else 'real'} RainViewer)...")
    env = {}
    if mock:
        env["RAINVIEWER_API_URL"] = "http://host.docker.internal:9876"
        env["RAINVIEWER_TILE_URL"] = "http://host.docker.internal:9876"

    import os
    full_env = {**os.environ, **env}
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.dev.yml", "up", "-d"],
        env=full_env,
        capture_output=True,
    )


def main() -> None:
    mock = "--mock" in sys.argv

    start_docker(mock=mock)
    wait_for_ha()
    token = onboard()
    add_integration(token)

    print(f"\nHA dev instance ready at {HA_URL}")
    print(f"Login: {DEV_USER['username']} / {DEV_USER['password']}")
    if mock:
        print("Mock RainViewer: http://localhost:9876")
    print(f"\nUseful commands:")
    print(f"  docker logs ha-dev -f --tail 50    # watch logs")
    print(f"  docker restart ha-dev              # restart after code changes")
    print(f"  docker compose -f docker-compose.dev.yml down  # stop")


if __name__ == "__main__":
    main()
