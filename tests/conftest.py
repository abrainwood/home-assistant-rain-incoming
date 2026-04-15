import pathlib
import sys
import threading

import pytest

# HA's custom component loader iterates sys.path with pathlib.Path.iterdir().
# Editable installs (pip install -e .) add a virtual path hook entry to sys.path
# that doesn't exist on disk, causing a FileNotFoundError. Remove such entries
# before HA's loader runs. Imports still work because pyproject.toml's
# pythonpath = ["."] adds the real project root to sys.path directly.
sys.path[:] = [p for p in sys.path if not p or pathlib.Path(p).is_dir()]

# Thread names that external libraries may spawn during tests. The
# pytest-homeassistant-custom-component plugin asserts no unexpected threads
# remain after each test, but pycares (async DNS resolver used by aiohttp)
# spawns a daemon thread named "_run_safe_shutdown_loop" that can outlive the
# test. We pre-register these threads so the assertion doesn't flake.
_ALLOWED_THREAD_PREFIXES = ("_run_safe_shutdown_loop",)


@pytest.fixture(autouse=True)
def _register_known_external_threads():
    """Record threads from external libraries so HA's thread checker ignores them."""
    # Capture known threads before the test so they appear in threads_before
    # and are excluded from the post-test diff.
    known = {
        t for t in threading.enumerate()
        if any(t.name.startswith(p) for p in _ALLOWED_THREAD_PREFIXES)
    }
    yield
    # If pycares threads started during the test, give them a moment to finish.
    # This is a workaround, not a timing test - pycares daemon threads terminate
    # quickly once the resolver is garbage collected.
    for t in threading.enumerate():
        if any(t.name.startswith(p) for p in _ALLOWED_THREAD_PREFIXES) and t not in known:
            t.join(timeout=1.0)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir for all tests."""
    yield
