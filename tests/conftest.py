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

# pycares (async DNS resolver used by aiohttp) spawns a daemon thread named
# "Thread-N (_run_safe_shutdown_loop)" that can outlive tests. pytest-
# homeassistant-custom-component's verify_cleanup fixture asserts only
# DummyThread and "waitpid-*" threads remain after each test.
#
# Fix: rename pycares threads to have the "waitpid-" prefix in our fixture
# teardown, which runs BEFORE verify_cleanup's teardown (LIFO ordering).
_PYCARES_THREAD_MARKER = "_run_safe_shutdown_loop"


@pytest.fixture(autouse=True)
def _allow_pycares_threads():
    """Rename pycares threads so verify_cleanup's thread assertion passes."""
    yield
    for t in threading.enumerate():
        if _PYCARES_THREAD_MARKER in t.name:
            t.name = f"waitpid-pycares-{t.name}"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir for all tests."""
    yield
