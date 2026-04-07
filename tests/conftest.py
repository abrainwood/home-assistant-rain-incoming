import pathlib
import sys

import pytest

# HA's custom component loader iterates sys.path with pathlib.Path.iterdir().
# Editable installs (pip install -e .) add a virtual path hook entry to sys.path
# that doesn't exist on disk, causing a FileNotFoundError. Remove such entries
# before HA's loader runs. Imports still work because pyproject.toml's
# pythonpath = ["."] adds the real project root to sys.path directly.
sys.path[:] = [p for p in sys.path if not p or pathlib.Path(p).is_dir()]


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir for all tests."""
    yield
