"""
Integration test conftest.

Prevents the coordinator from making real HTTP calls to RainViewer during tests
that exercise the config flow or other HA setup paths that auto-start the coordinator.
The sensor tests mock _async_update_data explicitly; this fixture covers the rest.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.incoming_rain.radar.detector import Confidence, DetectionResult

_MOCK_RESULT_UNAVAILABLE = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
)


@pytest.fixture(autouse=True)
def mock_coordinator_network(request):
    """
    Patch coordinator network calls for integration tests that set up real HA entries.

    The coordinator tests call _async_update_data directly with their own patches,
    so we skip this fixture for that module to avoid double-patching.
    """
    if "test_coordinator" in request.module.__name__:
        yield
        return

    with patch(
        "custom_components.incoming_rain.coordinator.RainDetectorCoordinator._async_update_data",
        new=AsyncMock(return_value=_MOCK_RESULT_UNAVAILABLE),
    ):
        yield
