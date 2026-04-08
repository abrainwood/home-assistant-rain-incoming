from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.incoming_rain.coordinator import RainDetectorCoordinator
from custom_components.incoming_rain.radar.detector import Confidence, DetectionResult


def make_entry(lat: float = -33.701, lon: float = 151.209, lookahead: int = 60):
    entry = MagicMock()
    entry.data = {
        "latitude": lat,
        "longitude": lon,
        "lookahead_minutes": lookahead,
    }
    entry.entry_id = "test_entry"
    return entry


EMPTY_RESULT = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
)


@pytest.mark.asyncio
async def test_coordinator_returns_detection_result(hass: HomeAssistant):
    entry = make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    mock_session = MagicMock()

    with (
        patch.object(coordinator._provider, "get_frames", new=AsyncMock(return_value=[])),
        patch("custom_components.incoming_rain.coordinator.detect", return_value=EMPTY_RESULT),
        patch(
            "custom_components.incoming_rain.coordinator.async_get_clientsession",
            return_value=mock_session,
        ),
        patch(
            "custom_components.incoming_rain.coordinator.fetch_precipitation_now",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await coordinator._async_update_data()

    assert isinstance(result, DetectionResult)
    assert result.confidence == Confidence.UNAVAILABLE


@pytest.mark.asyncio
async def test_coordinator_raises_update_failed_on_persistent_error(hass: HomeAssistant):
    from homeassistant.helpers.update_coordinator import UpdateFailed

    entry = make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)
    coordinator._consecutive_failures = 99  # simulate already exhausted backoff

    with patch.object(
        coordinator._provider,
        "get_frames",
        new=AsyncMock(side_effect=Exception("network error")),
    ):
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()
