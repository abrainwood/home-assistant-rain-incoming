from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.rain_incoming.coordinator import RainDetectorCoordinator
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult


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
        patch("custom_components.rain_incoming.coordinator.detect", return_value=EMPTY_RESULT),
        patch(
            "custom_components.rain_incoming.coordinator.async_get_clientsession",
            return_value=mock_session,
        ),
    ):
        result = await coordinator._async_update_data()

    assert isinstance(result, DetectionResult)
    assert result.confidence == Confidence.UNAVAILABLE


@pytest.mark.asyncio
async def test_coordinator_passes_ha_session_to_provider(hass: HomeAssistant):
    """The coordinator must pass HA's shared session to get_frames so the
    provider uses it for manifest fetches instead of creating its own."""
    entry = make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    mock_session = MagicMock()
    mock_get_frames = AsyncMock(return_value=[])

    with (
        patch.object(coordinator._provider, "get_frames", new=mock_get_frames),
        patch("custom_components.rain_incoming.coordinator.detect", return_value=EMPTY_RESULT),
        patch(
            "custom_components.rain_incoming.coordinator.async_get_clientsession",
            return_value=mock_session,
        ),
    ):
        await coordinator._async_update_data()

    mock_get_frames.assert_called_once()
    call_kwargs = mock_get_frames.call_args
    assert call_kwargs.kwargs.get("session") is mock_session, (
        "Coordinator must pass the HA shared session to provider.get_frames. "
        f"Got session={call_kwargs.kwargs.get('session')}"
    )


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
