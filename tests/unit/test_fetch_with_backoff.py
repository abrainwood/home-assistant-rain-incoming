"""Tests for coordinator._fetch_with_backoff retry limits.

I6: _fetch_with_backoff must cap the number of retry attempts to avoid
blocking _async_update_data for 255+ seconds.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.rain_incoming.coordinator import RainDetectorCoordinator
from custom_components.rain_incoming.const import BACKOFF_MAX_SECONDS


def _make_coordinator():
    hass = MagicMock()
    hass.config.latitude = -33.7
    hass.config.longitude = 151.2
    hass.config.path.return_value = "/fake/.storage"
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 60}
    return RainDetectorCoordinator(hass, entry)


class TestFetchWithBackoff:
    @pytest.mark.asyncio
    async def test_raises_after_max_retries_not_indefinitely(self):
        """_fetch_with_backoff must raise after a bounded number of attempts,
        not loop until backoff >= BACKOFF_MAX_SECONDS."""
        coordinator = _make_coordinator()
        call_count = 0

        async def _failing_get_frames(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("network down")

        with patch.object(
            coordinator._provider, "get_frames", side_effect=_failing_get_frames,
        ):
            with patch("custom_components.rain_incoming.coordinator.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(ConnectionError):
                    await coordinator._fetch_with_backoff(-33.7, 151.2)

        # Must not attempt more than 5 times (base=5, *2, *2, *2 = 40, *2 = 80 > 60 -> raise)
        # The exact count depends on the backoff progression but must be bounded.
        assert call_count <= 5, (
            f"_fetch_with_backoff made {call_count} attempts - must be bounded to "
            f"avoid blocking _async_update_data for hundreds of seconds"
        )

    @pytest.mark.asyncio
    async def test_succeeds_on_retry(self):
        """_fetch_with_backoff must return frames when a retry succeeds."""
        coordinator = _make_coordinator()
        mock_frames = [MagicMock(), MagicMock()]

        call_count = 0

        async def _flaky_get_frames(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return mock_frames

        with patch.object(
            coordinator._provider, "get_frames", side_effect=_flaky_get_frames,
        ):
            with patch("custom_components.rain_incoming.coordinator.asyncio.sleep", new=AsyncMock()):
                result = await coordinator._fetch_with_backoff(-33.7, 151.2)

        assert result == mock_frames
        assert call_count == 3
        assert coordinator.consecutive_failures == 0
