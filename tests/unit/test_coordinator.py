"""Unit tests for RainDetectorCoordinator helper methods."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_components.rain_incoming.coordinator import RainDetectorCoordinator
from custom_components.rain_incoming.const import (
    CONF_FORECAST_CONFIDENCE_ENABLED,
    CONF_SATELLITE_CONFIDENCE_ENABLED,
)
from custom_components.rain_incoming.providers.himawari import IRTile
from custom_components.rain_incoming.providers.rainviewer import RainViewerFrame
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult


def _make_coordinator(extra_data: dict | None = None):
    hass = MagicMock()
    hass.config.latitude = -33.7
    hass.config.longitude = 151.2
    hass.config.path.return_value = "/fake/.storage"
    hass.async_add_executor_job = AsyncMock(return_value=None)
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {"latitude": -33.7, "longitude": 151.2, "lookahead_minutes": 30, **(extra_data or {})}
    return RainDetectorCoordinator(hass, entry)


def _make_frame(ts: datetime) -> RainViewerFrame:
    frame = RainViewerFrame(timestamp=ts, path="/v2/radar/1234", zoom=7, colour_scheme=2)
    frame._cached_grid = np.zeros((512, 512), dtype=np.float32)
    frame._cached_bounds = MagicMock()
    return frame


_EMPTY_RESULT = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
)


class TestComputePopMultiplier:
    @pytest.mark.asyncio
    async def test_enabled_with_low_pop_returns_3x_multiplier(self):
        """Happy path: enabled=True + forecast with low PoP in next hour → multiplier 3.0.

        pop=3.0 (< 5%) maps to 3.0 via pop_to_threshold_multiplier, so the
        coordinator must return 3.0 when the forecast is within the 1-hour window.
        """
        fixed_now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        forecast = [(fixed_now, 3.0)]

        coordinator = _make_coordinator(extra_data={CONF_FORECAST_CONFIDENCE_ENABLED: True})
        mock_session = AsyncMock()

        with (
            patch(
                "custom_components.rain_incoming.coordinator.fetch_hourly_pop",
                new=AsyncMock(return_value=forecast),
            ),
            patch("custom_components.rain_incoming.coordinator.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            result = await coordinator._compute_pop_multiplier(-33.7, 151.2, mock_session)

        assert result == 3.0, f"Expected multiplier 3.0 for pop=3.0 (< 5%), got {result}"

    @pytest.mark.asyncio
    async def test_disabled_returns_1_without_fetching(self):
        """When CONF_FORECAST_CONFIDENCE_ENABLED is False, return 1.0 and skip fetch."""
        coordinator = _make_coordinator(extra_data={CONF_FORECAST_CONFIDENCE_ENABLED: False})
        mock_session = AsyncMock()

        with patch(
            "custom_components.rain_incoming.coordinator.fetch_hourly_pop",
            new=AsyncMock(),
        ) as mock_fetch:
            result = await coordinator._compute_pop_multiplier(-33.7, 151.2, mock_session)

        assert result == 1.0, f"Expected 1.0 when disabled, got {result}"
        mock_fetch.assert_not_called()


class TestComputeSatelliteMultiplier:
    @pytest.mark.asyncio
    async def test_enabled_with_clear_sky_returns_3x_multiplier(self):
        """Happy path: enabled=True + clear-sky tile → multiplier 3.0.

        A dark IR tile (luminance below threshold everywhere) yields
        cloud_fraction=0.0, which is below SATELLITE_CLOUD_PRESENCE_FRACTION,
        so cloud_to_threshold_multiplier returns 3.0.
        """
        coordinator = _make_coordinator(extra_data={CONF_SATELLITE_CONFIDENCE_ENABLED: True})
        mock_session = AsyncMock()

        dark_tile = IRTile(
            pixels=np.zeros((256, 256, 4), dtype=np.uint8),
            tile_x=0,
            tile_y=0,
            zoom=6,
        )

        with (
            patch(
                "custom_components.rain_incoming.coordinator.fetch_himawari_ir_tile",
                new=AsyncMock(return_value=dark_tile),
            ),
            patch(
                "custom_components.rain_incoming.coordinator.cloud_fraction_in_window",
                return_value=0.0,
            ),
        ):
            result = await coordinator._compute_satellite_multiplier(-33.7, 151.2, mock_session)

        assert result == 3.0, f"Expected multiplier 3.0 for clear sky (cloud_fraction=0.0), got {result}"

    @pytest.mark.asyncio
    async def test_disabled_returns_1_without_fetching(self):
        """When CONF_SATELLITE_CONFIDENCE_ENABLED is False, return 1.0 and skip fetch."""
        coordinator = _make_coordinator(extra_data={CONF_SATELLITE_CONFIDENCE_ENABLED: False})
        mock_session = AsyncMock()

        with patch(
            "custom_components.rain_incoming.coordinator.fetch_himawari_ir_tile",
            new=AsyncMock(),
        ) as mock_fetch:
            result = await coordinator._compute_satellite_multiplier(-33.7, 151.2, mock_session)

        assert result == 1.0, f"Expected 1.0 when disabled, got {result}"
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_tile_fetch_failure_returns_1_without_crash(self):
        """Enabled but tile fetch returns None → fail open with multiplier 1.0, no exception.

        `cloud_fraction_in_window` must NOT be called when tile is None - calling
        it on None would crash. Verify both the return value and that the
        guard short-circuits the cloud-fraction computation.
        """
        coordinator = _make_coordinator(extra_data={CONF_SATELLITE_CONFIDENCE_ENABLED: True})
        mock_session = AsyncMock()

        with (
            patch(
                "custom_components.rain_incoming.coordinator.fetch_himawari_ir_tile",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "custom_components.rain_incoming.coordinator.cloud_fraction_in_window",
            ) as mock_cloud_fraction,
        ):
            result = await coordinator._compute_satellite_multiplier(-33.7, 151.2, mock_session)

        assert result == 1.0, f"Expected 1.0 on tile fetch failure, got {result}"
        mock_cloud_fraction.assert_not_called()


class TestCombinedMultiplier:
    @pytest.mark.asyncio
    async def test_uses_max_of_pop_and_satellite_multipliers(self):
        """_async_update_data must take max(pop_mult, sat_mult) and pass it to _build_config.

        With pop_mult=2.0 and sat_mult=3.0, the DetectorConfig handed to detect()
        must carry pop_multiplier=3.0 - the larger of the two. (The field is
        called `pop_multiplier` historically; semantically it's now the combined
        confidence multiplier.)
        """
        fixed_now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        coordinator = _make_coordinator(
            extra_data={
                CONF_FORECAST_CONFIDENCE_ENABLED: True,
                CONF_SATELLITE_CONFIDENCE_ENABLED: True,
            }
        )
        frame = _make_frame(fixed_now)

        captured_configs = []

        def _capture_detect(**kwargs):
            captured_configs.append(kwargs["config"])
            return _EMPTY_RESULT

        with (
            patch.object(coordinator._provider, "get_frames", new=AsyncMock(return_value=[frame])),
            patch.object(coordinator._provider, "prefetch_frames", new=AsyncMock()),
            patch("custom_components.rain_incoming.coordinator.async_get_clientsession", return_value=MagicMock()),
            patch.object(
                coordinator, "_compute_pop_multiplier",
                new=AsyncMock(return_value=2.0),
            ),
            patch.object(
                coordinator, "_compute_satellite_multiplier",
                new=AsyncMock(return_value=3.0),
            ),
            patch("custom_components.rain_incoming.coordinator.detect", side_effect=_capture_detect),
        ):
            await coordinator._async_update_data()

        assert len(captured_configs) == 1, "detect() was not called"
        assert captured_configs[0].pop_multiplier == 3.0, (
            f"Expected combined multiplier 3.0 (max of pop=2.0 and sat=3.0), "
            f"got {captured_configs[0].pop_multiplier}. "
            "_async_update_data is not taking the max of the two multipliers."
        )


class TestBuildConfigUsesPopMultiplier:
    @pytest.mark.asyncio
    async def test_pop_multiplier_flows_into_detect_config(self):
        """W2: _async_update_data computes pop_multiplier and passes it to detect().

        When forecast confidence is enabled and fetch returns low PoP (3.0%),
        the DetectorConfig passed to detect() must have pop_multiplier=3.0, not 1.0.
        """
        fixed_now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        forecast = [(fixed_now, 3.0)]  # 3% PoP → multiplier 3.0

        coordinator = _make_coordinator(extra_data={CONF_FORECAST_CONFIDENCE_ENABLED: True})
        frame = _make_frame(fixed_now)

        captured_configs = []

        def _capture_detect(**kwargs):
            captured_configs.append(kwargs["config"])
            return _EMPTY_RESULT

        with (
            patch.object(coordinator._provider, "get_frames", new=AsyncMock(return_value=[frame])),
            patch.object(coordinator._provider, "prefetch_frames", new=AsyncMock()),
            patch("custom_components.rain_incoming.coordinator.async_get_clientsession", return_value=MagicMock()),
            patch("custom_components.rain_incoming.coordinator.fetch_hourly_pop", new=AsyncMock(return_value=forecast)),
            patch("custom_components.rain_incoming.coordinator.datetime") as mock_dt,
            patch("custom_components.rain_incoming.coordinator.detect", side_effect=_capture_detect),
        ):
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            await coordinator._async_update_data()

        assert len(captured_configs) == 1, "detect() was not called"
        assert captured_configs[0].pop_multiplier == 3.0, (
            f"Expected pop_multiplier=3.0 (from 3% PoP forecast), got {captured_configs[0].pop_multiplier}. "
            "_compute_pop_multiplier result is not flowing into DetectorConfig."
        )
