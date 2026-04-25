"""FutureRadarVerifier: checks whether rain actually arrived after a prediction.

For each prediction, looks at captures in [window_end_ts, window_end_ts + lookahead]
and checks whether any future capture shows rain at the location.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from custom_components.rain_incoming.const import INTENSITY_THRESHOLD, PROXIMITY_RADIUS_KM
from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import _tile_bounds

from scripts.backtest.captured_frame import CapturedRadarFrame
from scripts.backtest.data_loader import CaptureRecord
from scripts.backtest.metrics import VerificationRecord
from scripts.backtest.replay import PredictionRecord

_LOGGER = logging.getLogger(__name__)

_GRID_SIZE = 512  # 2 tiles × 256 pixels per tile


def _proximity_half_pixels(proximity_km: float, bounds: BoundingBox, grid_size: int) -> int:
    """Convert a proximity radius in km to a half-width in pixels.

    Uses the same formula as the detector's proximity_px calculation
    (detector.py _pixel_size_km) so verification matches detection.
    """
    lat_span_km = (bounds.lat_max - bounds.lat_min) * 111.0
    lon_span_km = (
        (bounds.lon_max - bounds.lon_min)
        * 111.0
        * math.cos(math.radians((bounds.lat_min + bounds.lat_max) / 2))
    )
    km_per_row = lat_span_km / grid_size
    km_per_col = lon_span_km / grid_size
    avg_km_per_px = (km_per_row + km_per_col) / 2
    return int(proximity_km / avg_km_per_px)


@dataclass
class VerifierConfig:
    lookahead_seconds: int = 3600
    intensity_threshold: float = INTENSITY_THRESHOLD
    proximity_km: float = PROXIMITY_RADIUS_KM


class FutureRadarVerifier:
    """Verify predictions by checking future radar captures.

    For each prediction, looks at captures in the
    [window_end_ts, window_end_ts + lookahead_seconds] window. Loads each
    capture's tiles, checks intensity at the location pixel and its proximity
    neighborhood. If any future capture shows rain at the location,
    actual_rain=True.
    """

    def __init__(
        self,
        captures: list[CaptureRecord],
        config: VerifierConfig | None = None,
    ) -> None:
        self._captures = sorted(captures, key=lambda c: c.frame_ts)
        self._config = config or VerifierConfig()
        # Pre-build frames once so _rain_at_location reuses them across
        # predictions instead of re-decoding tile PNGs for every check.
        self._frame_cache: dict[int, CapturedRadarFrame] = {
            c.frame_ts: CapturedRadarFrame(
                _timestamp=datetime.fromtimestamp(c.frame_ts, tz=timezone.utc),
                _tile_paths=c.tile_paths,
                _zoom=c.zoom,
            )
            for c in self._captures
        }

    def verify(self, predictions: list[PredictionRecord]) -> list[VerificationRecord]:
        """Verify each prediction against future radar captures."""
        return [self._verify_one(p) for p in predictions]

    def _verify_one(self, prediction: PredictionRecord) -> VerificationRecord:
        config = self._config
        window_start = prediction.window_end_ts
        window_end = prediction.window_end_ts + config.lookahead_seconds

        future_captures = [
            c for c in self._captures
            if window_start <= c.frame_ts <= window_end
        ]

        actual_rain = False
        actual_arrival_minutes: float | None = None

        for capture in future_captures:
            if self._rain_at_location(capture):
                actual_rain = True
                delta_seconds = capture.frame_ts - prediction.window_end_ts
                actual_arrival_minutes = delta_seconds / 60.0
                break  # first rain capture determines arrival time

        return VerificationRecord(
            window_end_ts=prediction.window_end_ts,
            predicted_rain=prediction.rain_incoming,
            actual_rain=actual_rain,
            predicted_arrival_minutes=prediction.arrival_minutes,
            actual_arrival_minutes=actual_arrival_minutes,
        )

    def _rain_at_location(self, capture: CaptureRecord) -> bool:
        """Return True if any future capture shows rain in the location neighborhood."""
        config = self._config
        bounds = _build_analysis_bounds(capture)
        frame = self._frame_cache[capture.frame_ts]
        grid = frame.get_intensity_grid(bounds, _GRID_SIZE, _GRID_SIZE)

        loc_row, loc_col = _location_pixel(capture.lat, capture.lon, bounds, _GRID_SIZE)
        half = _proximity_half_pixels(config.proximity_km, bounds, _GRID_SIZE)

        row_min = max(0, loc_row - half)
        row_max = min(_GRID_SIZE, loc_row + half + 1)
        col_min = max(0, loc_col - half)
        col_max = min(_GRID_SIZE, loc_col + half + 1)

        neighborhood = grid[row_min:row_max, col_min:col_max]
        return float(neighborhood.max()) >= config.intensity_threshold


def _build_analysis_bounds(capture: CaptureRecord) -> BoundingBox:
    tiles = list(capture.tile_paths.keys())
    all_bounds = [_tile_bounds(tx, ty, capture.zoom) for tx, ty in tiles]
    return BoundingBox(
        lat_min=min(b.lat_min for b in all_bounds),
        lat_max=max(b.lat_max for b in all_bounds),
        lon_min=min(b.lon_min for b in all_bounds),
        lon_max=max(b.lon_max for b in all_bounds),
    )


def _location_pixel(
    lat: float, lon: float, bounds: BoundingBox, grid_size: int
) -> tuple[int, int]:
    """Return (row, col) of the location in the intensity grid."""
    loc_col = int((lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * grid_size)
    loc_row = int((bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * grid_size)
    loc_col = max(0, min(grid_size - 1, loc_col))
    loc_row = max(0, min(grid_size - 1, loc_row))
    return loc_row, loc_col
