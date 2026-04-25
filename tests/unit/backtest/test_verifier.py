"""Tests for FutureRadarVerifier - verifies predictions against future radar captures.

TDD: these tests were written first. FutureRadarVerifier is implemented in
scripts/backtest/verifier.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.const import INTENSITY_THRESHOLD
from custom_components.rain_incoming.providers.rainviewer import (
    PRECIP_COLOURS,
    _tile_bounds,
    _tile_to_intensity_array,
)
from scripts.backtest.data_loader import CaptureRecord
from scripts.backtest.replay import PredictionRecord


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_LAT = -37.8136
_LON = 144.9631
_ZOOM = 7
_ANALYSIS_TILES = [(115, 78), (116, 78), (115, 79), (116, 79)]
_BASE_TS = 1_776_725_400


_GOLDEN_TILES = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "golden_v2"
    / "Melbourne_dry"
    / "bronze"
    / "tiles"
)
_DRY_TS = 1776725400


def _make_prediction(
    window_end_ts: int = _BASE_TS,
    rain_incoming: bool = False,
    rain_at_location: bool = False,
    arrival_minutes: float | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        window_end_ts=window_end_ts,
        rain_incoming=rain_incoming,
        rain_at_location=rain_at_location,
        arrival_minutes=arrival_minutes,
        confidence="normal",
        cell_count=0,
        max_intensity=0.0,
    )


def _make_capture(
    frame_ts: int,
    tile_paths: dict[tuple[int, int], Path],
    lat: float = _LAT,
    lon: float = _LON,
) -> CaptureRecord:
    return CaptureRecord(
        capture_utc=datetime.fromtimestamp(frame_ts, tz=timezone.utc),
        frame_ts=frame_ts,
        tile_paths=tile_paths,
        location_name="Melbourne_dry",
        lat=lat,
        lon=lon,
        zoom=_ZOOM,
    )


def _make_transparent_tile_png() -> bytes:
    """256x256 fully transparent PNG - produces zero intensity."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rain_tile_png(colour_index: int = 0) -> bytes:
    """256x256 PNG filled with PRECIP_COLOURS[colour_index] at full opacity."""
    r, g, b, _ = PRECIP_COLOURS[colour_index]
    img = Image.new("RGBA", (256, 256), (r, g, b, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_tile_files(
    tmp_path: Path, ts: int, tile_data: dict[tuple[int, int], bytes]
) -> dict[tuple[int, int], Path]:
    """Write tile PNGs to tmp_path and return the tile_paths dict."""
    tile_dir = tmp_path / str(ts)
    tile_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for (tx, ty), data in tile_data.items():
        path = tile_dir / f"7_{tx}_{ty}_s2.png"
        path.write_bytes(data)
        paths[(tx, ty)] = path
    return paths


def _dry_tile_paths(tmp_path: Path, ts: int) -> dict[tuple[int, int], Path]:
    """All transparent tiles for the analysis grid."""
    tile_data = {(tx, ty): _make_transparent_tile_png() for tx, ty in _ANALYSIS_TILES}
    return _write_tile_files(tmp_path, ts, tile_data)


def _rain_tile_paths(
    tmp_path: Path, ts: int, colour_index: int = 0
) -> dict[tuple[int, int], Path]:
    """All rain-coloured tiles for the analysis grid."""
    tile_data = {
        (tx, ty): _make_rain_tile_png(colour_index) for tx, ty in _ANALYSIS_TILES
    }
    return _write_tile_files(tmp_path, ts, tile_data)


# ---------------------------------------------------------------------------
# Verify test tiles produce expected intensities (pre-flight check)
# ---------------------------------------------------------------------------


def test_synthetic_dry_tile_produces_zero_intensity():
    """Transparent PNG produces zero intensity via production decoder."""
    grid = _tile_to_intensity_array(_make_transparent_tile_png())
    assert grid.max() == 0.0


def test_synthetic_rain_tile_produces_nonzero_intensity():
    """Rain-coloured PNG produces expected intensity via production decoder."""
    _, _, _, expected = PRECIP_COLOURS[0]
    grid = _tile_to_intensity_array(_make_rain_tile_png(0))
    assert grid.max() == pytest.approx(expected, abs=1e-4)
    assert grid.max() >= INTENSITY_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Empty predictions → empty records
# ---------------------------------------------------------------------------


class TestVerifyEmptyPredictionsReturnsEmpty:
    def test_empty_predictions_returns_empty(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier

        captures = [_make_capture(_BASE_TS + 600, _dry_tile_paths(tmp_path, _BASE_TS + 600))]
        verifier = FutureRadarVerifier(captures)
        result = verifier.verify([])
        assert result == []


# ---------------------------------------------------------------------------
# 2. No future captures → actual_rain=False
# ---------------------------------------------------------------------------


class TestVerifyNoFutureCapturesReturnsNoRain:
    def test_no_captures_after_window_end(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # Capture is BEFORE the prediction window end - no future captures
        earlier_ts = _BASE_TS - 600
        captures = [
            _make_capture(earlier_ts, _dry_tile_paths(tmp_path, earlier_ts))
        ]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].actual_rain is False
        assert result[0].actual_arrival_minutes is None

    def test_no_captures_at_all(self):
        from scripts.backtest.verifier import FutureRadarVerifier

        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier([])
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].actual_rain is False


# ---------------------------------------------------------------------------
# 3. Rain predicted and arrived → hit
# ---------------------------------------------------------------------------


class TestVerifyRainPredictedAndArrived:
    def test_hit_actual_rain_true(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        tile_paths = _rain_tile_paths(tmp_path, future_ts)
        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(
            window_end_ts=_BASE_TS, rain_incoming=True, arrival_minutes=10.0
        )
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].predicted_rain is True
        assert result[0].actual_rain is True

    def test_hit_predicted_rain_preserved(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(
            window_end_ts=_BASE_TS, rain_incoming=True, arrival_minutes=20.0
        )
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].predicted_arrival_minutes == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 4. Rain predicted but dry → false alarm
# ---------------------------------------------------------------------------


class TestVerifyRainPredictedButDry:
    def test_false_alarm_actual_rain_false(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _dry_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(
            window_end_ts=_BASE_TS, rain_incoming=True, arrival_minutes=10.0
        )
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].predicted_rain is True
        assert result[0].actual_rain is False
        assert result[0].actual_arrival_minutes is None


# ---------------------------------------------------------------------------
# 5. No rain predicted but arrived → miss
# ---------------------------------------------------------------------------


class TestVerifyNoRainPredictedButArrived:
    def test_miss_actual_rain_true(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=False)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].predicted_rain is False
        assert result[0].actual_rain is True


# ---------------------------------------------------------------------------
# 6. No rain predicted and dry → correct negative
# ---------------------------------------------------------------------------


class TestVerifyNoRainPredictedAndDry:
    def test_correct_negative(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _dry_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=False)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].predicted_rain is False
        assert result[0].actual_rain is False


# ---------------------------------------------------------------------------
# 7. Lookahead window respected
# ---------------------------------------------------------------------------


class TestVerifyRespectsLookaheadWindow:
    def test_capture_within_lookahead_is_included(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # Exactly at the boundary: window_end_ts + lookahead_seconds
        lookahead = 3600
        future_ts = _BASE_TS + lookahead
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=lookahead))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True

    def test_capture_beyond_lookahead_is_ignored(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        lookahead = 3600
        future_ts = _BASE_TS + lookahead + 1  # just outside the window
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=lookahead))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is False

    def test_capture_at_window_end_ts_is_included(self, tmp_path):
        """Capture exactly at window_end_ts (offset=0) is within the window."""
        future_ts = _BASE_TS  # exactly at window_end_ts
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True


# ---------------------------------------------------------------------------
# 8. Actual arrival minutes
# ---------------------------------------------------------------------------


class TestVerifyActualArrivalMinutes:
    def test_first_rain_capture_determines_arrival(self, tmp_path):
        """actual_arrival_minutes = time from window_end_ts to first rain capture."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        ts_dry_1 = _BASE_TS + 300   # 5 min later, dry
        ts_rain_1 = _BASE_TS + 900  # 15 min later, rain
        ts_rain_2 = _BASE_TS + 1500 # 25 min later, rain too

        captures = [
            _make_capture(ts_dry_1, _dry_tile_paths(tmp_path, ts_dry_1)),
            _make_capture(ts_rain_1, _rain_tile_paths(tmp_path, ts_rain_1)),
            _make_capture(ts_rain_2, _rain_tile_paths(tmp_path, ts_rain_2)),
        ]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True
        # 900 seconds / 60 = 15.0 minutes
        assert result[0].actual_arrival_minutes == pytest.approx(15.0)

    def test_no_rain_means_no_arrival_minutes(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _dry_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=False)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].actual_arrival_minutes is None

    def test_predicted_arrival_minutes_comes_from_prediction(self, tmp_path):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        future_ts = _BASE_TS + 600
        captures = [_make_capture(future_ts, _rain_tile_paths(tmp_path, future_ts))]
        prediction = _make_prediction(
            window_end_ts=_BASE_TS, rain_incoming=True, arrival_minutes=42.0
        )
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].predicted_arrival_minutes == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 9. Real dry tiles → actual_rain=False
# ---------------------------------------------------------------------------


class TestVerifyWithRealDryTiles:
    """Use Melbourne_dry golden tiles as future captures - expect no rain detected."""

    def _make_golden_capture(self, ts: int) -> CaptureRecord:
        tile_dir = _GOLDEN_TILES / str(ts)
        tile_paths = {
            (tx, ty): tile_dir / f"7_{tx}_{ty}_s2.png"
            for tx, ty in _ANALYSIS_TILES
        }
        return _make_capture(ts, tile_paths)

    def test_dry_tiles_produce_no_rain(self):
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # Use the first dry timestamp as a future capture
        window_end = _DRY_TS - 600  # prediction ends just before the dry capture
        future_capture = self._make_golden_capture(_DRY_TS)

        prediction = _make_prediction(window_end_ts=window_end, rain_incoming=False)
        verifier = FutureRadarVerifier(
            [future_capture], VerifierConfig(lookahead_seconds=3600)
        )
        result = verifier.verify([prediction])

        assert len(result) == 1
        assert result[0].actual_rain is False, (
            f"Expected no rain from dry golden tiles, but actual_rain=True"
        )

    def test_dry_tiles_multiple_captures_still_no_rain(self):
        """Multiple future dry captures should still yield actual_rain=False."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # Use several consecutive dry timestamps
        dry_timestamps = [_DRY_TS + i * 600 for i in range(3)]
        window_end = _DRY_TS - 600

        captures = [self._make_golden_capture(ts) for ts in dry_timestamps]
        prediction = _make_prediction(window_end_ts=window_end, rain_incoming=False)
        verifier = FutureRadarVerifier(
            captures, VerifierConfig(lookahead_seconds=7200)
        )
        result = verifier.verify([prediction])

        assert result[0].actual_rain is False


# ---------------------------------------------------------------------------
# 10. Proximity check
# ---------------------------------------------------------------------------


class TestVerifyProximityCheck:
    """Rain within the proximity neighborhood of location is detected; rain outside is not."""

    def _make_sparse_rain_tile(
        self, pixel_row: int, pixel_col: int, tile_x: int, tile_y: int
    ) -> bytes:
        """Create a tile with rain only at one specific pixel."""
        r, g, b, _ = PRECIP_COLOURS[0]
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        img.putpixel((pixel_col, pixel_row), (r, g, b, 255))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _location_pixel_in_grid(self) -> tuple[int, int]:
        """Return (row, col) of the location in the 512x512 grid."""
        all_bounds = [_tile_bounds(tx, ty, _ZOOM) for tx, ty in _ANALYSIS_TILES]
        from custom_components.rain_incoming.providers.base import BoundingBox
        bounds = BoundingBox(
            lat_min=min(b.lat_min for b in all_bounds),
            lat_max=max(b.lat_max for b in all_bounds),
            lon_min=min(b.lon_min for b in all_bounds),
            lon_max=max(b.lon_max for b in all_bounds),
        )
        loc_col = int((_LON - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * 512)
        loc_row = int((bounds.lat_max - _LAT) / (bounds.lat_max - bounds.lat_min) * 512)
        return loc_row, loc_col

    def test_rain_at_location_pixel_detected(self, tmp_path):
        """A rain pixel exactly at the location pixel is detected."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        loc_row, loc_col = self._location_pixel_in_grid()

        # The grid stitches tiles in order: (115,78), (116,78), (115,79), (116,79)
        # min_x=115, min_y=78.  Tile (115,78) starts at canvas (0,0).
        # loc_col is in [0, 512), loc_row is in [0, 512).
        # Work out which tile holds the location pixel:
        tile_col = loc_col // 256  # 0 or 1 (within x tiles)
        tile_row = loc_row // 256  # 0 or 1 (within y tiles)
        tx = 115 + tile_col
        ty = 78 + tile_row
        within_tile_col = loc_col % 256
        within_tile_row = loc_row % 256

        # Only the target tile has rain; others are transparent
        future_ts = _BASE_TS + 600
        tile_dir = tmp_path / str(future_ts)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_paths = {}
        for (ttx, tty) in _ANALYSIS_TILES:
            path = tile_dir / f"7_{ttx}_{tty}_s2.png"
            if ttx == tx and tty == ty:
                path.write_bytes(
                    self._make_sparse_rain_tile(within_tile_row, within_tile_col, ttx, tty)
                )
            else:
                path.write_bytes(_make_transparent_tile_png())
            tile_paths[(ttx, tty)] = path

        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True, (
            f"Expected rain detected at location pixel ({loc_row},{loc_col}), "
            f"within tile ({tx},{ty}) at ({within_tile_row},{within_tile_col})"
        )

    def test_rain_far_from_location_not_detected(self, tmp_path):
        """Rain far outside the proximity neighborhood is not detected."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        loc_row, loc_col = self._location_pixel_in_grid()

        # Place rain far away from the location - at corner (0, 0) of the grid
        # (top-left tile at pixel (0,0)) - well outside any proximity window
        tx_rain, ty_rain = 115, 78  # top-left tile
        far_row, far_col = 0, 0     # top-left pixel of that tile

        # Only put rain at (0,0) of tile (115,78), which is far from loc_row/loc_col
        future_ts = _BASE_TS + 600
        tile_dir = tmp_path / str(future_ts)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_paths = {}
        for (ttx, tty) in _ANALYSIS_TILES:
            path = tile_dir / f"7_{ttx}_{tty}_s2.png"
            if ttx == tx_rain and tty == ty_rain:
                path.write_bytes(
                    self._make_sparse_rain_tile(far_row, far_col, ttx, tty)
                )
            else:
                path.write_bytes(_make_transparent_tile_png())
            tile_paths[(ttx, tty)] = path

        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=False)
        # 2km proximity - far rain at grid corner is well outside
        verifier = FutureRadarVerifier(
            captures, VerifierConfig(lookahead_seconds=3600, proximity_km=2.0)
        )
        result = verifier.verify([prediction])

        assert result[0].actual_rain is False, (
            f"Rain at grid (0,0) should not be detected at location ({loc_row},{loc_col})"
        )

    def test_rain_at_edge_of_neighborhood_detected(self, tmp_path):
        """Rain at the edge of the km-based proximity neighborhood is detected."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # 5km at Melbourne resolution (~0.95 km/px) -> half = int(5.0/0.95) = 5 pixels
        proximity_km = 5.0
        expected_half = 5
        loc_row, loc_col = self._location_pixel_in_grid()

        # Place rain at exactly 'expected_half' pixels from location - at the boundary
        target_row = loc_row + expected_half
        target_col = loc_col

        # Map back to which tile and intra-tile pixel
        tile_col = target_col // 256
        tile_row = target_row // 256
        tx = 115 + tile_col
        ty = 78 + tile_row
        within_tile_col = target_col % 256
        within_tile_row = target_row % 256

        future_ts = _BASE_TS + 600
        tile_dir = tmp_path / str(future_ts)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_paths = {}
        for (ttx, tty) in _ANALYSIS_TILES:
            path = tile_dir / f"7_{ttx}_{tty}_s2.png"
            if ttx == tx and tty == ty:
                path.write_bytes(
                    self._make_sparse_rain_tile(within_tile_row, within_tile_col, ttx, tty)
                )
            else:
                path.write_bytes(_make_transparent_tile_png())
            tile_paths[(ttx, tty)] = path

        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        verifier = FutureRadarVerifier(
            captures, VerifierConfig(lookahead_seconds=3600, proximity_km=proximity_km)
        )
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True, (
            f"Rain at ({target_row},{target_col}) should be detected within "
            f"{proximity_km}km neighborhood of ({loc_row},{loc_col})"
        )

    def test_rain_just_outside_neighborhood_not_detected(self, tmp_path):
        """Rain just beyond the proximity window is not detected."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # 2km at Melbourne resolution (~0.95 km/px) -> half = int(2.0/0.95) = 2 pixels
        proximity_km = 2.0
        loc_row, loc_col = self._location_pixel_in_grid()

        # Place rain well outside the 2-pixel half-width
        outside_col = loc_col + 10
        outside_row = loc_row

        tile_col = outside_col // 256
        tile_row = outside_row // 256
        tx = 115 + tile_col
        ty = 78 + tile_row
        within_tile_col = outside_col % 256
        within_tile_row = outside_row % 256

        future_ts = _BASE_TS + 600
        tile_dir = tmp_path / str(future_ts)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_paths = {}
        for (ttx, tty) in _ANALYSIS_TILES:
            path = tile_dir / f"7_{ttx}_{tty}_s2.png"
            if ttx == tx and tty == ty:
                path.write_bytes(
                    self._make_sparse_rain_tile(within_tile_row, within_tile_col, ttx, tty)
                )
            else:
                path.write_bytes(_make_transparent_tile_png())
            tile_paths[(ttx, tty)] = path

        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=False)
        verifier = FutureRadarVerifier(
            captures, VerifierConfig(lookahead_seconds=3600, proximity_km=proximity_km)
        )
        result = verifier.verify([prediction])

        assert result[0].actual_rain is False, (
            f"Rain at ({outside_row},{outside_col}) should NOT be detected "
            f"with proximity_km={proximity_km} and location at ({loc_row},{loc_col})"
        )

    def test_multiple_predictions_verified_independently(self, tmp_path):
        """Multiple predictions are each verified against future captures independently."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        # First prediction window ends at BASE_TS, second at BASE_TS + 1800
        ts_rain_for_first = _BASE_TS + 600      # 10 min after first window
        ts_dry_for_second = _BASE_TS + 2400     # 10 min after second window

        captures = [
            _make_capture(ts_rain_for_first, _rain_tile_paths(tmp_path, ts_rain_for_first)),
            _make_capture(ts_dry_for_second, _dry_tile_paths(tmp_path, ts_dry_for_second)),
        ]
        predictions = [
            _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True),
            _make_prediction(window_end_ts=_BASE_TS + 1800, rain_incoming=False),
        ]
        verifier = FutureRadarVerifier(
            captures, VerifierConfig(lookahead_seconds=1200)
        )
        result = verifier.verify(predictions)

        assert len(result) == 2
        # First prediction: rain capture at +600 is within its lookahead (1200s)
        assert result[0].actual_rain is True
        # Second prediction: dry capture at BASE_TS+2400 is within its lookahead
        # (BASE_TS+1800 + 1200 = BASE_TS+3000), and it's dry
        assert result[1].actual_rain is False


# ---------------------------------------------------------------------------
# Frame caching: verifier must not rebuild frames for the same capture
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Km-based proximity: verifier should use km, not fixed pixels
# ---------------------------------------------------------------------------


class TestProximityKmComputation:
    """The verifier should compute proximity in pixels from a km radius,
    matching how the detector converts PROXIMITY_RADIUS_KM to pixels."""

    def test_proximity_half_pixels_penrith_bounds(self):
        """15km at Penrith resolution (~1.0 km/px) gives ~14 pixels."""
        from scripts.backtest.verifier import _proximity_half_pixels
        from custom_components.rain_incoming.providers.base import BoundingBox

        # Penrith-like bounds: 2x2 tiles at zoom 7, lat -33.75
        bounds = BoundingBox(
            lat_min=-36.60, lat_max=-31.95,
            lon_min=149.06, lon_max=154.69,
        )
        result = _proximity_half_pixels(15.0, bounds, 512)
        assert result == 14

    def test_proximity_half_pixels_equatorial_bounds(self):
        """15km at equatorial resolution (~1.2 km/px) gives ~12 pixels."""
        from scripts.backtest.verifier import _proximity_half_pixels
        from custom_components.rain_incoming.providers.base import BoundingBox

        # Darwin-like bounds: 2x2 tiles at zoom 7, lat -12.46
        bounds = BoundingBox(
            lat_min=-15.28, lat_max=-9.62,
            lon_min=128.67, lon_max=134.30,
        )
        result = _proximity_half_pixels(15.0, bounds, 512)
        assert result == 12

    def test_proximity_half_pixels_high_latitude_bounds(self):
        """15km at high-latitude resolution (~0.7 km/px) gives more pixels."""
        from scripts.backtest.verifier import _proximity_half_pixels
        from custom_components.rain_incoming.providers.base import BoundingBox

        # Ketchikan actual tile bounds: 2x2 tiles at zoom 7, lat ~55.7
        bounds = BoundingBox(
            lat_min=54.1624, lat_max=57.3265,
            lon_min=-135.0000, lon_max=-129.3750,
        )
        result = _proximity_half_pixels(15.0, bounds, 512)
        assert result == 21


class TestVerifierUsesKmProximity:
    """VerifierConfig.proximity_km replaces proximity_pixels for consistent
    km-based proximity across all locations."""

    def test_config_defaults_to_production_proximity(self):
        """VerifierConfig.proximity_km should default to PROXIMITY_RADIUS_KM."""
        from scripts.backtest.verifier import VerifierConfig
        from custom_components.rain_incoming.const import PROXIMITY_RADIUS_KM

        config = VerifierConfig()
        assert config.proximity_km == PROXIMITY_RADIUS_KM

    def test_rain_within_15km_detected(self, tmp_path):
        """Rain at ~10km from location (10 pixels at ~1 km/px) is detected
        with the new 15km proximity but would have been missed with old 5px half."""
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig

        loc_row, loc_col = TestVerifyProximityCheck._location_pixel_in_grid(
            TestVerifyProximityCheck()
        )

        # Place rain 10 pixels from location (~10km at Melbourne resolution)
        # Old verifier: half=5 -> misses at 10px offset
        # New verifier: 15km -> half~14px -> catches at 10px offset
        target_row = loc_row + 10
        target_col = loc_col

        tile_col = target_col // 256
        tile_row = target_row // 256
        tx = 115 + tile_col
        ty = 78 + tile_row
        within_tile_col = target_col % 256
        within_tile_row = target_row % 256

        future_ts = _BASE_TS + 600
        tile_dir = tmp_path / str(future_ts)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_paths = {}
        for (ttx, tty) in _ANALYSIS_TILES:
            path = tile_dir / f"7_{ttx}_{tty}_s2.png"
            if ttx == tx and tty == ty:
                r, g, b, _ = PRECIP_COLOURS[0]
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                img.putpixel((within_tile_col, within_tile_row), (r, g, b, 255))
                buf = BytesIO()
                img.save(buf, format="PNG")
                path.write_bytes(buf.getvalue())
            else:
                path.write_bytes(_make_transparent_tile_png())
            tile_paths[(ttx, tty)] = path

        captures = [_make_capture(future_ts, tile_paths)]
        prediction = _make_prediction(window_end_ts=_BASE_TS, rain_incoming=True)
        # Use default proximity_km (15.0) - should detect rain at 10px offset
        verifier = FutureRadarVerifier(captures, VerifierConfig(lookahead_seconds=3600))
        result = verifier.verify([prediction])

        assert result[0].actual_rain is True, (
            f"Rain at ({target_row},{target_col}), 10px from location ({loc_row},{loc_col}), "
            f"should be detected with 15km proximity (~14px half at Melbourne resolution)"
        )


# ---------------------------------------------------------------------------
# 12. Frame caching
# ---------------------------------------------------------------------------


class TestVerifierFrameCaching:
    """The verifier checks multiple predictions against the same future captures.
    Without caching, _rain_at_location builds a new CapturedRadarFrame for each
    check, re-decoding tile PNGs repeatedly."""

    def test_captured_frame_built_once_per_unique_capture(self, tmp_path):
        """CapturedRadarFrame must be created at most once per unique capture,
        not once per _rain_at_location call."""
        from unittest.mock import patch as mock_patch
        from scripts.backtest.verifier import FutureRadarVerifier, VerifierConfig
        from scripts.backtest.captured_frame import CapturedRadarFrame

        # Create 3 dry captures with transparent tile PNGs
        tile_bytes = _make_transparent_tile_png()
        captures = []
        for i in range(3):
            ts = _BASE_TS + i * 600
            tile_dir = tmp_path / str(ts)
            tile_dir.mkdir()
            tile_paths = {}
            for tx, ty in _ANALYSIS_TILES:
                path = tile_dir / f"7_{tx}_{ty}_s2.png"
                path.write_bytes(tile_bytes)
                tile_paths[(tx, ty)] = path
            captures.append(_make_capture(ts, tile_paths))

        # 2 predictions that both look at overlapping future captures
        predictions = [
            PredictionRecord(
                window_end_ts=_BASE_TS,  # looks at captures 0,1,2
                rain_incoming=True, rain_at_location=False,
                arrival_minutes=30.0, confidence="normal",
                cell_count=1, max_intensity=0.5,
            ),
            PredictionRecord(
                window_end_ts=_BASE_TS + 600,  # looks at captures 1,2
                rain_incoming=True, rain_at_location=False,
                arrival_minutes=30.0, confidence="normal",
                cell_count=1, max_intensity=0.5,
            ),
        ]
        config = VerifierConfig(lookahead_seconds=1800)

        with mock_patch(
            "scripts.backtest.verifier.CapturedRadarFrame",
            wraps=CapturedRadarFrame,
        ) as mock_frame_cls:
            verifier = FutureRadarVerifier(captures, config)
            verifier.verify(predictions)

        # With caching: each capture constructed at most once in __init__ = 3
        # Without caching: capture 1 and 2 checked by both predictions = 5
        assert mock_frame_cls.call_count <= len(captures), (
            f"CapturedRadarFrame constructed {mock_frame_cls.call_count} times "
            f"for {len(captures)} captures - frames not being cached"
        )
