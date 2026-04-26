"""Unit tests for scripts.backtest.replay.ReplayEngine.

TDD: these tests were written first.  ReplayEngine is implemented in
scripts/backtest/replay.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.backtest.data_loader import CaptureRecord
from scripts.backtest.replay import PredictionRecord, ReplayConfig, ReplayEngine
from custom_components.rain_incoming.radar.detector import (
    Confidence,
    DetectionResult,
    DiagnosticTrace,
    FrameDiagnostic,
    DecisionDiagnostic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAT = -37.8136
_LON = 144.9631
_ZOOM = 7
_TILE_COORDS = [(115, 78), (116, 78), (115, 79), (116, 79)]

# Base timestamp (spaced 600 s apart = 10 min, well within max_gap_seconds=900)
_BASE_TS = 1_776_725_400


def _make_record(index: int, frame_ts: int | None = None) -> CaptureRecord:
    """Build a synthetic CaptureRecord.  Tile paths point to non-existent files
    (tests that don't actually call detect() don't need real tiles)."""
    ts = frame_ts if frame_ts is not None else _BASE_TS + index * 600
    tile_paths = {
        (x, y): Path(f"/nonexistent/{ts}/7_{x}_{y}_s6.png")
        for x, y in _TILE_COORDS
    }
    return CaptureRecord(
        capture_utc=datetime.fromtimestamp(ts, tz=timezone.utc),
        frame_ts=ts,
        tile_paths=tile_paths,
        location_name="Melbourne_dry",
        lat=_LAT,
        lon=_LON,
        zoom=_ZOOM,
    )


def _make_records(count: int) -> list[CaptureRecord]:
    return [_make_record(i) for i in range(count)]


def _mock_detect_result(
    rain_incoming: bool = False,
    rain_at_location: bool = False,
    confidence: Confidence = Confidence.NORMAL,
) -> DetectionResult:
    return DetectionResult(
        rain_incoming=rain_incoming,
        rain_at_location=rain_at_location,
        confidence=confidence,
        tracked_cells=[],
        arrival_time=None,
        frame_count=8,
        max_approaching_intensity=0.0,
        last_labeled=None,
    )


# ---------------------------------------------------------------------------
# 1. Empty captures → empty predictions
# ---------------------------------------------------------------------------


class TestReplayEmptyCapturesReturnsEmpty:
    def test_empty_captures_returns_empty(self) -> None:
        engine = ReplayEngine()
        result = engine.replay([])
        assert result == []


# ---------------------------------------------------------------------------
# 2. Fewer captures than window_size → empty predictions
# ---------------------------------------------------------------------------


class TestReplayFewerThanWindowSizeReturnsEmpty:
    def test_fewer_than_window_size_returns_empty(self) -> None:
        engine = ReplayEngine(ReplayConfig(window_size=8))
        captures = _make_records(5)
        with patch("scripts.backtest.replay.detect") as mock_detect:
            result = engine.replay(captures)
        assert result == []
        mock_detect.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Exact window_size → one prediction
# ---------------------------------------------------------------------------


class TestReplayExactWindowSizeReturnsOnePrediction:
    def test_exact_window_size_returns_one_prediction(self) -> None:
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        captures = _make_records(8)
        mock_result = _mock_detect_result()
        with patch("scripts.backtest.replay.detect", return_value=mock_result):
            predictions = engine.replay(captures)
        assert len(predictions) == 1
        assert isinstance(predictions[0], PredictionRecord)


# ---------------------------------------------------------------------------
# 4. 10 captures → 3 predictions (windows 0-7, 1-8, 2-9)
# ---------------------------------------------------------------------------


class TestReplaySlidingWindowCount:
    def test_sliding_window_count(self) -> None:
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        captures = _make_records(10)
        mock_result = _mock_detect_result()
        with patch("scripts.backtest.replay.detect", return_value=mock_result) as mock_detect:
            predictions = engine.replay(captures)
        assert len(predictions) == 3
        assert mock_detect.call_count == 3


# ---------------------------------------------------------------------------
# 5. Gap in window → skips that window
# ---------------------------------------------------------------------------


class TestReplaySkipsGap:
    def test_skips_window_with_large_gap(self) -> None:
        """A 20-minute gap (1200 s) between frames 4 and 5 in a window of 8
        exceeds max_gap_seconds=900, so that window is skipped."""
        records = _make_records(8)
        # Replace the 5th record (index 4) with a timestamp 1200 s after the 4th
        ts_before = records[3].frame_ts
        gapped = _make_record(99, frame_ts=ts_before + 1200)
        records = records[:4] + [gapped] + records[5:]
        # Re-sort so replay sees them in order
        records = sorted(records, key=lambda r: r.frame_ts)

        engine = ReplayEngine(ReplayConfig(window_size=8, max_gap_seconds=900, qc_enabled=False))
        mock_result = _mock_detect_result()
        with patch("scripts.backtest.replay.detect", return_value=mock_result) as mock_detect:
            predictions = engine.replay(records)

        # The only 8-record window contains the gap - so no predictions
        assert predictions == []
        mock_detect.assert_not_called()


class TestReplayGapSkipsOnlyAffectedWindows:
    def test_gap_only_skips_windows_containing_it(self) -> None:
        """With 12 records and a gap between records 4 and 5 (0-indexed), only
        windows that span the gap are skipped; later windows that don't include
        records 4 or earlier still run."""
        # Records 0-3: timestamps 0-3*600
        # Record 4: +1200 s after record 3 (gap)
        # Records 5-11: +600 s each after record 4
        records = []
        ts = _BASE_TS
        for i in range(4):
            records.append(_make_record(i, frame_ts=ts))
            ts += 600
        ts += 1200  # the gap
        for i in range(8):
            records.append(_make_record(100 + i, frame_ts=ts))
            ts += 600
        # Total: 12 records

        engine = ReplayEngine(ReplayConfig(window_size=8, max_gap_seconds=900, qc_enabled=False))
        mock_result = _mock_detect_result()
        with patch("scripts.backtest.replay.detect", return_value=mock_result) as mock_detect:
            predictions = engine.replay(records)

        # Windows containing the gap (positions 0-4 through 3-4, i.e. any window
        # that spans indices 3-4 in the record list) are skipped.
        # The last window [5:13] (indices 4-11 in 0-indexed window terms) doesn't
        # span the gap; since records 5-11 are all 600 s apart, that window is valid.
        # Only 1 valid window: records[4:12]
        assert len(predictions) == 1
        assert mock_detect.call_count == 1


# ---------------------------------------------------------------------------
# 6. PredictionRecord fields are populated correctly
# ---------------------------------------------------------------------------


class TestReplayPredictionFields:
    def test_prediction_record_fields(self) -> None:
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        captures = _make_records(8)
        last_ts = captures[-1].frame_ts

        mock_result = DetectionResult(
            rain_incoming=True,
            rain_at_location=False,
            confidence=Confidence.DEGRADED,
            tracked_cells=[MagicMock()],
            arrival_time=datetime.fromtimestamp(last_ts + 1800, tz=timezone.utc),
            frame_count=8,
            max_approaching_intensity=0.65,
            last_labeled=None,
        )
        with patch("scripts.backtest.replay.detect", return_value=mock_result):
            predictions = engine.replay(captures)

        assert len(predictions) == 1
        pred = predictions[0]

        assert pred.window_end_ts == last_ts
        assert pred.rain_incoming is True
        assert pred.rain_at_location is False
        assert pred.confidence == "degraded"
        assert pred.cell_count == 1
        assert pred.max_intensity == pytest.approx(0.65)
        # arrival is 30 min after last frame
        assert pred.arrival_minutes == pytest.approx(30.0)


class TestReplayPredictionFieldsNoArrival:
    def test_prediction_record_no_arrival_time(self) -> None:
        """When arrival_time is None, arrival_minutes should be None."""
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        captures = _make_records(8)
        mock_result = _mock_detect_result()  # arrival_time=None
        with patch("scripts.backtest.replay.detect", return_value=mock_result):
            predictions = engine.replay(captures)
        assert predictions[0].arrival_minutes is None


# ---------------------------------------------------------------------------
# 7. qc_enabled=False → detect() called without confidence_maps
# ---------------------------------------------------------------------------


class TestReplayQcDisabled:
    def test_qc_disabled_calls_detect_without_confidence_maps(self) -> None:
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        captures = _make_records(8)
        mock_result = _mock_detect_result()
        with patch("scripts.backtest.replay.detect", return_value=mock_result) as mock_detect:
            engine.replay(captures)

        assert mock_detect.call_count == 1
        _, kwargs = mock_detect.call_args
        assert kwargs.get("confidence_maps") is None


class TestReplayQcEnabled:
    def test_qc_enabled_calls_detect_with_confidence_maps(self) -> None:
        """When qc_enabled=True, detect() is called with a non-None confidence_maps list."""
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=True))
        captures = _make_records(8)
        mock_result = _mock_detect_result()
        # We also need to mock CapturedRadarFrame.get_intensity_grid so it
        # returns a zero grid (tiles don't exist on disk)
        import numpy as np
        fake_grid = np.zeros((512, 512), dtype=np.float32)
        with (
            patch("scripts.backtest.replay.detect", return_value=mock_result) as mock_detect,
            patch(
                "scripts.backtest.captured_frame.CapturedRadarFrame.get_intensity_grid",
                return_value=fake_grid,
            ),
        ):
            engine.replay(captures)

        assert mock_detect.call_count == 1
        _, kwargs = mock_detect.call_args
        confidence_maps = kwargs.get("confidence_maps")
        assert confidence_maps is not None
        assert len(confidence_maps) == 8


# ---------------------------------------------------------------------------
# 8. Real Melbourne_dry tiles → rain_incoming=False
# ---------------------------------------------------------------------------

_FIXTURES = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "golden_v2"
    / "Melbourne_dry"
    / "bronze"
    / "tiles"
)

# The 4 analysis tiles that _build_analysis_tiles() produces for Melbourne_dry
# at zoom 7 (same as used in test_captured_frame.py)
_MEL_ANALYSIS_TILES = [(115, 78), (116, 78), (115, 79), (116, 79)]

# All 13 frame timestamps available in the fixture
_MEL_TIMESTAMPS = [
    1776725400,
    1776726000,
    1776726600,
    1776727200,
    1776727800,
    1776728400,
    1776729000,
    1776729600,
    1776730200,
    1776730800,
    1776731400,
    1776732000,
    1776732600,
]


def _mel_dry_capture(frame_ts: int) -> CaptureRecord:
    tile_dir = _FIXTURES / str(frame_ts)
    tile_paths = {
        (x, y): tile_dir / f"7_{x}_{y}_s6.png"
        for x, y in _MEL_ANALYSIS_TILES
    }
    return CaptureRecord(
        capture_utc=datetime.fromtimestamp(frame_ts, tz=timezone.utc),
        frame_ts=frame_ts,
        tile_paths=tile_paths,
        location_name="Melbourne_dry",
        lat=-37.8136,
        lon=144.9631,
        zoom=7,
    )


@pytest.mark.skipif(
    not _FIXTURES.exists(),
    reason="Golden v2 Melbourne_dry fixtures not present",
)
class TestReplayWithRealTiles:
    def test_melbourne_dry_predicts_no_rain(self) -> None:
        """End-to-end: replay 13 dry Melbourne frames through detect().

        All predictions must have rain_incoming=False - no rain is approaching.
        """
        captures = [_mel_dry_capture(ts) for ts in _MEL_TIMESTAMPS]
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))
        predictions = engine.replay(captures)

        # 13 frames → 6 windows (indices 0-7, 1-8, ..., 5-12)
        assert len(predictions) == 6

        for pred in predictions:
            assert pred.rain_incoming is False, (
                f"Window ending at ts={pred.window_end_ts} incorrectly "
                f"predicted rain_incoming=True for dry Melbourne data"
            )

    def test_melbourne_dry_no_rain_would_fail_with_wrong_data(self) -> None:
        """Inverse validation: a synthetic result that claims rain_incoming=True
        would be caught by the assertion above.

        This confirms the assertion in test_melbourne_dry_predicts_no_rain is
        genuinely load-bearing and not vacuously true.
        """
        captures = [_mel_dry_capture(ts) for ts in _MEL_TIMESTAMPS]
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))

        # Inject a fake detect() that always says rain is incoming
        from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult

        def _always_rain(*args, **kwargs):
            return DetectionResult(
                rain_incoming=True,
                rain_at_location=False,
                confidence=Confidence.NORMAL,
                tracked_cells=[],
                arrival_time=None,
                frame_count=8,
                max_approaching_intensity=0.5,
                last_labeled=None,
            )

        with patch("scripts.backtest.replay.detect", side_effect=_always_rain):
            predictions = engine.replay(captures)

        # Every prediction should be rain_incoming=True (injected false positives)
        # The test asserts this would FAIL the assertion in the production test above
        assert all(p.rain_incoming is True for p in predictions), (
            "Inverse validation failed: expected all predictions to be rain_incoming=True "
            "when detect() is mocked to always return rain"
        )


# ---------------------------------------------------------------------------
# Frame caching: overlapping windows must reuse CapturedRadarFrame instances
# ---------------------------------------------------------------------------


class TestFrameCachingAcrossWindows:
    """Frames shared between overlapping windows must not be rebuilt.

    Without caching, each window creates new CapturedRadarFrame objects,
    re-reading and re-decoding tile PNGs for every overlapping frame.
    With 920 captures and window_size=8, the same tile gets decoded
    ~8x per frame instead of once.
    """

    def test_frame_from_record_called_once_per_capture(self):
        """_frame_from_record must be called exactly len(captures) times,
        not len(captures) * windows_containing_that_capture times."""
        captures = _make_records(10)  # 3 windows of 8
        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))

        with patch("scripts.backtest.replay.detect", return_value=_mock_detect_result()):
            with patch(
                "scripts.backtest.replay._frame_from_record",
                wraps=__import__("scripts.backtest.replay", fromlist=["_frame_from_record"])._frame_from_record,
            ) as mock_frame:
                engine.replay(captures)

        # Should be called once per capture (10), not once per window slot (3*8=24)
        assert mock_frame.call_count == len(captures), (
            f"_frame_from_record called {mock_frame.call_count} times for "
            f"{len(captures)} captures - frames are being rebuilt per window "
            f"instead of cached"
        )


# ---------------------------------------------------------------------------
# Replay from manifest: only replay specific windows, not all consecutive
# ---------------------------------------------------------------------------


class TestReplayManifest:
    """replay_manifest must replay only the windows listed in the manifest."""

    def test_replays_only_manifest_windows(self):
        """Only windows whose window_end_ts matches a manifest entry should be replayed."""
        from scripts.backtest.manifest import ManifestEntry

        captures = _make_records(10)  # ts: BASE+0 through BASE+5400
        # Pick only 2 specific windows (window_end_ts = ts of last frame in window)
        manifest_entries = [
            ManifestEntry(location="Melbourne_dry", window_end_ts=_BASE_TS + 7 * 600,
                          category="hit", subcategory="strong"),
            ManifestEntry(location="Melbourne_dry", window_end_ts=_BASE_TS + 9 * 600,
                          category="false_alarm", subcategory="overhead_noise"),
        ]

        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))

        with patch("scripts.backtest.replay.detect", return_value=_mock_detect_result()):
            predictions = engine.replay_manifest(captures, manifest_entries)

        assert len(predictions) == 2, (
            f"Expected 2 predictions for 2 manifest entries, got {len(predictions)}"
        )
        # Verify the correct window_end_ts values
        pred_ts = {p.window_end_ts for p in predictions}
        assert _BASE_TS + 7 * 600 in pred_ts
        assert _BASE_TS + 9 * 600 in pred_ts


class TestReplayManifestOnlyLoadNeededCaptures:
    """replay_manifest must only build frames for captures that are part of
    a manifest window, not all captures in the dataset."""

    def test_frame_from_record_called_only_for_needed_captures(self):
        """With 100 captures but a manifest selecting 2 windows (needing ~16 captures),
        _frame_from_record should be called ~16 times, not 100."""
        from scripts.backtest.manifest import ManifestEntry

        captures = _make_records(100)
        # Select 2 windows: one at index 7 (needs captures 0-7) and one at index 15 (needs 8-15)
        manifest_entries = [
            ManifestEntry(location="Melbourne_dry", window_end_ts=_BASE_TS + 7 * 600,
                          category="hit", subcategory="strong"),
            ManifestEntry(location="Melbourne_dry", window_end_ts=_BASE_TS + 15 * 600,
                          category="miss", subcategory="popup"),
        ]

        engine = ReplayEngine(ReplayConfig(window_size=8, qc_enabled=False))

        with patch("scripts.backtest.replay.detect", return_value=_mock_detect_result()):
            with patch(
                "scripts.backtest.replay._frame_from_record",
                wraps=__import__("scripts.backtest.replay", fromlist=["_frame_from_record"])._frame_from_record,
            ) as mock_frame:
                engine.replay_manifest(captures, manifest_entries)

        # 2 windows × 8 frames = 16 unique captures needed (no overlap in this case)
        assert mock_frame.call_count <= 16, (
            f"_frame_from_record called {mock_frame.call_count} times for "
            f"2 manifest windows - should only build frames for needed captures, not all {len(captures)}"
        )


# ---------------------------------------------------------------------------
# 9. inspect_window: extract diagnostic trace for a target window
# ---------------------------------------------------------------------------


class TestInspectWindow:
    def test_inspect_window_returns_prediction_and_trace_for_target_window(self) -> None:
        """inspect_window runs detect() on the window ending at window_end_ts.

        Returns both the PredictionRecord and the DiagnosticTrace that was
        passed to detect()."""
        captures = _make_records(8)
        target_ts = captures[-1].frame_ts

        def populate_trace(frames, location, config, confidence_maps=None, diagnostics=None):
            """Mock detect() that populates the trace if provided."""
            if diagnostics is not None:
                for _ in frames:
                    diagnostics.frames.append(FrameDiagnostic(cell_count=2))
                diagnostics.decision = DecisionDiagnostic(
                    rain_incoming=True,
                    arrival_minutes=15.0,
                    rain_at_location=False,
                )
            return _mock_detect_result(rain_incoming=True)

        engine = ReplayEngine(ReplayConfig(qc_enabled=False))
        with patch("scripts.backtest.replay.detect", side_effect=populate_trace):
            prediction, trace = engine.inspect_window(captures, window_end_ts=target_ts)

        assert isinstance(prediction, PredictionRecord)
        assert prediction.rain_incoming is True
        assert isinstance(trace, DiagnosticTrace)
        assert len(trace.frames) == 8
        assert trace.frames[0].cell_count == 2
        assert trace.decision is not None
        assert trace.decision.rain_incoming is True
        assert trace.decision.arrival_minutes == 15.0
