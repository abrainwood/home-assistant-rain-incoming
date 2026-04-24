"""Unit tests for scripts.backtest.cli."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.backtest.replay import PredictionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_META_JSON = {
    "capture_utc": "2026-04-21T10:00:00Z",
    "frame_ts": 1776700000,
    "frame_path": "/v2/radar/test",
    "location": {"name": "test_loc", "lat": -33.0, "lon": 151.0},
    "zoom": 7,
    "tiles": [
        {"x": 115, "y": 78, "file": "1000_115_78.png"},
        {"x": 116, "y": 78, "file": "1000_116_78.png"},
        {"x": 115, "y": 79, "file": "1000_115_79.png"},
        {"x": 116, "y": 79, "file": "1000_116_79.png"},
    ],
    "bytes_total": 1234,
}


def _make_location_dir(captures_dir: Path, name: str) -> Path:
    """Create a location directory with one synthetic meta.json."""
    loc_dir = captures_dir / name
    date_dir = loc_dir / "2026-04-21"
    date_dir.mkdir(parents=True)
    meta_path = date_dir / "1000_meta.json"
    meta_path.write_text(json.dumps(_META_JSON))
    return loc_dir


def _make_prediction(
    rain_incoming: bool = True,
    rain_at_location: bool = False,
    arrival_minutes: float | None = 12.0,
    confidence: str = "normal",
    cell_count: int = 3,
    max_intensity: float = 0.5,
    window_end_ts: int = 1776700000,
) -> PredictionRecord:
    return PredictionRecord(
        window_end_ts=window_end_ts,
        rain_incoming=rain_incoming,
        rain_at_location=rain_at_location,
        arrival_minutes=arrival_minutes,
        confidence=confidence,
        cell_count=cell_count,
        max_intensity=max_intensity,
    )


# ---------------------------------------------------------------------------
# Test: no args -> argparse exits with code 2
# ---------------------------------------------------------------------------


class TestMainWithNoArgs:
    def test_main_with_no_args_exits_with_error(self) -> None:
        """Calling main() with no arguments triggers argparse SystemExit(2)."""
        from scripts.backtest.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Test: missing data-dir -> exits with code 1 + error message
# ---------------------------------------------------------------------------


class TestMainWithMissingDataDir:
    def test_main_with_missing_data_dir_exits(
        self, tmp_path: Path, capsys
    ) -> None:
        """A non-existent --data-dir causes SystemExit(1) and an error on stderr."""
        from scripts.backtest.cli import main

        missing = tmp_path / "no_such_dir"

        with pytest.raises(SystemExit) as exc_info:
            main(["--data-dir", str(missing)])
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.err


# ---------------------------------------------------------------------------
# Test: discovers all location dirs when --locations=all
# ---------------------------------------------------------------------------


class TestMainDiscoversAllLocations:
    def test_main_discovers_all_locations(
        self, tmp_path: Path, capsys
    ) -> None:
        """With --locations=all both location subdirs are processed."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "loc1")
        _make_location_dir(captures_dir, "loc2")

        predictions = [_make_prediction(rain_incoming=False, arrival_minutes=None)]

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.replay.return_value = predictions

            main(["--data-dir", str(data_dir)])

        assert instance.replay.call_count == 2

        captured = capsys.readouterr()
        assert "loc1" in captured.out
        assert "loc2" in captured.out


# ---------------------------------------------------------------------------
# Test: --locations=loc1 with two dirs -> only loc1 processed
# ---------------------------------------------------------------------------


class TestMainFiltersSpecificLocations:
    def test_main_filters_specific_locations(
        self, tmp_path: Path, capsys
    ) -> None:
        """--locations=loc1 with two location dirs processes only loc1."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "loc1")
        _make_location_dir(captures_dir, "loc2")

        predictions = [_make_prediction()]

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.replay.return_value = predictions

            main(["--data-dir", str(data_dir), "--locations", "loc1"])

        assert instance.replay.call_count == 1

        captured = capsys.readouterr()
        assert "loc1" in captured.out
        assert "loc2" not in captured.out


# ---------------------------------------------------------------------------
# Test: --locations=nonexistent -> exits with code 1
# ---------------------------------------------------------------------------


class TestMainMissingLocationExits:
    def test_main_missing_location_exits(
        self, tmp_path: Path, capsys
    ) -> None:
        """--locations pointing at a non-existent dir causes SystemExit(1)."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        captures_dir.mkdir(parents=True)

        with pytest.raises(SystemExit) as exc_info:
            main(["--data-dir", str(data_dir), "--locations", "nonexistent"])
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.err


# ---------------------------------------------------------------------------
# Test: --qc=none disables QC in ReplayConfig
# ---------------------------------------------------------------------------


class TestMainQcNoneDisablesQc:
    def test_main_qc_none_disables_qc(
        self, tmp_path: Path, capsys
    ) -> None:
        """--qc=none sets qc_enabled=False in the ReplayConfig passed to ReplayEngine."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "loc1")

        predictions = [_make_prediction()]

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine, \
             patch("scripts.backtest.cli.ReplayConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            instance = MockEngine.return_value
            instance.replay.return_value = predictions

            main(["--data-dir", str(data_dir), "--qc", "none"])

        MockConfig.assert_called_once()
        _, kwargs = MockConfig.call_args
        assert kwargs.get("qc_enabled") is False


# ---------------------------------------------------------------------------
# Test: stdout contains location name, window count, rain percentage
# ---------------------------------------------------------------------------


class TestMainPrintsSummary:
    def test_main_prints_summary(
        self, tmp_path: Path, capsys
    ) -> None:
        """Output includes location name, total windows, and rain percentage."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "sydney")

        predictions = [
            _make_prediction(rain_incoming=True, window_end_ts=1776700000),
            _make_prediction(rain_incoming=False, window_end_ts=1776700600),
            _make_prediction(rain_incoming=True, window_end_ts=1776701200),
        ]

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.replay.return_value = predictions

            main(["--data-dir", str(data_dir)])

        captured = capsys.readouterr()
        # Location name in summary line
        assert "sydney" in captured.out
        # 3 windows total
        assert "3 windows" in captured.out
        # 2 out of 3 rain -> 67%
        assert "2 rain" in captured.out
        assert "67%" in captured.out


# ---------------------------------------------------------------------------
# Test: empty location (no meta.json) prints "no captures found"
# ---------------------------------------------------------------------------


class TestMainSkipsEmptyLocation:
    def test_main_skips_empty_location(
        self, tmp_path: Path, capsys
    ) -> None:
        """A location dir with no *_meta.json prints 'no captures found' and skips replay."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"

        # Create a location dir with NO meta.json - just an empty date dir
        empty_loc = captures_dir / "empty_loc"
        (empty_loc / "2026-04-21").mkdir(parents=True)

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value

            main(["--data-dir", str(data_dir)])

        # replay() must NOT have been called for this empty location
        instance.replay.assert_not_called()

        captured = capsys.readouterr()
        assert "empty_loc" in captured.out
        assert "no captures found" in captured.out


# ---------------------------------------------------------------------------
# Test: --verify flag prints scorecard (POD/FAR/CSI)
# ---------------------------------------------------------------------------


class TestVerifyFlagPrintsScorecard:
    def test_verify_flag_prints_scorecard(
        self, tmp_path: Path, capsys
    ) -> None:
        """With --verify, output contains POD/FAR/CSI scorecard instead of per-prediction ts= lines."""
        from scripts.backtest.cli import main
        from scripts.backtest.metrics import ContingencyTable, ScoreCard

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "sydney")

        predictions = [_make_prediction(rain_incoming=True, window_end_ts=1776700000)]

        scorecard = ScoreCard(
            location="sydney",
            contingency=ContingencyTable(hits=1, misses=0, false_alarms=0, correct_negatives=0),
            total_windows=2,
            skipped_gaps=1,
            lead_time_errors=[5.0],
        )

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine, \
             patch("scripts.backtest.cli.FutureRadarVerifier") as MockVerifier, \
             patch("scripts.backtest.cli.compute_scorecard", return_value=scorecard):
            instance = MockEngine.return_value
            instance.replay.return_value = predictions
            verifier_instance = MockVerifier.return_value
            verifier_instance.verify.return_value = []

            main(["--data-dir", str(data_dir), "--verify"])

        captured = capsys.readouterr()
        assert "POD" in captured.out
        assert "FAR" in captured.out
        assert "CSI" in captured.out
        # Should NOT contain per-prediction ts= lines
        assert "ts=" not in captured.out


# ---------------------------------------------------------------------------
# Test: without --verify, output contains per-prediction ts= lines
# ---------------------------------------------------------------------------


class TestVerifyFlagNotSetPrintsPredictions:
    def test_verify_flag_not_set_prints_predictions(
        self, tmp_path: Path, capsys
    ) -> None:
        """Without --verify, output contains per-prediction ts= lines (existing behavior)."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "sydney")

        predictions = [_make_prediction(rain_incoming=True, window_end_ts=1776700000)]

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.replay.return_value = predictions

            main(["--data-dir", str(data_dir)])

        captured = capsys.readouterr()
        assert "ts=" in captured.out
        assert "POD" not in captured.out


# ---------------------------------------------------------------------------
# Test: --verify with no predictions prints a zeros scorecard
# ---------------------------------------------------------------------------


class TestVerifyWithNoPredictionsPrintsZeros:
    def test_verify_with_no_predictions_prints_zeros(
        self, tmp_path: Path, capsys
    ) -> None:
        """A location with no valid windows still prints a scorecard with all zeros."""
        from scripts.backtest.cli import main
        from scripts.backtest.metrics import ContingencyTable, ScoreCard

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "empty_loc")

        scorecard = ScoreCard(
            location="empty_loc",
            contingency=ContingencyTable(hits=0, misses=0, false_alarms=0, correct_negatives=0),
            total_windows=0,
            skipped_gaps=0,
            lead_time_errors=[],
        )

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine, \
             patch("scripts.backtest.cli.FutureRadarVerifier") as MockVerifier, \
             patch("scripts.backtest.cli.compute_scorecard", return_value=scorecard):
            instance = MockEngine.return_value
            instance.replay.return_value = []  # no predictions
            verifier_instance = MockVerifier.return_value
            verifier_instance.verify.return_value = []

            main(["--data-dir", str(data_dir), "--verify"])

        captured = capsys.readouterr()
        assert "POD" in captured.out
        assert "no verified arrival times" in captured.out


# ---------------------------------------------------------------------------
# Test: --dump-errors prints misses and false alarms
# ---------------------------------------------------------------------------


class TestDumpErrors:
    def test_dump_errors_prints_misses_and_false_alarms(
        self, tmp_path: Path, capsys
    ) -> None:
        """--dump-errors must list misses and false alarms with timestamps."""
        from scripts.backtest.cli import main
        from scripts.backtest.metrics import (
            ContingencyTable, ScoreCard, VerificationRecord,
        )

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "test_loc")

        records = [
            VerificationRecord(window_end_ts=1000, predicted_rain=True, actual_rain=True),   # hit
            VerificationRecord(window_end_ts=2000, predicted_rain=False, actual_rain=True),  # miss
            VerificationRecord(window_end_ts=3000, predicted_rain=True, actual_rain=False),  # FA
            VerificationRecord(window_end_ts=4000, predicted_rain=False, actual_rain=False), # CN
        ]
        scorecard = ScoreCard(
            location="test_loc",
            contingency=ContingencyTable(hits=1, misses=1, false_alarms=1, correct_negatives=1),
            total_windows=4,
            skipped_gaps=0,
            lead_time_errors=[],
        )

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine, \
             patch("scripts.backtest.cli.FutureRadarVerifier") as MockVerifier, \
             patch("scripts.backtest.cli.compute_scorecard", return_value=scorecard):
            instance = MockEngine.return_value
            instance.replay.return_value = [MagicMock()]
            verifier_instance = MockVerifier.return_value
            verifier_instance.verify.return_value = records

            main(["--data-dir", str(data_dir), "--verify", "--dump-errors"])

        captured = capsys.readouterr()
        # Should contain scorecard
        assert "POD" in captured.out
        # Should contain misses section with timestamp
        assert "MISSES" in captured.out
        assert "2000" in captured.out
        # Should contain false alarms section with timestamp
        assert "FALSE ALARMS" in captured.out
        assert "3000" in captured.out
        # Should NOT contain hits or correct negatives
        assert "1000" not in captured.out or "hit" not in captured.out.lower().split("1000")[0][-20:]
        assert "4000" not in captured.out or "correct" not in captured.out.lower()

    def test_dump_errors_without_verify_is_ignored(
        self, tmp_path: Path, capsys
    ) -> None:
        """--dump-errors without --verify should not crash."""
        from scripts.backtest.cli import main

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "test_loc")

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.replay.return_value = []

            main(["--data-dir", str(data_dir), "--dump-errors"])

        captured = capsys.readouterr()
        # Should not crash, should not print error sections
        assert "MISSES" not in captured.out


# ---------------------------------------------------------------------------
# Test: --output-dir writes CSV and markdown files
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_output_dir_creates_csv_and_markdown(
        self, tmp_path: Path, capsys
    ) -> None:
        """--verify --output-dir must write CSV and markdown scorecard."""
        from scripts.backtest.cli import main
        from scripts.backtest.metrics import (
            ContingencyTable, ScoreCard, VerificationRecord,
        )

        data_dir = tmp_path / "data"
        captures_dir = data_dir / "captures"
        _make_location_dir(captures_dir, "test_loc")

        output_dir = tmp_path / "reports"

        records = [
            VerificationRecord(window_end_ts=1000, predicted_rain=True, actual_rain=True),
        ]
        scorecard = ScoreCard(
            location="test_loc",
            contingency=ContingencyTable(hits=1),
            total_windows=1,
            skipped_gaps=0,
            lead_time_errors=[],
        )

        with patch("scripts.backtest.cli.ReplayEngine") as MockEngine, \
             patch("scripts.backtest.cli.FutureRadarVerifier") as MockVerifier, \
             patch("scripts.backtest.cli.compute_scorecard", return_value=scorecard):
            instance = MockEngine.return_value
            instance.replay.return_value = [MagicMock()]
            verifier_instance = MockVerifier.return_value
            verifier_instance.verify.return_value = records

            main(["--data-dir", str(data_dir), "--verify", "--output-dir", str(output_dir)])

        assert (output_dir / "test_loc.csv").exists(), "CSV file not created"
        assert (output_dir / "scorecard.md").exists(), "Markdown scorecard not created"
