"""Unit tests for scripts.backtest.report - CSV and markdown report generation."""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from scripts.backtest.metrics import (
    ContingencyTable,
    ScoreCard,
    VerificationRecord,
)


def _make_records() -> list[VerificationRecord]:
    return [
        VerificationRecord(window_end_ts=1000, predicted_rain=True, actual_rain=True,
                           predicted_arrival_minutes=30.0, actual_arrival_minutes=25.0),
        VerificationRecord(window_end_ts=2000, predicted_rain=False, actual_rain=True,
                           actual_arrival_minutes=10.0),
        VerificationRecord(window_end_ts=3000, predicted_rain=True, actual_rain=False,
                           predicted_arrival_minutes=45.0),
        VerificationRecord(window_end_ts=4000, predicted_rain=False, actual_rain=False),
    ]


class TestWriteVerificationCsv:
    def test_csv_has_header_and_data_rows(self, tmp_path):
        """CSV must have a header row plus one row per record."""
        from scripts.backtest.report import write_verification_csv

        records = _make_records()
        csv_path = tmp_path / "results.csv"
        write_verification_csv(records, "darwin", csv_path)

        lines = csv_path.read_text().splitlines()
        assert len(lines) == 5  # 1 header + 4 records

        reader = csv.DictReader(StringIO(csv_path.read_text()))
        rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["location"] == "darwin"
        assert rows[0]["window_end_ts"] == "1000"
        assert rows[0]["predicted_rain"] == "True"
        assert rows[0]["actual_rain"] == "True"
        assert rows[0]["predicted_arrival_minutes"] == "30.0"
        assert rows[0]["actual_arrival_minutes"] == "25.0"

    def test_csv_none_values_are_empty(self, tmp_path):
        """None arrival times should be empty strings in CSV, not 'None'."""
        from scripts.backtest.report import write_verification_csv

        records = [VerificationRecord(window_end_ts=1000, predicted_rain=False, actual_rain=False)]
        csv_path = tmp_path / "results.csv"
        write_verification_csv(records, "test", csv_path)

        reader = csv.DictReader(StringIO(csv_path.read_text()))
        row = next(reader)
        assert row["predicted_arrival_minutes"] == ""
        assert row["actual_arrival_minutes"] == ""


class TestWriteMarkdownScorecard:
    def test_scorecard_contains_metrics(self, tmp_path):
        """Markdown scorecard must contain POD, FAR, CSI, Bias and contingency table."""
        from scripts.backtest.report import write_scorecard_markdown

        scorecard = ScoreCard(
            location="darwin",
            contingency=ContingencyTable(hits=99, misses=33, false_alarms=129, correct_negatives=627),
            total_windows=916,
            skipped_gaps=28,
            lead_time_errors=[-7.0, 0.0, 5.0],
        )
        md_path = tmp_path / "scorecard.md"
        write_scorecard_markdown([scorecard], md_path)

        content = md_path.read_text()
        assert "darwin" in content
        assert "POD" in content
        assert "FAR" in content
        assert "CSI" in content
        assert "99" in content  # hits
        assert "33" in content  # misses
        assert "129" in content  # false alarms

    def test_multiple_locations_in_scorecard(self, tmp_path):
        """Scorecard with multiple locations should include all of them."""
        from scripts.backtest.report import write_scorecard_markdown

        scorecards = [
            ScoreCard(location="darwin", contingency=ContingencyTable(hits=10),
                      total_windows=20, skipped_gaps=0, lead_time_errors=[]),
            ScoreCard(location="cairns", contingency=ContingencyTable(hits=5),
                      total_windows=15, skipped_gaps=0, lead_time_errors=[]),
        ]
        md_path = tmp_path / "scorecard.md"
        write_scorecard_markdown(scorecards, md_path)

        content = md_path.read_text()
        assert "darwin" in content
        assert "cairns" in content
