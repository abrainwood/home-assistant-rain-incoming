"""Unit tests for scripts.backtest.selector - representative window selection."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file from a list of dicts."""
    fields = ["location", "window_end_ts", "predicted_rain", "actual_rain",
              "predicted_arrival_minutes", "actual_arrival_minutes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_rows(location: str, n_hits: int, n_misses: int, n_fa: int, n_cn: int) -> list[dict]:
    """Build synthetic CSV rows with known category distribution."""
    rows = []
    ts = 1776200000
    for i in range(n_hits):
        rows.append({"location": location, "window_end_ts": ts, "predicted_rain": "True",
                      "actual_rain": "True", "predicted_arrival_minutes": str(i * 5),
                      "actual_arrival_minutes": str(i * 5 + 2)})
        ts += 600
    for i in range(n_misses):
        rows.append({"location": location, "window_end_ts": ts, "predicted_rain": "False",
                      "actual_rain": "True", "predicted_arrival_minutes": "",
                      "actual_arrival_minutes": str(i * 10)})
        ts += 600
    for i in range(n_fa):
        rows.append({"location": location, "window_end_ts": ts, "predicted_rain": "True",
                      "actual_rain": "False", "predicted_arrival_minutes": str(i * 3),
                      "actual_arrival_minutes": ""})
        ts += 600
    for i in range(n_cn):
        rows.append({"location": location, "window_end_ts": ts, "predicted_rain": "False",
                      "actual_rain": "False", "predicted_arrival_minutes": "",
                      "actual_arrival_minutes": ""})
        ts += 600
    return rows


class TestSelectRepresentativeWindows:
    def test_selects_from_all_categories(self, tmp_path):
        """Selector must pick windows from hits, misses, FA, and CN."""
        from scripts.backtest.selector import select_representative_windows

        _write_csv(tmp_path / "darwin.csv", _make_rows("darwin", 50, 20, 30, 100))

        entries = select_representative_windows(tmp_path, n_per_category=5)

        categories = {e.category for e in entries}
        assert "hit" in categories
        assert "miss" in categories
        assert "false_alarm" in categories
        assert "correct_negative" in categories

    def test_respects_n_per_category(self, tmp_path):
        """Must select at most n_per_category per category per location."""
        from scripts.backtest.selector import select_representative_windows

        _write_csv(tmp_path / "darwin.csv", _make_rows("darwin", 50, 20, 30, 100))

        entries = select_representative_windows(tmp_path, n_per_category=5)
        darwin_entries = [e for e in entries if e.location == "darwin"]

        for cat in ["hit", "miss", "false_alarm", "correct_negative"]:
            count = sum(1 for e in darwin_entries if e.category == cat)
            assert count <= 5, f"Expected at most 5 {cat}, got {count}"

    def test_takes_all_when_fewer_than_n(self, tmp_path):
        """When a category has fewer than N windows, take all of them."""
        from scripts.backtest.selector import select_representative_windows

        _write_csv(tmp_path / "darwin.csv", _make_rows("darwin", 3, 2, 1, 100))

        entries = select_representative_windows(tmp_path, n_per_category=20)
        darwin_entries = [e for e in entries if e.location == "darwin"]

        hits = [e for e in darwin_entries if e.category == "hit"]
        misses = [e for e in darwin_entries if e.category == "miss"]
        fa = [e for e in darwin_entries if e.category == "false_alarm"]
        assert len(hits) == 3
        assert len(misses) == 2
        assert len(fa) == 1

    def test_multiple_locations(self, tmp_path):
        """Selector must process all CSV files in the directory."""
        from scripts.backtest.selector import select_representative_windows

        _write_csv(tmp_path / "darwin.csv", _make_rows("darwin", 10, 5, 5, 20))
        _write_csv(tmp_path / "cairns.csv", _make_rows("cairns", 10, 5, 5, 20))

        entries = select_representative_windows(tmp_path, n_per_category=3)

        locations = {e.location for e in entries}
        assert "darwin" in locations
        assert "cairns" in locations
