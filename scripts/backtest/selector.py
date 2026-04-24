"""Select representative windows from a baseline run for quick backtesting.

Reads per-location CSVs and picks N windows per category, spread across
the score distribution so the curated set covers the important scenarios.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.backtest.manifest import ManifestEntry


def _classify_subcategory(row: dict) -> str:
    """Assign a subcategory based on arrival times."""
    predicted = row.get("predicted_arrival_minutes", "")
    actual = row.get("actual_arrival_minutes", "")
    category = _classify_category(row)

    if category == "hit":
        # Strong vs marginal based on lead time error
        if predicted and actual:
            error = abs(float(predicted) - float(actual))
            return "strong" if error <= 10 else "marginal"
        return "strong"

    if category == "false_alarm":
        if predicted:
            mins = float(predicted)
            if mins <= 0:
                return "overhead_noise"
            if mins <= 25:
                return "near_miss"
            return "dissipated"
        return "overhead_noise"

    if category == "miss":
        if actual:
            mins = float(actual)
            return "approaching_undetected" if mins <= 30 else "popup"
        return "popup"

    # correct_negative
    return "dead_dry"


def _classify_category(row: dict) -> str:
    """Classify a CSV row into hit/miss/false_alarm/correct_negative."""
    predicted = row["predicted_rain"] == "True"
    actual = row["actual_rain"] == "True"
    if predicted and actual:
        return "hit"
    if not predicted and actual:
        return "miss"
    if predicted and not actual:
        return "false_alarm"
    return "correct_negative"


def _spread_select(items: list, n: int) -> list:
    """Select N items evenly spread across the list.

    Takes items at regular intervals including first and last,
    giving a representative sample of the distribution.
    """
    if len(items) <= n:
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[int(round(i * step))] for i in range(n)]


def select_representative_windows(
    csv_dir: Path,
    n_per_category: int = 20,
) -> list[ManifestEntry]:
    """Select representative windows from baseline CSVs.

    For each location CSV, groups rows by category and selects N
    windows spread across the distribution.
    """
    entries: list[ManifestEntry] = []

    for csv_path in sorted(csv_dir.glob("*.csv")):
        # Group rows by category
        by_category: dict[str, list[dict]] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                cat = _classify_category(row)
                by_category.setdefault(cat, []).append(row)

        location = csv_path.stem

        for category, rows in by_category.items():
            selected = _spread_select(rows, n_per_category)
            for row in selected:
                entries.append(ManifestEntry(
                    location=location,
                    window_end_ts=int(row["window_end_ts"]),
                    category=category,
                    subcategory=_classify_subcategory(row),
                ))

    return entries
