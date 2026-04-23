"""Report generation for the backtesting framework.

Writes CSV verification records and markdown scorecards.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.backtest.metrics import ScoreCard, VerificationRecord


_CSV_FIELDS = [
    "location",
    "window_end_ts",
    "predicted_rain",
    "actual_rain",
    "predicted_arrival_minutes",
    "actual_arrival_minutes",
]


def write_verification_csv(
    records: list[VerificationRecord],
    location: str,
    path: Path,
) -> None:
    """Write verification records to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "location": location,
                "window_end_ts": r.window_end_ts,
                "predicted_rain": r.predicted_rain,
                "actual_rain": r.actual_rain,
                "predicted_arrival_minutes": r.predicted_arrival_minutes if r.predicted_arrival_minutes is not None else "",
                "actual_arrival_minutes": r.actual_arrival_minutes if r.actual_arrival_minutes is not None else "",
            })


def write_scorecard_markdown(
    scorecards: list[ScoreCard],
    path: Path,
) -> None:
    """Write a markdown report with per-location scorecards."""
    lines = ["# Backtest Results", ""]

    lines.append("| Location | Windows | POD | FAR | CSI | Bias | Hits | Misses | FA | CN |")
    lines.append("|----------|---------|-----|-----|-----|------|------|--------|----|----|")

    for sc in scorecards:
        ct = sc.contingency
        lines.append(
            f"| {sc.location} | {sc.total_windows} "
            f"| {sc.pod:.3f} | {sc.far:.3f} | {sc.csi:.3f} | {sc.bias:.2f} "
            f"| {ct.hits} | {ct.misses} | {ct.false_alarms} | {ct.correct_negatives} |"
        )

    # Lead time summary per location
    lines.extend(["", "## Lead Time Errors", ""])
    for sc in scorecards:
        mean = sc.mean_lead_time_error
        median = sc.median_lead_time_error
        if mean is not None and median is not None:
            n = len(sc.lead_time_errors)
            lines.append(f"- **{sc.location}**: mean={mean:.1f}min, median={median:.1f}min (n={n})")
        else:
            lines.append(f"- **{sc.location}**: no verified arrival times")

    lines.append("")
    path.write_text("\n".join(lines))
