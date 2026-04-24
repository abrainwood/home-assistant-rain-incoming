"""Report generation for the backtesting framework.

Writes CSV verification records and markdown scorecards.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.backtest.metrics import ContingencyTable, ScoreCard, VerificationRecord


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


def write_scorecard_json(
    scorecards: list[ScoreCard],
    path: Path,
) -> None:
    """Write scorecards as JSON for machine-readable comparison."""
    data = []
    for sc in scorecards:
        ct = sc.contingency
        data.append({
            "location": sc.location,
            "total_windows": sc.total_windows,
            "skipped_gaps": sc.skipped_gaps,
            "hits": ct.hits,
            "misses": ct.misses,
            "false_alarms": ct.false_alarms,
            "correct_negatives": ct.correct_negatives,
            "lead_time_errors": sc.lead_time_errors,
        })
    path.write_text(json.dumps(data, indent=2))


def _format_delta(value: float) -> str:
    """Format a delta with sign and direction indicator."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}"


def compare_scorecards(
    current: list[ScoreCard],
    previous: list[ScoreCard],
) -> str:
    """Format a comparison table showing deltas between two runs."""
    prev_by_loc = {sc.location: sc for sc in previous}

    lines = ["", "## Comparison", ""]
    lines.append("| Location | POD | FAR | CSI | Bias |")
    lines.append("|----------|-----|-----|-----|------|")

    for sc in current:
        prev = prev_by_loc.get(sc.location)
        if prev is None:
            lines.append(
                f"| {sc.location} | {sc.pod:.3f} (new) "
                f"| {sc.far:.3f} (new) | {sc.csi:.3f} (new) | {sc.bias:.2f} (new) |"
            )
        else:
            pod_d = sc.pod - prev.pod
            far_d = sc.far - prev.far
            csi_d = sc.csi - prev.csi
            bias_d = sc.bias - prev.bias
            lines.append(
                f"| {sc.location} "
                f"| {prev.pod:.3f}→{sc.pod:.3f} ({_format_delta(pod_d)}) "
                f"| {prev.far:.3f}→{sc.far:.3f} ({_format_delta(far_d)}) "
                f"| {prev.csi:.3f}→{sc.csi:.3f} ({_format_delta(csi_d)}) "
                f"| {prev.bias:.2f}→{sc.bias:.2f} ({_format_delta(bias_d)}) |"
            )

    lines.append("")
    return "\n".join(lines)


def load_scorecard_json(path: Path) -> list[ScoreCard]:
    """Load scorecards from a JSON file."""
    data = json.loads(path.read_text())
    scorecards = []
    for entry in data:
        scorecards.append(ScoreCard(
            location=entry["location"],
            contingency=ContingencyTable(
                hits=entry["hits"],
                misses=entry["misses"],
                false_alarms=entry["false_alarms"],
                correct_negatives=entry["correct_negatives"],
            ),
            total_windows=entry["total_windows"],
            skipped_gaps=entry["skipped_gaps"],
            lead_time_errors=entry["lead_time_errors"],
        ))
    return scorecards
