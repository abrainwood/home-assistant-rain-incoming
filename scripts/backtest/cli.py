"""CLI entry point for the backtesting framework.

Wires together DataLoader and ReplayEngine so captures can be replayed
from the command line::

    python -m scripts.backtest --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.backtest.data_loader import load_captures
from scripts.backtest.metrics import ScoreCard, compute_scorecard
from scripts.backtest.replay import ReplayConfig, ReplayEngine
from scripts.backtest.verifier import FutureRadarVerifier


def _print_scorecard(scorecard: ScoreCard) -> None:
    """Print a scorecard summary to stdout."""
    ct = scorecard.contingency
    print(
        f"{scorecard.location}: {scorecard.total_windows} windows "
        f"({scorecard.skipped_gaps} gaps skipped)"
    )
    print(
        f"  POD:  {scorecard.pod:.3f}  FAR:  {scorecard.far:.3f}  "
        f"CSI:  {scorecard.csi:.3f}  Bias: {scorecard.bias:.2f}"
    )
    print(
        f"  Hits: {ct.hits}  Misses: {ct.misses}  "
        f"FA: {ct.false_alarms}  CN: {ct.correct_negatives}"
    )
    mean = scorecard.mean_lead_time_error
    median = scorecard.median_lead_time_error
    if mean is not None and median is not None:
        count = len(scorecard.lead_time_errors)
        print(f"  Lead time error: mean={mean:.1f}min median={median:.1f}min (n={count})")
    else:
        print("  Lead time error: no verified arrival times")


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m scripts.backtest`."""
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Replay captured radar data through the detection pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing captures/ and observations/",
    )
    parser.add_argument(
        "--locations",
        type=str,
        default="all",
        help="Comma-separated location names, or 'all'",
    )
    parser.add_argument(
        "--qc",
        choices=["full", "none"],
        default="full",
        help="QC mode: 'full' runs confidence maps, 'none' skips",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-gap-seconds", type=int, default=900)
    parser.add_argument("--lookahead-minutes", type=int, default=60)
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify predictions against future captures and print a scorecard",
    )

    args = parser.parse_args(argv)

    captures_dir = args.data_dir / "captures"
    if not captures_dir.is_dir():
        print(f"Error: {captures_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Discover locations
    if args.locations == "all":
        location_dirs = sorted(p for p in captures_dir.iterdir() if p.is_dir())
    else:
        location_dirs = [captures_dir / name for name in args.locations.split(",")]
        for d in location_dirs:
            if not d.is_dir():
                print(f"Error: location directory {d} not found", file=sys.stderr)
                sys.exit(1)

    config = ReplayConfig(
        window_size=args.window_size,
        max_gap_seconds=args.max_gap_seconds,
        qc_enabled=(args.qc == "full"),
        lookahead_seconds=args.lookahead_minutes * 60,
    )
    engine = ReplayEngine(config)

    for loc_dir in location_dirs:
        location_name = loc_dir.name
        captures = load_captures(loc_dir)
        if not captures:
            print(f"{location_name}: no captures found, skipping")
            continue

        predictions = engine.replay(captures)

        if args.verify:
            total_possible_windows = max(0, len(captures) - config.window_size + 1)
            skipped_gaps = total_possible_windows - len(predictions)
            verifier = FutureRadarVerifier(captures)
            records = verifier.verify(predictions)
            scorecard = compute_scorecard(
                location_name, records, total_possible_windows, skipped_gaps
            )
            _print_scorecard(scorecard)
        else:
            total = len(predictions)
            if total:
                rain_count = sum(1 for p in predictions if p.rain_incoming)
                print(
                    f"{location_name}: {total} windows, "
                    f"{rain_count} rain ({rain_count / total * 100:.0f}%)"
                )
            else:
                print(f"{location_name}: no valid windows")

            for p in predictions:
                arrival = f"{p.arrival_minutes:.0f}min" if p.arrival_minutes is not None else "-"
                print(
                    f"  ts={p.window_end_ts} rain={p.rain_incoming} "
                    f"at_loc={p.rain_at_location} arrival={arrival} "
                    f"cells={p.cell_count} conf={p.confidence}"
                )
