"""CLI entry point for the backtesting framework.

Wires together DataLoader and ReplayEngine so captures can be replayed
from the command line::

    python -m scripts.backtest --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from scripts.backtest.data_loader import load_captures
from scripts.backtest.metrics import ScoreCard, VerificationRecord, compute_scorecard
from scripts.backtest.replay import ReplayConfig, ReplayEngine
from scripts.backtest.report import (
    compare_scorecards,
    load_scorecard_json,
    write_scorecard_json,
    write_scorecard_markdown,
    write_verification_csv,
)
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


def _print_errors(records: list[VerificationRecord]) -> None:
    """Print individual misses and false alarms with timestamps."""
    from datetime import datetime, timezone

    misses = [r for r in records if not r.predicted_rain and r.actual_rain]
    false_alarms = [r for r in records if r.predicted_rain and not r.actual_rain]

    if misses:
        print(f"\n  MISSES ({len(misses)}) - rain occurred but not predicted:")
        for r in misses:
            dt = datetime.fromtimestamp(r.window_end_ts, tz=timezone.utc)
            arrival = f"arrived after {r.actual_arrival_minutes:.0f}min" if r.actual_arrival_minutes is not None else "arrival time unknown"
            print(f"    ts={r.window_end_ts}  {dt:%Y-%m-%d %H:%M}Z  {arrival}")

    if false_alarms:
        print(f"\n  FALSE ALARMS ({len(false_alarms)}) - rain predicted but did not occur:")
        for r in false_alarms:
            dt = datetime.fromtimestamp(r.window_end_ts, tz=timezone.utc)
            predicted = f"predicted in {r.predicted_arrival_minutes:.0f}min" if r.predicted_arrival_minutes is not None else "no arrival estimate"
            print(f"    ts={r.window_end_ts}  {dt:%Y-%m-%d %H:%M}Z  {predicted}")

    if not misses and not false_alarms:
        print("\n  No errors - perfect forecast!")


def _build_replay_config(args: argparse.Namespace) -> ReplayConfig:
    """Build a ReplayConfig from parsed CLI args, applying optional overrides."""
    kwargs: dict = {
        "window_size": args.window_size,
        "max_gap_seconds": args.max_gap_seconds,
        "qc_enabled": args.qc == "full",
        "lookahead_seconds": args.lookahead_minutes * 60,
    }
    if args.intensity_threshold is not None:
        kwargs["intensity_threshold"] = args.intensity_threshold
    if args.min_cell_area is not None:
        kwargs["min_cell_area_pixels"] = args.min_cell_area
    if args.use_acceleration:
        kwargs["use_acceleration"] = True
    if args.use_intensity_trend:
        kwargs["use_intensity_trend"] = True
    if args.frame_scale_by_lookahead:
        kwargs["frame_scale_by_lookahead"] = True
    return ReplayConfig(**kwargs)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m scripts.backtest`."""
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Replay captured radar data through the detection pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory containing captures/ and observations/",
    )
    parser.add_argument(
        "--generate-subset",
        type=Path,
        default=None,
        help="Generate a manifest from a previous run's CSV directory, then exit",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=20,
        help="With --generate-subset, number of windows per category per location",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="With --generate-subset, path to write the manifest JSON",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Replay only windows listed in this manifest JSON",
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
    parser.add_argument("--lookahead-minutes", type=int, default=30)
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify predictions against future captures and print a scorecard",
    )
    parser.add_argument(
        "--dump-errors",
        action="store_true",
        default=False,
        help="With --verify, list individual misses and false alarms with timestamps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="With --verify, write CSV and markdown reports to this directory",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare against a previous run's output directory (requires scorecards.json)",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=None,
        help="Override the intensity threshold for both detection and verification (default: production value)",
    )
    parser.add_argument(
        "--min-cell-area",
        type=int,
        default=None,
        help="Override the minimum cell area in pixels for detection (default: production value)",
    )
    parser.add_argument(
        "--use-acceleration",
        action="store_true",
        default=False,
        help="Enable acceleration-aware cell projection (experimental)",
    )
    parser.add_argument(
        "--use-intensity-trend",
        action="store_true",
        default=False,
        help="Suppress detection of cells whose intensity is sharply declining (experimental)",
    )
    parser.add_argument(
        "--frame-scale-by-lookahead",
        action="store_true",
        default=False,
        help="Scale min_temporal_frames by lookahead (2 frames at <=20min, 3 at <=40min, 4 at >40min)",
    )
    parser.add_argument(
        "--inspect",
        nargs=2,
        metavar=("LOCATION", "TIMESTAMP"),
        default=None,
        help="Run detect() on a single window and print the diagnostic trace as JSON. Args: location_name, window_end_ts",
    )
    parser.add_argument(
        "--inspect-set",
        type=Path,
        default=None,
        help="Run --inspect on each window in a manifest file. Outputs a JSON list of traces.",
    )

    args = parser.parse_args(argv)

    # Handle --generate-subset mode (doesn't need --data-dir)
    if args.generate_subset:
        from scripts.backtest.selector import select_representative_windows
        from scripts.backtest.manifest import save_manifest

        entries = select_representative_windows(args.generate_subset, args.n_per_category)
        output_path = args.manifest_output or Path("manifest.json")
        save_manifest(entries, "quick", "Auto-generated curated subset",
                      str(args.generate_subset), output_path)
        print(f"Generated manifest with {len(entries)} windows at {output_path}")
        return

    # Handle --inspect-set mode
    if args.inspect_set is not None:
        from scripts.backtest.manifest import load_manifest

        manifest = load_manifest(args.inspect_set)

        # Group entries by location (preserve manifest order)
        entries_by_location: dict[str, list] = {}
        for entry in manifest.entries:
            entries_by_location.setdefault(entry.location, []).append(entry)

        config = _build_replay_config(args)
        engine = ReplayEngine(config)

        captures_dir = args.data_dir / "captures"
        traces_list: list[dict] = []
        for location_name, loc_entries in entries_by_location.items():
            loc_dir = captures_dir / location_name
            captures = load_captures(loc_dir)
            for entry in loc_entries:
                _prediction, trace = engine.inspect_window(
                    captures, window_end_ts=entry.window_end_ts
                )
                traces_list.append(dataclasses.asdict(trace))

        print(json.dumps(traces_list, indent=2, default=str))
        return

    if args.data_dir is None:
        print("Error: --data-dir is required for replay mode", file=sys.stderr)
        sys.exit(1)

    captures_dir = args.data_dir / "captures"
    if not captures_dir.is_dir():
        print(f"Error: {captures_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Handle --inspect mode: run detect() on a single window and dump the trace as JSON
    if args.inspect:
        location_name, ts_str = args.inspect
        window_end_ts = int(ts_str)
        loc_dir = captures_dir / location_name
        captures = load_captures(loc_dir)
        config = _build_replay_config(args)
        engine = ReplayEngine(config)
        _prediction, trace = engine.inspect_window(captures, window_end_ts=window_end_ts)
        print(json.dumps(dataclasses.asdict(trace), indent=2, default=str))
        return

    # Discover locations
    if args.locations == "all":
        location_dirs = sorted(p for p in captures_dir.iterdir() if p.is_dir())
    else:
        location_dirs = [captures_dir / name for name in args.locations.split(",")]
        for d in location_dirs:
            if not d.is_dir():
                print(f"Error: location directory {d} not found", file=sys.stderr)
                sys.exit(1)

    config = _build_replay_config(args)
    engine = ReplayEngine(config)
    all_scorecards: list[ScoreCard] = []
    all_records: dict[str, list[VerificationRecord]] = {}

    # Load manifest if --subset specified
    manifest_by_location: dict[str, list] | None = None
    if args.subset:
        from scripts.backtest.manifest import load_manifest
        manifest = load_manifest(args.subset)
        manifest_by_location = {}
        for entry in manifest.entries:
            manifest_by_location.setdefault(entry.location, []).append(entry)
        print(f"Using manifest: {manifest.name} ({len(manifest.entries)} windows)")

    for loc_dir in location_dirs:
        location_name = loc_dir.name
        captures = load_captures(loc_dir)
        if not captures:
            print(f"{location_name}: no captures found, skipping")
            continue

        if manifest_by_location is not None:
            loc_entries = manifest_by_location.get(location_name, [])
            if not loc_entries:
                continue  # no manifest entries for this location
            predictions = engine.replay_manifest(captures, loc_entries)
        else:
            predictions = engine.replay(captures)

        if args.verify:
            if manifest_by_location is not None:
                total_possible_windows = len(manifest_by_location.get(location_name, []))
            else:
                total_possible_windows = max(0, len(captures) - config.window_size + 1)
            skipped_gaps = total_possible_windows - len(predictions)
            # In manifest mode, only pass captures needed for verification
            # (the windows themselves + lookahead period) to avoid loading
            # all 1000+ captures into the verifier's frame cache
            if manifest_by_location is not None and predictions:
                min_ts = min(p.window_end_ts for p in predictions)
                max_ts = max(p.window_end_ts for p in predictions) + config.lookahead_seconds
                verify_captures = [c for c in captures if min_ts <= c.frame_ts <= max_ts]
            else:
                verify_captures = captures
            from scripts.backtest.verifier import VerifierConfig
            verifier_config = VerifierConfig(
                lookahead_seconds=config.lookahead_seconds,
                intensity_threshold=config.intensity_threshold,
            )
            verifier = FutureRadarVerifier(verify_captures, config=verifier_config)
            records = verifier.verify(predictions)
            scorecard = compute_scorecard(
                location_name, records, total_possible_windows, skipped_gaps
            )
            _print_scorecard(scorecard)
            if args.dump_errors:
                _print_errors(records)
            all_scorecards.append(scorecard)
            all_records[location_name] = records
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

    # Write reports if --output-dir and --verify were both specified
    if args.output_dir and all_scorecards:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for location_name, records in all_records.items():
            write_verification_csv(records, location_name, args.output_dir / f"{location_name}.csv")
        write_scorecard_markdown(all_scorecards, args.output_dir / "scorecard.md")
        write_scorecard_json(all_scorecards, args.output_dir / "scorecards.json")
        print(f"\nReports written to {args.output_dir}")

    # Compare against previous run if requested
    if args.compare and all_scorecards:
        prev_json = args.compare / "scorecards.json"
        if prev_json.exists():
            previous = load_scorecard_json(prev_json)
            print(compare_scorecards(all_scorecards, previous))
        else:
            print(f"\nWarning: {prev_json} not found, skipping comparison", file=sys.stderr)
