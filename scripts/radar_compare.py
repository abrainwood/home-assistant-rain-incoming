#!/usr/bin/env python3
"""
Generate a radar comparison report for any location.

Fetches live RainViewer data, runs the full QC and detection pipelines,
renders an animated GIF, and writes a text report. Useful for assessing
QC quality without eyeballing raw GIFs.

Usage:
    .venv/bin/python scripts/radar_compare.py                              # Terry Hills default
    .venv/bin/python scripts/radar_compare.py -33.701 151.209 "Sydney"     # positional
    .venv/bin/python scripts/radar_compare.py --lat -36.381 --lon 145.399 --name "Shepparton"
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import aiohttp
import aiohttp.resolver
import numpy as np

# Add project root to path so we can import custom_components without HA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from custom_components.rain_incoming.const import (
    INTENSITY_THRESHOLD,
    MAX_ANGULAR_VARIANCE_RADIANS,
    MAX_STORM_SPEED_KMH,
    MIN_CELL_AREA_PIXELS,
    MIN_TEMPORAL_FRAMES,
    PROXIMITY_RADIUS_KM,
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_TILE_SIZE,
    RAINVIEWER_ZOOM,
)
from custom_components.rain_incoming.providers.base import BoundingBox
from custom_components.rain_incoming.providers.rainviewer import (
    MANIFEST_URL,
    RainViewerFrame,
    RainViewerProvider,
)
from custom_components.rain_incoming.radar.composite import render_animated_composite
from custom_components.rain_incoming.radar.detector import DetectorConfig, detect
from custom_components.rain_incoming.radar.qc import (
    QCConfig,
    compute_confidence_map,
    refine_confidence_post_detection,
)

# Suppress expected QC warnings about missing factor scores (speed/motion/clutter
# aren't available in the pre-detection pass - that's by design)
logging.getLogger("custom_components.rain_incoming.radar.qc.scoring").setLevel(
    logging.ERROR
)

DEFAULT_LAT = -33.701
DEFAULT_LON = 151.209
DEFAULT_NAME = "Terry Hills"

FRAMES_TO_FETCH = 8
RADIUS_KM = 128
OUTPUT_SIZE = 640
GIF_FRAME_DURATION_MS = 500
LOOKAHEAD_MINUTES = 60


def _build_analysis_bounds(lat: float, lon: float) -> BoundingBox:
    tile_deg = 360.0 / (2 ** RAINVIEWER_ZOOM)
    half = tile_deg * RAINVIEWER_ANALYSIS_GRID
    return BoundingBox(
        lat_min=lat - half,
        lat_max=lat + half,
        lon_min=lon - half,
        lon_max=lon + half,
    )


def _build_detector_config(lat: float, lon: float) -> DetectorConfig:
    bounds = _build_analysis_bounds(lat, lon)
    grid_side = RAINVIEWER_TILE_SIZE * (2 * RAINVIEWER_ANALYSIS_GRID + 1)
    return DetectorConfig(
        lookahead_seconds=LOOKAHEAD_MINUTES * 60,
        intensity_threshold=INTENSITY_THRESHOLD,
        min_cell_area_pixels=MIN_CELL_AREA_PIXELS,
        min_temporal_frames=MIN_TEMPORAL_FRAMES,
        max_angular_variance=MAX_ANGULAR_VARIANCE_RADIANS,
        max_storm_speed_kmh=MAX_STORM_SPEED_KMH,
        proximity_radius_km=PROXIMITY_RADIUS_KM,
        analysis_bounds=bounds,
        grid_width=grid_side,
        grid_height=grid_side,
    )


def _classify_intensity(max_intensity: float) -> str:
    """Classify intensity using the same labels as the HA sensor entity."""
    from custom_components.rain_incoming.sensor import _intensity_label
    return _intensity_label(max_intensity)


def _build_report(
    name: str,
    lat: float,
    lon: float,
    local_time: datetime,
    grids: list[np.ndarray],
    confidence_maps: list[np.ndarray],
    detection_result,
) -> str:
    latest_grid = grids[-1]
    total_pixels = latest_grid.size
    echo_mask = latest_grid > 0.0
    echo_count = int(echo_mask.sum())
    max_intensity = float(latest_grid.max()) if echo_count > 0 else 0.0
    mean_echo = float(latest_grid[echo_mask].mean()) if echo_count > 0 else 0.0

    latest_conf = confidence_maps[-1]
    echo_conf = latest_conf[echo_mask]
    conf_mean = float(echo_conf.mean()) if echo_conf.size > 0 else 0.0
    conf_min = float(echo_conf.min()) if echo_conf.size > 0 else 0.0
    conf_max = float(echo_conf.max()) if echo_conf.size > 0 else 0.0
    cubed = latest_conf ** 3
    cubed_echo = cubed[echo_mask]
    cubed_mean = float(cubed_echo.mean()) if cubed_echo.size > 0 else 0.0

    tz_label = local_time.strftime("%Z") or "UTC"

    intensity_label = _classify_intensity(detection_result.max_approaching_intensity)
    arrival_str = (
        detection_result.arrival_time.isoformat()
        if detection_result.arrival_time
        else "None"
    )

    lines = [
        f"Location: {name} ({lat}, {lon})",
        f"Time: {local_time.strftime('%Y-%m-%d %H:%M')} {tz_label}",
        "",
        "Raw Radar:",
        f"  Echo pixels: {echo_count} / {total_pixels}",
        f"  Max intensity: {max_intensity:.2f}",
        f"  Mean intensity (echo only): {mean_echo:.3f}",
        "",
        "QC Confidence (echo pixels):",
        f"  Mean: {conf_mean:.3f}",
        f"  Min: {conf_min:.3f}",
        f"  Max: {conf_max:.3f}",
        f"  After cubing: mean {cubed_mean:.3f}",
        "",
        "Detection:",
        f"  rain_incoming: {detection_result.rain_incoming}",
        f"  tracked_cells: {len(detection_result.tracked_cells)}",
        f"  arrival_time: {arrival_str}",
        f"  intensity: {intensity_label}",
    ]

    if detection_result.tracked_cells:
        lines.append("")
        lines.append("Cells:")
        for i, cell in enumerate(detection_result.tracked_cells, 1):
            lines.append(
                f"  #{i}: ({cell.lat:.2f}, {cell.lon:.2f}) "
                f"speed={cell.velocity_kmh:.0f}km/h "
                f"bearing={cell.bearing:.0f} "
                f"conf={cell.confidence:.2f}"
            )

    lines.append("")  # trailing newline
    return "\n".join(lines)


async def run_comparison(lat: float, lon: float, name: str) -> None:
    print(f"Running comparison for {name} ({lat}, {lon})...")

    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Fetch radar frames (directly, bypassing RainViewerProvider which
        #    creates its own session without ThreadedResolver)
        print("  Fetching radar frames...")
        async with session.get(MANIFEST_URL) as resp:
            resp.raise_for_status()
            manifest = await resp.json(content_type=None)

        past = manifest.get("radar", {}).get("past", [])
        selected = past[-FRAMES_TO_FETCH:]
        frames = [
            RainViewerFrame(
                timestamp=datetime.fromtimestamp(entry["time"], tz=timezone.utc),
                path=entry["path"],
                zoom=RainViewerProvider.ZOOM,
                colour_scheme=RainViewerProvider.COLOUR_SCHEME,
            )
            for entry in selected
        ]
        print(f"  Got {len(frames)} frames")

        if len(frames) < 2:
            print("  ERROR: Not enough frames for analysis")
            return

        # 2. Fetch tile grids
        config = _build_detector_config(lat, lon)
        bounds = config.analysis_bounds
        W, H = config.grid_width, config.grid_height

        print("  Fetching intensity grids...")
        fetch_tasks = [
            frame._fetch_stitched_grid(bounds, W, H, session)
            for frame in frames
        ]

        await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # 3. Extract grids and run QC
        print("  Running QC pipeline...")
        grids = [f.get_intensity_grid(bounds, W, H) for f in frames]
        qc_config = QCConfig()

        confidence_maps: list[np.ndarray] = []
        for i, grid in enumerate(grids):
            grids_up_to_i = grids[: i + 1]
            cmap = compute_confidence_map(
                grid,
                config=qc_config,
                grids=grids_up_to_i,
            )
            confidence_maps.append(cmap.confidence)

        # 4. Run detection
        print("  Running detection pipeline...")
        result = detect(
            frames=frames,
            location=(lat, lon),
            config=config,
            confidence_maps=confidence_maps,
        )

        # Post-detection refinement
        if result.tracked_cells and result.last_labeled is not None and confidence_maps:
            refined = refine_confidence_post_detection(
                pre_confidence=confidence_maps[-1],
                cells=result.tracked_cells,
                labeled=result.last_labeled,
                config=qc_config,
            )
            confidence_maps[-1] = refined

        # 5. Render GIF
        print("  Rendering animated GIF...")
        import zoneinfo
        try:
            local_tz = zoneinfo.ZoneInfo("Australia/Sydney")
        except Exception:
            local_tz = timezone.utc

        local_timestamps = [ts.astimezone(local_tz) for ts in [f.timestamp for f in frames]]
        frame_paths = [f.path for f in frames]

        gif_bytes = await render_animated_composite(
            lat=lat,
            lon=lon,
            radius_km=RADIUS_KM,
            frame_paths=frame_paths,
            output_size=OUTPUT_SIZE,
            frame_duration_ms=GIF_FRAME_DURATION_MS,
            frame_timestamps=local_timestamps,
            tz_name="Australia/Sydney",
            confidence_maps=confidence_maps,
            session=session,
        )

        # 6. Build report
        now_local = datetime.now(local_tz)
        report = _build_report(
            name, lat, lon, now_local, grids, confidence_maps, result,
        )

    # 7. Write output
    timestamp_str = now_local.strftime("%Y-%m-%dT%H%M")
    safe_name = name.lower().replace(" ", "_")
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "reports", f"{timestamp_str}_{safe_name}"
    )
    os.makedirs(output_dir, exist_ok=True)

    gif_path = os.path.join(output_dir, "radar_128km.gif")
    with open(gif_path, "wb") as f:
        f.write(gif_bytes)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n  Output: {os.path.abspath(output_dir)}")
    print(f"  GIF:    {os.path.basename(gif_path)}")
    print(f"  Report: {os.path.basename(report_path)}")
    print()
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate radar comparison report for a location.",
    )
    parser.add_argument("lat", nargs="?", type=float, default=None, help="Latitude")
    parser.add_argument("lon", nargs="?", type=float, default=None, help="Longitude")
    parser.add_argument("name", nargs="?", type=str, default=None, help="Location name")
    parser.add_argument("--lat", dest="lat_flag", type=float, default=None, help="Latitude (flag)")
    parser.add_argument("--lon", dest="lon_flag", type=float, default=None, help="Longitude (flag)")
    parser.add_argument("--name", dest="name_flag", type=str, default=None, help="Location name (flag)")

    args = parser.parse_args()

    lat = args.lat_flag if args.lat_flag is not None else args.lat
    lon = args.lon_flag if args.lon_flag is not None else args.lon
    name = args.name_flag if args.name_flag is not None else args.name

    if lat is None:
        lat = DEFAULT_LAT
    if lon is None:
        lon = DEFAULT_LON
    if name is None:
        name = DEFAULT_NAME

    asyncio.run(run_comparison(lat, lon, name))


if __name__ == "__main__":
    main()
