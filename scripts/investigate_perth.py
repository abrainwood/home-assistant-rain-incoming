"""
Investigate Perth false-negative 2026-04-27.

Loads bronze tiles directly (no HTTP), builds RainViewerFrame objects with
pre-populated grids, runs detect() with DiagnosticTrace, and prints JSON.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/investigate_perth.py
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# --- paths ---
BRONZE_DIR = Path("tests/fixtures/golden_v2/Perth_false_negative_20260427/bronze")
TILES_DIR = BRONZE_DIR / "tiles"

# --- location ---
LAT, LON = -31.95, 115.86

# timestamps of interest: user saw rain at 13:10 AEST, sensor dry at 14:20 AEST
WINDOW_ENDS = [1777259400, 1777263600]
FRAMES_PER_WINDOW = 8

DETECTION_SCHEME = 6


def load_manifest():
    with open(BRONZE_DIR / "manifest.json") as f:
        d = json.load(f)
    return d["radar"]["past"]


def build_intensity_grid_from_tiles(
    ts: int,
    bounds,
    width: int,
    height: int,
    zoom: int,
) -> np.ndarray:
    """Load scheme-6 tiles for one frame timestamp, stitch, crop, resample."""
    from custom_components.rain_incoming.providers.rainviewer import (
        _tile_to_intensity_array,
        _tile_bounds,
        TILE_SIZE,
    )

    ts_dir = TILES_DIR / str(ts)
    if not ts_dir.exists():
        print(f"  WARNING: no tile dir for ts={ts}", file=sys.stderr)
        return np.zeros((height, width), dtype=np.float32)

    pattern = re.compile(rf"(\d+)_(\d+)_(\d+)_s{DETECTION_SCHEME}\.png")
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for f in ts_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            tile_zoom, tx, ty = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if tile_zoom == zoom:
                tiles[(tx, ty)] = _tile_to_intensity_array(f.read_bytes())

    if not tiles:
        print(f"  WARNING: no s{DETECTION_SCHEME} tiles at zoom {zoom} for ts={ts}", file=sys.stderr)
        return np.zeros((height, width), dtype=np.float32)

    min_tx = min(k[0] for k in tiles)
    max_tx = max(k[0] for k in tiles)
    min_ty = min(k[1] for k in tiles)
    max_ty = max(k[1] for k in tiles)
    cols = max_tx - min_tx + 1
    rows = max_ty - min_ty + 1

    canvas = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)
    for (tx, ty), arr in tiles.items():
        px = (tx - min_tx) * TILE_SIZE
        py = (ty - min_ty) * TILE_SIZE
        canvas[py: py + TILE_SIZE, px: px + TILE_SIZE] = arr

    canvas_bounds = type(bounds)(
        lat_min=_tile_bounds(min_tx, max_ty, zoom).lat_min,
        lat_max=_tile_bounds(min_tx, min_ty, zoom).lat_max,
        lon_min=_tile_bounds(min_tx, min_ty, zoom).lon_min,
        lon_max=_tile_bounds(max_tx, min_ty, zoom).lon_max,
    )

    ch, cw = canvas.shape

    def lat_to_row(lat, cb, h):
        return int((cb.lat_max - lat) / (cb.lat_max - cb.lat_min) * h)

    def lon_to_col(lon, cb, w):
        return int((lon - cb.lon_min) / (cb.lon_max - cb.lon_min) * w)

    r0 = max(0, lat_to_row(bounds.lat_max, canvas_bounds, ch))
    r1 = min(ch, lat_to_row(bounds.lat_min, canvas_bounds, ch))
    c0 = max(0, lon_to_col(bounds.lon_min, canvas_bounds, cw))
    c1 = min(cw, lon_to_col(bounds.lon_max, canvas_bounds, cw))

    if r1 <= r0 or c1 <= c0:
        print(f"  WARNING: crop result empty for ts={ts}", file=sys.stderr)
        return np.zeros((height, width), dtype=np.float32)

    cropped = canvas[r0:r1, c0:c1]
    pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
    pil_img = pil_img.resize((width, height), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0


def make_frame(ts: int, path: str, bounds, W: int, H: int, zoom: int):
    from custom_components.rain_incoming.providers.rainviewer import RainViewerFrame
    frame = RainViewerFrame(
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
        path=path,
        zoom=zoom,
        colour_scheme=DETECTION_SCHEME,
    )
    grid = build_intensity_grid_from_tiles(ts, bounds, W, H, zoom)
    frame._cached_grid = grid
    frame._cached_bounds = bounds
    return frame


def trace_to_dict(trace) -> dict:
    return dataclasses.asdict(trace)


def run_window(window_end_ts: int, all_frames_meta: list, bounds, W: int, H: int, zoom: int, config):
    from custom_components.rain_incoming.radar.detector import detect, DiagnosticTrace

    # select up to FRAMES_PER_WINDOW frames ending at window_end_ts
    eligible = [f for f in all_frames_meta if f["time"] <= window_end_ts]
    selected = eligible[-FRAMES_PER_WINDOW:]

    print(f"\n=== Window end: {window_end_ts} ({datetime.fromtimestamp(window_end_ts, tz=timezone.utc).isoformat()}) ===")
    print(f"  Using {len(selected)} frames: {[f['time'] for f in selected]}")

    frames = [make_frame(f["time"], f["path"], bounds, W, H, zoom) for f in selected]

    non_zero = [(i, float(np.max(fr._cached_grid))) for i, fr in enumerate(frames)]
    print(f"  Frame max intensities: {[(i, f'{v:.3f}') for i, v in non_zero]}")

    trace = DiagnosticTrace()
    result = detect(
        frames=frames,
        location=(LAT, LON),
        config=config,
        diagnostics=trace,
    )

    print(f"  Result: rain_incoming={result.rain_incoming}, confidence={result.confidence.name}, "
          f"arrival={result.arrival_time}, max_intensity={result.max_approaching_intensity:.3f}")
    print(f"  Trace:")
    print(json.dumps(trace_to_dict(trace), indent=2, default=str))
    return result, trace


def main():
    from custom_components.rain_incoming.coordinator import _build_analysis_bounds
    from custom_components.rain_incoming.radar.detector import DetectorConfig
    from custom_components.rain_incoming.const import (
        DEFAULT_LOOKAHEAD_MINUTES,
        INTENSITY_THRESHOLD,
        MIN_CELL_AREA_PIXELS,
        MIN_TEMPORAL_FRAMES,
        MAX_ANGULAR_VARIANCE_RADIANS,
        MAX_STORM_SPEED_KMH,
        PROXIMITY_RADIUS_KM,
        RAINVIEWER_ZOOM,
        RAINVIEWER_TILE_SIZE,
    )

    zoom = RAINVIEWER_ZOOM
    W = H = RAINVIEWER_TILE_SIZE * 2
    bounds = _build_analysis_bounds(LAT, LON)
    print(f"Analysis bounds: {bounds}")
    print(f"Grid size: {W}x{H}")

    config = DetectorConfig(
        lookahead_seconds=DEFAULT_LOOKAHEAD_MINUTES * 60,
        intensity_threshold=INTENSITY_THRESHOLD,
        min_cell_area_pixels=MIN_CELL_AREA_PIXELS,
        min_temporal_frames=MIN_TEMPORAL_FRAMES,
        max_angular_variance=MAX_ANGULAR_VARIANCE_RADIANS,
        max_storm_speed_kmh=MAX_STORM_SPEED_KMH,
        proximity_radius_km=PROXIMITY_RADIUS_KM,
        analysis_bounds=bounds,
        grid_width=W,
        grid_height=H,
    )

    all_frames_meta = load_manifest()
    print(f"Manifest: {len(all_frames_meta)} frames, ts range "
          f"{all_frames_meta[0]['time']}..{all_frames_meta[-1]['time']}")

    for ts in WINDOW_ENDS:
        run_window(ts, all_frames_meta, bounds, W, H, zoom, config)


if __name__ == "__main__":
    main()
