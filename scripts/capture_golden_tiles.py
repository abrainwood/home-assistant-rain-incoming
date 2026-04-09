#!/usr/bin/env python3
"""Capture raw RainViewer tile PNGs for golden test data.

Fetches the actual tile PNG bytes from RainViewer for a given location,
exactly as the coordinator would fetch them.

Usage:
    .venv/bin/python scripts/capture_golden_tiles.py -36.381 145.399 "Shepparton"
    .venv/bin/python scripts/capture_golden_tiles.py -33.701 151.209 "sydney_test"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import aiohttp.resolver

# Add project root to path so we can import custom_components without HA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from custom_components.incoming_rain.const import (
    RAINVIEWER_ANALYSIS_GRID,
    RAINVIEWER_COLOUR_SCHEME,
    RAINVIEWER_ZOOM,
)
from custom_components.incoming_rain.providers.rainviewer import (
    MANIFEST_URL,
    TILE_BASE_URL,
    TILE_SIZE,
)
from custom_components.incoming_rain.radar.composite import render_composite
from custom_components.incoming_rain.radar.geo import lat_lon_to_tile

ZOOM = RAINVIEWER_ZOOM
GRID_HALF = RAINVIEWER_ANALYSIS_GRID
DETECTION_SCHEME = RAINVIEWER_COLOUR_SCHEME  # 6 - what the detection pipeline uses
RENDER_SCHEME = 2  # what the renderer uses for visual composites
RADIUS_KM = 128
OUTPUT_SIZE = 640


def _tile_coords(lat: float, lon: float) -> list[tuple[int, int]]:
    """Return the (2*GRID_HALF+1)^2 tile coordinates centered on the location."""
    cx, cy = lat_lon_to_tile(lat, lon, ZOOM)
    coords = []
    for dy in range(-GRID_HALF, GRID_HALF + 1):
        for dx in range(-GRID_HALF, GRID_HALF + 1):
            coords.append((cx + dx, cy + dy))
    return coords


async def _fetch_tile_bytes(
    session: aiohttp.ClientSession,
    frame_path: str,
    zoom: int,
    tx: int,
    ty: int,
    colour_scheme: int,
) -> bytes:
    """Fetch a single tile PNG and return raw bytes."""
    url = f"{TILE_BASE_URL}{frame_path}/{TILE_SIZE}/{zoom}/{tx}/{ty}/{colour_scheme}/0.png"
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read()


async def capture(lat: float, lon: float, name: str) -> None:
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "tests" / "fixtures" / "golden_v2" / name
    bronze_dir = base_dir / "bronze"
    silver_dir = base_dir / "silver"

    print(f"Capturing golden data for {name} ({lat}, {lon})")
    print(f"  Output: {base_dir}")

    connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Fetch manifest
        print("  Fetching manifest...")
        async with session.get(MANIFEST_URL) as resp:
            resp.raise_for_status()
            manifest = await resp.json(content_type=None)

        # Save bronze manifest
        bronze_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = bronze_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  Saved manifest ({len(json.dumps(manifest))} bytes)")

        # 2. Determine tiles and frames
        tile_coords = _tile_coords(lat, lon)
        cx, cy = lat_lon_to_tile(lat, lon, ZOOM)
        past_frames = manifest.get("radar", {}).get("past", [])
        nowcast_frames = manifest.get("radar", {}).get("nowcast", [])
        all_frames = past_frames + nowcast_frames

        print(f"  Frames: {len(past_frames)} past + {len(nowcast_frames)} nowcast = {len(all_frames)}")
        print(f"  Tiles per frame: {len(tile_coords)} ({2 * GRID_HALF + 1}x{2 * GRID_HALF + 1} grid at zoom {ZOOM})")
        print(f"  Center tile: ({cx}, {cy})")

        # 3. Save capture info
        capture_info = {
            "location": {"lat": lat, "lon": lon, "name": name},
            "capture_time_utc": datetime.now(timezone.utc).isoformat(),
            "zoom": ZOOM,
            "center_tile": {"x": cx, "y": cy},
            "grid_half": GRID_HALF,
            "tile_coords": [{"x": x, "y": y} for x, y in tile_coords],
            "detection_colour_scheme": DETECTION_SCHEME,
            "render_colour_scheme": RENDER_SCHEME,
            "frame_count": len(all_frames),
            "past_frame_count": len(past_frames),
            "nowcast_frame_count": len(nowcast_frames),
        }
        info_path = bronze_dir / "capture_info.json"
        info_path.write_text(json.dumps(capture_info, indent=2))

        # 4. Fetch and save all tiles for every frame
        for i, frame_entry in enumerate(all_frames):
            ts = frame_entry["time"]
            path = frame_entry["path"]
            frame_type = "past" if i < len(past_frames) else "nowcast"
            tile_dir = bronze_dir / "tiles" / str(ts)
            tile_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Frame {i + 1}/{len(all_frames)} (ts={ts}, {frame_type}): ", end="", flush=True)

            # Fetch detection scheme tiles (scheme 6)
            detection_tasks = [
                _fetch_tile_bytes(session, path, ZOOM, tx, ty, DETECTION_SCHEME)
                for tx, ty in tile_coords
            ]
            # Fetch render scheme tiles (scheme 2)
            render_tasks = [
                _fetch_tile_bytes(session, path, ZOOM, tx, ty, RENDER_SCHEME)
                for tx, ty in tile_coords
            ]

            all_tile_results = await asyncio.gather(
                *detection_tasks, *render_tasks, return_exceptions=True,
            )

            n_tiles = len(tile_coords)
            detection_results = all_tile_results[:n_tiles]
            render_results = all_tile_results[n_tiles:]

            saved = 0
            errors = 0
            for j, (tx, ty) in enumerate(tile_coords):
                # Detection scheme tile
                det_data = detection_results[j]
                if isinstance(det_data, Exception):
                    errors += 1
                    print(f"E", end="", flush=True)
                else:
                    tile_file = tile_dir / f"{ZOOM}_{tx}_{ty}_s{DETECTION_SCHEME}.png"
                    tile_file.write_bytes(det_data)
                    saved += 1

                # Render scheme tile
                ren_data = render_results[j]
                if isinstance(ren_data, Exception):
                    errors += 1
                    print(f"E", end="", flush=True)
                else:
                    tile_file = tile_dir / f"{ZOOM}_{tx}_{ty}_s{RENDER_SCHEME}.png"
                    tile_file.write_bytes(ren_data)
                    saved += 1

            print(f" {saved} tiles saved", end="")
            if errors:
                print(f", {errors} errors", end="")
            print()

        # 5. Render silver frames (visual composites WITHOUT QC)
        silver_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = silver_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        print("  Rendering silver frames (no QC)...")
        for i, frame_entry in enumerate(all_frames):
            ts = frame_entry["time"]
            path = frame_entry["path"]
            utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hhmm = utc_dt.strftime("%H%M")
            frame_label = f"frame_{i:02d}_{hhmm}utc"

            png_bytes = await render_composite(
                lat=lat,
                lon=lon,
                radius_km=RADIUS_KM,
                frame_path=path,
                output_size=OUTPUT_SIZE,
                session=session,
            )

            frame_file = frames_dir / f"{frame_label}.png"
            frame_file.write_bytes(png_bytes)
            print(f"    {frame_file.name} ({len(png_bytes)} bytes)")

        # 6. Build silver manifest (copy of bronze with classification fields)
        silver_manifest = json.loads(json.dumps(manifest))
        for section in ("past", "nowcast"):
            entries = silver_manifest.get("radar", {}).get(section, [])
            for entry in entries:
                entry["classification"] = "TODO"

        silver_manifest_path = silver_dir / "manifest.json"
        silver_manifest_path.write_text(json.dumps(silver_manifest, indent=2))

    print()
    print(f"Done! Golden data saved to: {base_dir}")
    print(f"  Bronze: {bronze_dir}")
    print(f"  Silver: {silver_dir}")
    print()
    print("Next steps:")
    print("  1. Open the silver frames in Finder to inspect them")
    print("  2. Edit silver/manifest.json to classify each frame")
    print("     (set 'classification' to 'clear', 'light', 'moderate', 'heavy', etc.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture raw RainViewer tiles for golden test data.",
    )
    parser.add_argument("lat", type=float, help="Latitude")
    parser.add_argument("lon", type=float, help="Longitude")
    parser.add_argument("name", type=str, help="Location name (used as directory name)")

    args = parser.parse_args()
    asyncio.run(capture(args.lat, args.lon, args.name))


if __name__ == "__main__":
    main()
