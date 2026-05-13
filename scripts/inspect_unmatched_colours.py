"""Diagnostic for GH issue #180 (revised): list unique RGB values in the
'unmatched' band (d > 60) of a live RainViewer tile, sorted by frequency.

If RainViewer scheme 2 emits multiple distinct colours that we currently
classify as land-mask, one or more may actually be low-dBZ precipitation
levels we're incorrectly dropping. That would explain why our rendered
composite shows sparser precipitation than BOM's raw radar.

NOT production code. One-shot diagnostic.

Run:
    python scripts/inspect_unmatched_colours.py
    python scripts/inspect_unmatched_colours.py --tx 117 --ty 76 --top 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from custom_components.rain_incoming.providers.rainviewer import (  # noqa: E402
    MAX_COLOUR_DISTANCE,
    PRECIP_COLOURS,
    TILE_SIZE,
)

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_BASE_URL = "https://tilecache.rainviewer.com"
COLOUR_SCHEME = 2

_PRECIP_RGB = np.array([[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32)


def fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=15) as resp:
        return resp.read()


def get_recent_frame_paths(n: int = 1) -> list[tuple[int, str]]:
    raw = fetch_bytes(MANIFEST_URL)
    manifest = json.loads(raw)
    past = manifest["radar"]["past"]
    return [(entry["time"], entry["path"]) for entry in past[-n:]]


def fetch_tile(frame_path: str, zoom: int, tx: int, ty: int) -> np.ndarray:
    url = f"{TILE_BASE_URL}{frame_path}/{TILE_SIZE}/{zoom}/{tx}/{ty}/{COLOUR_SCHEME}/0.png"
    data = fetch_bytes(url)
    return np.array(Image.open(BytesIO(data)).convert("RGBA"), dtype=np.uint8)


def distance_to_palette(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    diff = rgb[:, :, np.newaxis, :] - _PRECIP_RGB[np.newaxis, np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).min(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=int, default=117)
    parser.add_argument("--ty", type=int, default=76)
    parser.add_argument("--zoom", type=int, default=7)
    parser.add_argument("--top", type=int, default=30, help="Show top-N colours by count")
    parser.add_argument("--num-tiles", type=int, default=4, help="Sample N adjacent tiles for better coverage")
    args = parser.parse_args()

    frames = get_recent_frame_paths(n=1)
    _, frame_path = frames[0]
    print(f"Frame path: {frame_path}")
    print(f"Sampling {args.num_tiles} tiles around ({args.tx}, {args.ty}) at zoom {args.zoom}")
    print()

    print(f"Documented palette (we treat these as precipitation, d <= {MAX_COLOUR_DISTANCE}):")
    for r, g, b, intensity in PRECIP_COLOURS:
        print(f"  ({r:>3}, {g:>3}, {b:>3})  intensity={intensity:.2f}")
    print()

    # Sample several adjacent tiles to get broad coverage of land/water/precip
    tiles: list[np.ndarray] = []
    coords = [
        (args.tx, args.ty),
        (args.tx + 1, args.ty),
        (args.tx, args.ty + 1),
        (args.tx + 1, args.ty + 1),
    ][:args.num_tiles]
    for tx, ty in coords:
        try:
            tiles.append(fetch_tile(frame_path, args.zoom, tx, ty))
        except Exception as e:
            print(f"  tile ({tx},{ty}) fetch failed: {e}")

    all_rgba = np.concatenate([t.reshape(-1, 4) for t in tiles], axis=0)
    opaque = all_rgba[:, 3] > 10
    distances_flat = distance_to_palette(all_rgba.reshape(1, -1, 4))[0]

    unmatched_mask = opaque & (distances_flat > MAX_COLOUR_DISTANCE)
    unmatched = all_rgba[unmatched_mask][:, :3]

    matched_mask = opaque & (distances_flat <= MAX_COLOUR_DISTANCE)
    matched = all_rgba[matched_mask][:, :3]

    print(f"Total opaque pixels across {len(tiles)} tiles: {int(opaque.sum())}")
    print(f"  Matched (d <= 60):       {int(matched_mask.sum())}")
    print(f"  Unmatched (d > 60):      {int(unmatched_mask.sum())}")
    print()

    # Unique colours in the unmatched (currently treated as land-mask) band
    unique, counts = np.unique(unmatched, axis=0, return_counts=True)
    order = np.argsort(-counts)
    print(f"Top {args.top} unique RGB values in the d > 60 band (currently dropped as land-mask):")
    print(f"  {'RGB':<22} {'count':>8} {'% of band':>10} {'nearest_palette':>24} {'dist':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*24} {'-'*8}")
    total_unmatched = unmatched.shape[0]
    for i in order[: args.top]:
        rgb = unique[i]
        count = counts[i]
        # Distance to nearest palette entry
        diffs = _PRECIP_RGB - rgb.astype(np.float32)
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        nearest_idx = int(dists.argmin())
        nearest = _PRECIP_RGB[nearest_idx].astype(int).tolist()
        dist = float(dists.min())
        print(
            f"  ({rgb[0]:>3}, {rgb[1]:>3}, {rgb[2]:>3})      "
            f"{count:>8} {100*count/total_unmatched:>9.2f}% "
            f"({nearest[0]:>3},{nearest[1]:>3},{nearest[2]:>3})  {dist:>7.1f}"
        )
    print()

    # And for completeness: unique colours in matched band, to confirm we
    # see all 9 palette colours showing up.
    unique_m, counts_m = np.unique(matched, axis=0, return_counts=True)
    order_m = np.argsort(-counts_m)
    print(f"Unique RGB values in the matched (d <= 60) band, top 15:")
    print(f"  {'RGB':<22} {'count':>8} {'nearest_palette':>24} {'dist':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*24} {'-'*8}")
    for i in order_m[:15]:
        rgb = unique_m[i]
        count = counts_m[i]
        diffs = _PRECIP_RGB - rgb.astype(np.float32)
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        nearest_idx = int(dists.argmin())
        nearest = _PRECIP_RGB[nearest_idx].astype(int).tolist()
        dist = float(dists.min())
        print(
            f"  ({rgb[0]:>3}, {rgb[1]:>3}, {rgb[2]:>3})      "
            f"{count:>8} ({nearest[0]:>3},{nearest[1]:>3},{nearest[2]:>3})  {dist:>7.1f}"
        )


if __name__ == "__main__":
    main()
