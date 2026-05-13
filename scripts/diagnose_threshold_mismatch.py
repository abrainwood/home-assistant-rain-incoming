"""Diagnostic for GH issue #180: is the renderer's stricter L2 threshold
actually the cause of whole cells disappearing between frames?

NOT production code. One-shot diagnostic. Delete or convert to a regression
test once the data has confirmed/rejected the hypothesis.

Fetches two consecutive RainViewer tiles for a fixed region (default: North
Richmond / Sydney basin), computes per-pixel L2 distance to the documented
PRECIP_COLOURS palette, and reports:

- Histogram of distances over opaque pixels
- Pixel counts by band: (0, 30]   = both detector and renderer keep
                       (30, 60]  = detector keeps, renderer drops
                       (60, inf) = both drop (assumed land mask)
- Cross-frame stability of the (30, 60] band: how many pixels move into a
  different band on frame N+1
- Diagnostic PNGs:
    out_frame0.png / out_frame1.png   = original tiles
    out_band_overlay_0.png / _1.png   = original tile with (30, 60] pixels
                                        tinted red (these are what the renderer
                                        drops while the detector keeps)

Run:
    python scripts/diagnose_threshold_mismatch.py
    python scripts/diagnose_threshold_mismatch.py --tx 117 --ty 76 --zoom 7
"""
from __future__ import annotations

import argparse
import json
import sys
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
from custom_components.rain_incoming.radar.composite import (  # noqa: E402
    _FILTER_MAX_COLOUR_DISTANCE,
)

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_BASE_URL = "https://tilecache.rainviewer.com"
COLOUR_SCHEME = 2

_PRECIP_RGB = np.array([[r, g, b] for r, g, b, _ in PRECIP_COLOURS], dtype=np.float32)


def fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=15) as resp:
        return resp.read()


def get_recent_frame_paths(n: int = 2) -> list[tuple[int, str]]:
    raw = fetch_bytes(MANIFEST_URL)
    manifest = json.loads(raw)
    past = manifest["radar"]["past"]
    return [(entry["time"], entry["path"]) for entry in past[-n:]]


def tile_url(frame_path: str, zoom: int, tx: int, ty: int) -> str:
    return (
        f"{TILE_BASE_URL}{frame_path}"
        f"/{TILE_SIZE}/{zoom}/{tx}/{ty}/{COLOUR_SCHEME}/0.png"
    )


def fetch_tile(frame_path: str, zoom: int, tx: int, ty: int) -> np.ndarray:
    url = tile_url(frame_path, zoom, tx, ty)
    data = fetch_bytes(url)
    return np.array(Image.open(BytesIO(data)).convert("RGBA"), dtype=np.uint8)


def distance_to_palette(rgba: np.ndarray) -> np.ndarray:
    """Min L2 distance to PRECIP_COLOURS for each pixel. Returns (H, W) float32."""
    rgb = rgba[:, :, :3].astype(np.float32)
    diff = rgb[:, :, np.newaxis, :] - _PRECIP_RGB[np.newaxis, np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).min(axis=-1)


def opaque_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[:, :, 3] > 10


def histogram(distances: np.ndarray, opaque: np.ndarray) -> dict[str, int]:
    d = distances[opaque]
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 10_000)]
    out = {}
    for lo, hi in bands:
        key = f"({lo:>3}, {hi:>3}]" if hi != 10_000 else f"({lo:>3}, inf )"
        out[key] = int(((d > lo) & (d <= hi)).sum())
    return out


def band_label(distance: float) -> str:
    if distance <= _FILTER_MAX_COLOUR_DISTANCE:
        return "both_keep"
    if distance <= MAX_COLOUR_DISTANCE:
        return "renderer_drops"
    return "both_drop"


def band_array(distances: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Return string-coded band array same shape as distances; 'transparent' for non-opaque."""
    out = np.full(distances.shape, "transparent", dtype=object)
    out[opaque & (distances <= _FILTER_MAX_COLOUR_DISTANCE)] = "both_keep"
    out[opaque & (distances > _FILTER_MAX_COLOUR_DISTANCE) & (distances <= MAX_COLOUR_DISTANCE)] = "renderer_drops"
    out[opaque & (distances > MAX_COLOUR_DISTANCE)] = "both_drop"
    return out


def overlay_band(rgba: np.ndarray, band_mask: np.ndarray, tint: tuple[int, int, int]) -> Image.Image:
    """Return a PIL image of rgba with band_mask pixels tinted (preserves rest)."""
    out = rgba.copy()
    r, g, b = tint
    out[band_mask, 0] = r
    out[band_mask, 1] = g
    out[band_mask, 2] = b
    out[band_mask, 3] = 255
    return Image.fromarray(out, mode="RGBA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=int, default=117, help="Tile X (default 117 = Sydney basin at zoom 7)")
    parser.add_argument("--ty", type=int, default=76, help="Tile Y (default 76 = Sydney basin at zoom 7)")
    parser.add_argument("--zoom", type=int, default=7, help="Tile zoom (default 7, matches RAINVIEWER_ZOOM)")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/issue180"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detector threshold:  d <= {MAX_COLOUR_DISTANCE}")
    print(f"Renderer threshold:  d <= {_FILTER_MAX_COLOUR_DISTANCE}")
    print(f"Disagreement band:   ({_FILTER_MAX_COLOUR_DISTANCE}, {MAX_COLOUR_DISTANCE}]")
    print()

    print(f"Fetching manifest from {MANIFEST_URL} ...")
    frames = get_recent_frame_paths(n=2)
    print(f"Two most recent frames:")
    for ts, path in frames:
        print(f"  ts={ts}  path={path}")
    print()

    print(f"Fetching tile (zoom={args.zoom}, tx={args.tx}, ty={args.ty}) for both frames ...")
    tiles = [fetch_tile(p, args.zoom, args.tx, args.ty) for _, p in frames]
    print()

    distances = [distance_to_palette(t) for t in tiles]
    opaques = [opaque_mask(t) for t in tiles]

    for i, (rgba, dist, op) in enumerate(zip(tiles, distances, opaques)):
        total_pixels = rgba.shape[0] * rgba.shape[1]
        n_op = int(op.sum())
        print(f"--- Frame {i} ---")
        print(f"  Total pixels:   {total_pixels}")
        print(f"  Opaque pixels:  {n_op}  ({100 * n_op / total_pixels:.2f}%)")
        if n_op == 0:
            print(f"  (No opaque pixels in this tile - try a different tx/ty)")
            continue
        hist = histogram(dist, op)
        for band, count in hist.items():
            pct = 100 * count / n_op
            bar = "#" * int(pct / 2)
            print(f"  {band}: {count:>6}  ({pct:5.2f}%)  {bar}")

        bands = band_array(dist, op)
        n_both_keep = int((bands == "both_keep").sum())
        n_renderer_drops = int((bands == "renderer_drops").sum())
        n_both_drop = int((bands == "both_drop").sum())
        print(f"  both_keep         (d <= 30):  {n_both_keep:>6}  ({100*n_both_keep/n_op:5.2f}% of opaque)")
        print(f"  renderer_drops    (30 < d <= 60):  {n_renderer_drops:>6}  ({100*n_renderer_drops/n_op:5.2f}% of opaque)")
        print(f"  both_drop          (d > 60):  {n_both_drop:>6}  ({100*n_both_drop/n_op:5.2f}% of opaque)")
        print()

        Image.fromarray(rgba, mode="RGBA").save(args.out_dir / f"frame_{i}.png")
        overlay_band(rgba, bands == "renderer_drops", (255, 0, 0)).save(args.out_dir / f"frame_{i}_renderer_drops.png")
        overlay_band(rgba, bands == "both_drop", (255, 0, 255)).save(args.out_dir / f"frame_{i}_both_drop.png")

    if len(tiles) == 2 and opaques[0].any() and opaques[1].any():
        print("--- Cross-frame stability ---")
        bands0 = band_array(distances[0], opaques[0])
        bands1 = band_array(distances[1], opaques[1])

        renderer_drops_0 = bands0 == "renderer_drops"
        n0 = int(renderer_drops_0.sum())
        if n0 > 0:
            stayed = int(((bands1 == "renderer_drops") & renderer_drops_0).sum())
            moved_to_keep = int(((bands1 == "both_keep") & renderer_drops_0).sum())
            moved_to_drop = int(((bands1 == "both_drop") & renderer_drops_0).sum())
            moved_to_transparent = int(((bands1 == "transparent") & renderer_drops_0).sum())
            print(f"  Of {n0} pixels in (30, 60] band on frame 0:")
            print(f"    {stayed:>6} stayed in (30, 60]    ({100*stayed/n0:5.2f}%)")
            print(f"    {moved_to_keep:>6} moved to (0, 30]      ({100*moved_to_keep/n0:5.2f}%)")
            print(f"    {moved_to_drop:>6} moved to (60, inf)    ({100*moved_to_drop/n0:5.2f}%)")
            print(f"    {moved_to_transparent:>6} became transparent    ({100*moved_to_transparent/n0:5.2f}%)")

        keep_0 = bands0 == "both_keep"
        n_keep = int(keep_0.sum())
        if n_keep > 0:
            became_renderer_drops = int(((bands1 == "renderer_drops") & keep_0).sum())
            became_both_drop = int(((bands1 == "both_drop") & keep_0).sum())
            became_transparent = int(((bands1 == "transparent") & keep_0).sum())
            print(f"  Of {n_keep} pixels rendered (d <= 30) on frame 0:")
            print(f"    {became_renderer_drops:>6} fell into (30, 60] on frame 1  ({100*became_renderer_drops/n_keep:5.2f}%)  <-- visible flicker if non-trivial")
            print(f"    {became_both_drop:>6} fell into (60, inf) on frame 1  ({100*became_both_drop/n_keep:5.2f}%)")
            print(f"    {became_transparent:>6} became transparent on frame 1  ({100*became_transparent/n_keep:5.2f}%)")
        print()

    print(f"Diagnostic images written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
