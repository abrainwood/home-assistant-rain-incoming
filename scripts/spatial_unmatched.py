"""Diagnostic for GH issue #180 (revised v2): paint each undocumented colour
tier spatially over the original tile, so we can see whether each tier hugs
precipitation cells (trace precip) or covers land far from cells (land mask).

NOT production code. One-shot diagnostic.

Output:
    /tmp/issue180/spatial_overlay.png  - 4 panels:
        [original | tier_dark khaki | tier_mid khaki | tier_light khaki]

Run:
    python scripts/spatial_unmatched.py
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

MANIFEST_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_BASE_URL = "https://tilecache.rainviewer.com"
COLOUR_SCHEME = 2
TILE_SIZE = 256

# Three khaki tiers identified by inspect_unmatched_colours.py
TIER_DARK = (170, 158, 121)
TIER_MID = (206, 192, 135)
TIER_LIGHT = (218, 204, 147)


def fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=15) as resp:
        return resp.read()


def get_recent_frame_path() -> str:
    raw = fetch_bytes(MANIFEST_URL)
    manifest = json.loads(raw)
    return manifest["radar"]["past"][-1]["path"]


def fetch_tile(frame_path: str, zoom: int, tx: int, ty: int) -> np.ndarray:
    url = f"{TILE_BASE_URL}{frame_path}/{TILE_SIZE}/{zoom}/{tx}/{ty}/{COLOUR_SCHEME}/0.png"
    data = fetch_bytes(url)
    return np.array(Image.open(BytesIO(data)).convert("RGBA"), dtype=np.uint8)


def fetch_quad(frame_path: str, zoom: int, tx: int, ty: int) -> np.ndarray:
    """Stitch a 2x2 block of tiles into a single image."""
    tiles = {}
    for dy in (0, 1):
        for dx in (0, 1):
            tiles[(dx, dy)] = fetch_tile(frame_path, zoom, tx + dx, ty + dy)
    h, w = TILE_SIZE * 2, TILE_SIZE * 2
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for (dx, dy), t in tiles.items():
        out[dy * TILE_SIZE:(dy + 1) * TILE_SIZE, dx * TILE_SIZE:(dx + 1) * TILE_SIZE] = t
    return out


def highlight(rgba: np.ndarray, target_rgb: tuple[int, int, int], tint: tuple[int, int, int]) -> Image.Image:
    """Return a copy of rgba where pixels matching target_rgb are tinted, rest dimmed."""
    out = rgba.copy()
    mask = (
        (rgba[:, :, 0] == target_rgb[0])
        & (rgba[:, :, 1] == target_rgb[1])
        & (rgba[:, :, 2] == target_rgb[2])
        & (rgba[:, :, 3] > 10)
    )
    out[..., 0] = (out[..., 0].astype(np.float32) * 0.3).astype(np.uint8)
    out[..., 1] = (out[..., 1].astype(np.float32) * 0.3).astype(np.uint8)
    out[..., 2] = (out[..., 2].astype(np.float32) * 0.3).astype(np.uint8)
    out[mask, 0] = tint[0]
    out[mask, 1] = tint[1]
    out[mask, 2] = tint[2]
    out[mask, 3] = 255
    count = int(mask.sum())
    return Image.fromarray(out, mode="RGBA"), count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=int, default=117)
    parser.add_argument("--ty", type=int, default=76)
    parser.add_argument("--zoom", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/issue180"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame_path = get_recent_frame_path()
    print(f"Frame path: {frame_path}")
    rgba = fetch_quad(frame_path, args.zoom, args.tx, args.ty)

    Image.fromarray(rgba, mode="RGBA").save(args.out_dir / "spatial_original.png")
    print(f"Original saved to {args.out_dir / 'spatial_original.png'}")

    img_dark, count_dark = highlight(rgba, TIER_DARK, (255, 0, 0))
    img_mid, count_mid = highlight(rgba, TIER_MID, (255, 255, 0))
    img_light, count_light = highlight(rgba, TIER_LIGHT, (0, 255, 0))

    img_dark.save(args.out_dir / "spatial_tier_dark.png")
    img_mid.save(args.out_dir / "spatial_tier_mid.png")
    img_light.save(args.out_dir / "spatial_tier_light.png")

    print(f"Tier highlight images saved:")
    print(f"  tier_dark  ({TIER_DARK})  -> red   ({count_dark} pixels)")
    print(f"  tier_mid   ({TIER_MID})  -> yellow ({count_mid} pixels)")
    print(f"  tier_light ({TIER_LIGHT})  -> green  ({count_light} pixels)")


if __name__ == "__main__":
    main()
