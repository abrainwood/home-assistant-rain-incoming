# Perth false-negative — 2026-04-27

## What happened

User-reported false negative: rain visibly approaching Perth from the
NE/NW on the radar (rainviewer source) but the integration's sensor
showed `Rain Incoming = Dry`, `Arrival Time = Unknown`, `Imminent =
Off`.

Reported around AEST 14:20 (= AWST 12:20 = UTC 04:20). At that point
"Last Rain" was 50 minutes prior, so rain HAD been at the location
around AEST 13:30 (~UTC 03:30).

## Location

- **Name**: Perth, WA, Australia
- **Lat/Lon**: -31.95, 115.86
- **Timezone**: AWST (UTC+8). Note: rainviewer.com header may show AEST regardless.

## Captured data

13 frames at 10-min cadence. Timestamps in UTC seconds-epoch:

| Epoch | UTC | AWST | AEST |
|-------|-----|------|------|
| 1777257000 | 02:30 | 10:30 | 12:30 |
| 1777257600 | 02:40 | 10:40 | 12:40 |
| 1777258200 | 02:50 | 10:50 | 12:50 |
| 1777258800 | 03:00 | 11:00 | 13:00 |
| 1777259400 | 03:10 | 11:10 | **13:10** ← user observed rain incoming on radar |
| 1777260000 | 03:20 | 11:20 | 13:20 |
| 1777260600 | 03:30 | 11:30 | 13:30 ← "rain at location" per Last Rain stamp |
| 1777261200 | 03:40 | 11:40 | 13:40 |
| 1777261800 | 03:50 | 11:50 | 13:50 |
| 1777262400 | 04:00 | 12:00 | 14:00 |
| 1777263000 | 04:10 | 12:10 | 14:20 (close) |
| 1777263600 | 04:20 | 12:20 | **14:20** ← user observed sensor says Dry |
| 1777264200 | 04:30 | 12:30 | 14:30 |

Both colour schemes (s2 render, s6 detection) captured per timestamp.

## How to investigate

**Format mismatch caveat**: this capture is in golden_v2 format (`bronze/manifest.json` + `bronze/tiles/<frame_ts>/*.png`). The Tier 0 `--inspect` CLI reads backtest format (`backtest_data/captures/<loc>/<date>/<HHMM>_meta.json`). The raw tiles ARE preserved here; an adapter is needed before `--inspect` will work.

Two paths when picking this up:

**Option A — write a small adapter** (recommended). Convert this fixture's `bronze/manifest.json` + tile PNGs into per-frame `_meta.json` files matching the data_loader format (see `scripts/backtest/data_loader.py:_parse_capture` — needs `capture_utc`, `frame_ts`, `tiles[{x, y, file}]`, `location`, `zoom`). Drop the converted files into `backtest_data/captures/perth/<date>/` and run:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.backtest.cli \
  --data-dir backtest_data \
  --inspect perth 1777263600
```

**Option B — one-off Python** that loads `bronze/manifest.json` + tile bytes directly, builds `RainViewerFrame`s, calls `detect()` with a `DiagnosticTrace`, and prints the JSON. Bypasses the format mismatch entirely. ~30 lines.

Window-end timestamps of interest:
- `1777259400` = AEST 13:10 — user saw "rain incoming" on radar
- `1777263600` = AEST 14:20 — user saw sensor "Dry"

## Hypotheses worth checking against the trace

1. **Cell area split** — large rain field fragmented into many
   sub-MIN_CELL_AREA_PIXELS=4 cells, each filtered out.
2. **Velocity coherence** — multi-cell field with non-uniform motion
   failing the angular variance check.
3. **Direction** — cells actually moving away (NW→N or N→S),
   counterintuitive given typical Perth flow.
4. **Sparse track / min_temporal_frames** — cells not consistently
   labelled across enough frames.
5. **Lookahead** — cells too far to arrive within DEFAULT_LOOKAHEAD_MINUTES (30min) at typical storm speed.
6. **QC clutter rejection** — coastal noise mistakenly classified as
   clutter.

## User-supplied screenshots

- `perth_radar_animation.gif` — radar animation around 13:10 AEST (copied from Downloads)
- Sensor state at 14:20 AEST: at `~/Downloads/HA Perth Screenshot 2026-04-27 at 2.30.59 pm.png`. Sandbox blocked direct copy at capture time; drag into this directory via Finder if the investigation needs it.
