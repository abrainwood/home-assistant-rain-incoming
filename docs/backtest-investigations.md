# Backtest Investigations

Baseline run: 2026-04-24, 10 days of data, 8 locations, `--qc none`.

## Baseline Results

| Location | POD | FAR | CSI | Bias | Hits | Misses | FA | CN |
|----------|-----|-----|-----|------|------|--------|----|----|
| cairns_babinda | 0.972 | 0.052 | 0.923 | 1.02 | 827 | 24 | 45 | 361 |
| lake_margaret | 0.916 | 0.130 | 0.806 | 1.05 | 241 | 22 | 36 | 958 |
| quillayute | 0.939 | 0.388 | 0.588 | 1.53 | 413 | 27 | 262 | 555 |
| hilo | 0.860 | 0.263 | 0.658 | 1.17 | 154 | 25 | 55 | 1023 |
| mobile | 0.820 | 0.380 | 0.546 | 1.32 | 201 | 44 | 123 | 889 |
| darwin | 0.810 | 0.571 | 0.390 | 1.89 | 149 | 35 | 198 | 875 |
| penrith | 0.649 | 0.523 | 0.379 | 1.36 | 135 | 73 | 148 | 901 |
| ketchikan | - | - | 1.000 | 0.00 | 0 | 0 | 0 | 1257 |

## Key Findings

- Detection is good (POD 0.65-0.97) but false alarms are the main problem (FAR 0.05-0.57)
- Over-prediction everywhere (Bias > 1.0 at all rain locations)
- Tropical/maritime locations (Cairns, Hilo) outperform terrain-affected locations (Penrith, Darwin)
- Lead time errors consistently negative (late predictions, mean -4.6 to -12.0min)

## Investigation Queue

### Priority 1: Rerun with --qc full
Baseline was `--qc none`. The QC pipeline (texture, temporal, clutter) exists to filter noise. Many "overhead noise" false alarms (predicted arrival 0/-10min) should be suppressed by QC. Compare against baseline to measure QC impact on FAR.

### Priority 2: Detection vs verification proximity mismatch
The detector uses 128km proximity radius. The verifier uses a 10x10 pixel neighborhood (~30km). A cell 80km away triggers rain_incoming but falls outside the verifier's check, creating phantom false alarms. Need to align these or track them separately.

### Priority 3: False alarm categories
From Darwin's 129 false alarms:
- **Overhead noise (~30-40)**: predicted arrival 0min or -10min. Detector says "rain here now" but verifier disagrees. Likely ground clutter or near-miss cells.
- **Rain that never arrived (~60-70)**: predicted arrival 30-60min. Real approaching cells that dissipated before reaching location. Causes: virga, topographic effects, cell splitting/dying.
- **Near-miss cells (~20-30)**: predicted arrival 10-25min. Cells that passed nearby but not overhead.

### Priority 4: Penrith deep-dive
Worst performer (POD 0.649, FAR 0.523, lead time -12min). Near Blue Mountains - user reports rain approaching from west frequently dissipates before arriving at near-sea-level location. Topographic effects need terrain-aware detection.

### Priority 5: Ketchikan anomaly
Zero rain detected in 10 days at one of the wettest US cities. Possible causes:
- RainViewer coverage gap at 55N latitude
- Tile data present but no precipitation colours (different radar network)
- Genuine dry spell (unlikely for 10 days)

### Priority 6: Miss pattern analysis
Misses cluster in runs of 5-6 consecutive 10-min windows with descending arrival times (50, 40, 30, 20, 10min). Rain was steadily approaching but detector didn't report it. Possible causes:
- Cells entering from outside analysis area
- Below intensity threshold during approach, strengthening on arrival
- Cell tracking losing the cell between frames

### Priority 7: Lead time model
Constant-velocity cell projection assumption fails near terrain. Cells accelerate/decelerate. Penrith lead time error mean=-12min (worst) supports this. Consider:
- Acceleration-aware velocity estimation
- Historical velocity profiles per location
- Terrain-weighted arrival time adjustment

## Framework Improvements Needed

### Compare feature
`--compare <previous-dir>` to show POD/FAR/CSI deltas between runs. Save scorecards as JSON alongside markdown for machine-readable comparison.

### Subset/tagging
`--max-captures N` for quick iteration (~200 captures = ~1.5 days). Full run for significant changes. Tag output dirs meaningfully: `reports/baseline-qc-none`, `reports/qc-full`.

### Verifier performance
Quillayute (100% rain) took ~25min to verify. Verifier needs binary search or index for future capture lookup instead of linear scan.

### Dry window skip
Consecutive dry windows with unchanged latest frame produce identical detection results. Skip detection when the new frame is also dry - major speedup for locations with long dry periods.
