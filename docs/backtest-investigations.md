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

## QC Quick-Run Results (curated subset, 10 per category)

QC reduces false alarms but also kills real detections (cold clutter map).
Full `--qc full` run in progress to get population-level numbers with warm clutter map.

| Location | FA (qc=none) | FA (qc=full) | Hits (qc=none) | Hits (qc=full) |
|----------|--------------|--------------|----------------|----------------|
| darwin | 20 | 5 | 10 | 12 |
| penrith | 20 | 4 | 10 | 6 |
| mobile | 20 | 5 | 10 | 10 |
| cairns | 20 | 7 | 10 | 10 |

QC Bias dropped to 0.5-0.85 (under-forecasting). Need warmer clutter map or adjusted QC thresholds.

## Tuning Backlog

Each item is a hypothesis to test via quick subset run then validate with full run.

### Detection thresholds

- **T1: Intensity threshold** - currently 0.1. Lower catches more rain but increases noise. Test 0.05 and 0.15 to map the sensitivity curve.
- **T2: Min cell area** - currently 5 pixels. Raising to 10-15 would filter small clutter blobs but might miss isolated cells. Test 10 and 20.
- **T3: Proximity radius** - currently 128km. The verifier checks 10x10 pixels (~30km). Tightening detection proximity to 64km would reduce "near miss" false alarms but miss rain approaching from further out. Test 64km and 96km.
- **T4: Min temporal frames** - currently 3 frames. Raising to 4-5 means a cell must be tracked for 30-50 minutes before triggering. Reduces pop-up false alarms but delays detection. Test 4 and 5.

### QC pipeline

- **Q1: Clutter maturity period** - currently 2 weeks (2016 cycles). In backtesting the clutter map starts cold and never matures. Test with a pre-warmed clutter map from a separate warmup run.
- **Q2: QC confidence threshold** - QC multiplies intensity by confidence (0-1). Low-confidence cells get dimmed below the detection threshold. The effective threshold is `intensity * confidence >= 0.1`. Test raising/lowering the confidence floor.
- **Q3: Texture vs temporal weights** - texture analysis catches speckle noise, temporal catches inconsistent cells. Current weights may over-penalize real rain with speckle-like texture (convective rain is inherently noisy). Test reducing texture weight.

### Cell tracking

- **C1: Max storm speed** - currently 150 km/h. Cells moving faster are rejected. Rarely a factor but could filter bogus high-speed matches.
- **C2: Max angular variance** - currently ~90 degrees (in radians). Cells with inconsistent motion direction are rejected. Tightening reduces false matches but misses cells that change direction.
- **C3: Velocity estimation window** - uses consecutive frame pairs. Averaging over 3-4 frame pairs would smooth noisy velocity estimates but add lag.

### Arrival time

- **A1: Closing distance fallback** - when cell velocity points away from location but distance is decreasing, we use a fallback arrival estimate. This catches "approaching but not pointed directly at us" scenarios. May be too generous - test disabling it.
- **A2: Acceleration model** - constant velocity assumption fails near terrain. Test a simple linear acceleration model (velocity change between frame pairs).

### Verification alignment

- **V1: Align detection and verification proximity** - detector uses 128km, verifier uses ~30km. Many "false alarms" may be correct detections of rain that passes nearby but not overhead. Test with verifier proximity matching detector proximity.
- **V2: Extended verification window** - currently 60min. Some "false alarms" may be rain that arrived after 60min. Test 90min and 120min windows to reclassify.

## Framework Improvements Done

- [x] Compare feature (`--compare`)
- [x] Curated subset selection (`--generate-subset`, `--subset`)
- [x] Tile decode optimisation (15ms → 1.2ms)
- [x] Centroid extraction optimisation (12.5s → 25ms)
- [x] Frame caching in replay and verifier

## Framework Improvements Remaining

### Dry window skip
Consecutive dry windows with unchanged latest frame produce identical detection results. Skip detection when the new frame is also dry - major speedup for locations with long dry periods.

### Per-category comparison
Compare should show deltas per category (hit rate for hits, FA rate for FAs) not just aggregate POD/FAR. The curated set is balanced so aggregate scores are always ~0.5.

### Ground truth verifier
BOM/METAR observation correlation with cell-path-aware station matching. Separate from radar-only verification.
