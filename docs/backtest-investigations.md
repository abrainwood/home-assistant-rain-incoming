# Backtest Investigations

Baseline run: 2026-04-24, 10 days of data, 8 locations, `--qc none`.

## Baseline Results

Corrected baseline (V1: verifier aligned to 15km, matching detector proximity).
Previous baseline used a ~5km verifier radius that inflated FAR and deflated miss counts.

| Location | POD | FAR | CSI | Bias | Hits | Misses | FA | CN |
|----------|-----|-----|-----|------|------|--------|----|----|
| cairns_babinda | 0.956 | 0.025 | 0.933 | 0.98 | 850 | 39 | 22 | 346 |
| lake_margaret | 0.895 | 0.076 | 0.834 | 0.97 | 256 | 30 | 21 | 950 |
| quillayute | 0.830 | 0.135 | 0.735 | 0.96 | 584 | 120 | 91 | 462 |
| hilo | 0.785 | 0.110 | 0.715 | 0.88 | 186 | 51 | 23 | 997 |
| darwin | 0.772 | 0.297 | 0.582 | 1.10 | 244 | 72 | 103 | 838 |
| mobile | 0.641 | 0.164 | 0.569 | 0.77 | 271 | 152 | 53 | 781 |
| penrith | 0.502 | 0.265 | 0.425 | 0.68 | 208 | 206 | 75 | 768 |
| ketchikan | - | - | 1.000 | 0.00 | 0 | 0 | 0 | 1257 |

## Key Findings

- Tropical/maritime locations (Cairns CSI 0.933, Lake Margaret 0.834) strongly outperform terrain-affected locations (Penrith 0.425, Mobile 0.569)
- Penrith and Mobile are under-detecting (bias 0.68 and 0.77) - missing ~50% and ~36% of rain events
- Darwin is the only remaining over-predictor (bias 1.10, FAR 0.297)
- Lead time errors consistently negative (late predictions, mean -0.6 to -5.6min)
- Penrith has the worst lead time accuracy (mean -5.6min) - terrain slows/kills approaching cells

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

- **V1: Align detection and verification proximity** - DONE. See V1 results below.
- **V2: Extended verification window** - currently 60min. Some "false alarms" may be rain that arrived after 60min. Test 90min and 120min windows to reclassify.

## V1 Results: Verifier Proximity Alignment

**Problem**: The verifier used a fixed 10-pixel neighborhood (`proximity_pixels=10`, half=5) regardless of location. This translated to only ~5km radius at mid-latitudes and ~3.4km at high latitudes. Meanwhile the detector uses `PROXIMITY_RADIUS_KM=15.0` converted to pixels dynamically (~14px at mid-latitudes). Rain arriving within 15km (which the detector correctly flagged) was being classified as "no rain arrived" by the narrow verifier, creating phantom false alarms.

The original V1 hypothesis incorrectly stated "128km vs 30km" - 128km is a radar display radius, and 30km overstated the verifier's actual reach. The real mismatch was 15km (detector) vs 3-6km (verifier), a 2.5-4.4x ratio depending on latitude.

**Fix**: Replaced `VerifierConfig.proximity_pixels` with `proximity_km` (default `PROXIMITY_RADIUS_KM=15.0`). The verifier now computes pixel radius from km using the same formula as the detector, so detection and verification are geometrically consistent.

**Mismatch by location** (old 10px half=5 vs new 15km):

| Location | Latitude | Old radius (km) | New radius (km) | Ratio |
|----------|----------|-----------------|-----------------|-------|
| Ketchikan | 55.4 | 3.4 | 15.0 | 4.4x |
| Quillayute | 47.9 | 4.2 | 15.0 | 3.6x |
| Penrith | -33.8 | 5.0 | 15.0 | 3.0x |
| Hilo | 19.7 | 5.8 | 15.0 | 2.6x |
| Cairns | -17.4 | 5.8 | 15.0 | 2.6x |
| Darwin | -12.5 | 6.0 | 15.0 | 2.5x |

**Quick subset reclassification** (10 curated FAs per location):

| Location | Phantom FAs | Old CNs now Misses | Net |
|----------|-------------|--------------------|----|
| Cairns | 8/10 (80%) | 1/10 | Strong - nearly all FAs were real rain just outside old radius |
| Penrith | 7/10 (70%) | 0/10 | Strong - 70% of Penrith FAs were phantom |
| Hilo | 7/10 (70%) | 0/10 | Strong |
| Darwin | 6/10 (60%) | 0/10 | Good |
| Mobile | 6/10 (60%) | 0/10 | Good |
| Quillayute | 4/10 (40%) | 4/10 | Mixed - wider radius also catches more real rain events at high latitude |
| Lake Margaret | 3/10 (30%) | 0/10 | Modest |

**Full population results** (qc=none, 10 days, all windows):

| Location | POD | FAR | CSI | Bias | FA→Hit | CN→Miss |
|----------|-----|-----|-----|------|--------|---------|
| cairns_babinda | 0.956 | 0.025 | 0.933 | 0.98 | +23 | +15 |
| quillayute | 0.830 | 0.135 | 0.735 | 0.96 | +171 | +93 |
| lake_margaret | 0.895 | 0.076 | 0.834 | 0.97 | +15 | +8 |
| hilo | 0.785 | 0.110 | 0.715 | 0.88 | +32 | +26 |
| darwin | 0.772 | 0.297 | 0.582 | 1.10 | +95 | +37 |
| mobile | 0.641 | 0.164 | 0.569 | 0.77 | +70 | +108 |
| penrith | 0.502 | 0.265 | 0.425 | 0.68 | +73 | +133 |

**CSI improved at every location.** FAR dropped dramatically (Penrith 0.523→0.265, Darwin 0.571→0.297). Bias moved toward 1.0 everywhere - the detector is better calibrated than old scores suggested.

**Critical insight - tuning direction has changed**: The old verifier made it look like the system was over-predicting (bias > 1.0 everywhere). The aligned verifier reveals the opposite at Penrith (bias=0.68) and Mobile (bias=0.77) - these locations are UNDER-detecting, not over-predicting. For Penrith, 133 old CNs became misses (rain within 15km we weren't detecting AND the old verifier didn't see). Tuning should focus on improving detection rate (T1: lower threshold, T4: fewer frames) rather than reducing false alarms (T2: higher min area, T3: tighter proximity).

Full results in reports/full-v1-aligned/.

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
