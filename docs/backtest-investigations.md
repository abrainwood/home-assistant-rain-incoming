# Backtest Investigations

> **Note (2026-05-23)**: this doc is now a **historical investigation log**, not the active plan. The canonical phase plan for ongoing experimentation lives in [rain-incoming-backtester `docs/experiment-motion-estimation-comparison.md` § "Phase plan"](https://github.com/abrainwood/rain-incoming-backtester/blob/main/docs/experiment-motion-estimation-comparison.md). The Tier 1-4 / Priority 1-7 items below have been superseded or folded into the phase structure:
> - **Tier 1 (improve detection)** → addressed by Phase -2.5 (noise taxonomy) and Phase -2.7 (Mt Read orographic — issue #201).
> - **Tier 2 (arrival time)** → folded into Phase 4-13 winner selection criteria (motion estimator bake-off).
> - **Tier 3 (QC tuning)** → Phase -2.5 scope.
> - **Tier 4 (lower priority)** → tested-and-abandoned items.
> - Verifier methodology: all subsequent phases run with `--obs-window-forward-min 30` (post-#200 default; see `docs/issue-200-window-sweep-results.md`).
>
> Use this doc for historical context on what's been tried and rejected. Use the canonical plan for what to work on next.

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

Prioritized for Penrith improvement (worst performer: POD 0.502, bias 0.68, under-detecting).
Each item is a hypothesis to test via quick subset run then validate with full run.

### Tier 1: Improve detection (Penrith's core problem - missing 50% of rain)

- **T1: Intensity threshold** - DEAD END. The RainViewer colour palette is quantized: minimum non-zero intensity is 0.10 (the lightest colour). Our threshold is already at 0.10 - lowering it has zero effect because no pixels exist between 0.0 and 0.10. Raising it would worsen under-detection at Penrith/Mobile. The threshold can't be tuned given the input data.
- **V2: Verification window sensitivity** - DONE. Tested 20/40/60min windows for Penrith. CSI peaks at 20min (0.534, POD 0.646) and degrades to 0.425 at 60min (POD 0.502). The detector is decent at near-term (65% of rain within 20min) but misses most long-lead-time events. Each extra 20min of verification adds ~80 rain events that the detector mostly misses. The 60min window is right for the product (users want 60min warning), but the detection gap is 20-60min lead time rain.
- **T2: Min cell area** - DEAD END. At 512x512 grid resolution (~1 km/px), all precipitation cells are >= 4 pixels. Zero cells below 4px in 20 sampled rain frames. The filter never triggers - lowering it changes nothing.

### Tier 2: Improve arrival time (Penrith mean error -5.6min)

- **A2: Acceleration model** - TESTED, worsened results. Penrith CSI 0.425->0.411, +22 FAs for +2 hits, lead time error unchanged (-5.6 -> -5.7min). With only 2-7 velocity samples per track, acceleration estimates are too noisy and project cells incorrectly into the proximity radius. Would need much longer tracks to be useful.
- **C3: Velocity estimation window** - ALREADY IMPLEMENTED. The detector already averages velocity over all frame pairs in the track (detector.py lines 287-302). With 8-frame windows, this means up to 7 velocity estimates are averaged. No additional smoothing needed.
- **A1: Closing distance fallback** - currently enabled, helps detection. Test if it generates FAs in terrain where cells change direction near mountains.

### Tier 3: QC tuning (only after detection is better)

- **Q3: Texture vs temporal weights** - convective rain is inherently speckly. Current texture weights may over-penalize real rain. Test reducing texture weight.
- **Q1: Clutter maturity period** - currently 2 weeks (2016 cycles). In backtesting the clutter map starts cold and never matures. Test with a pre-warmed clutter map.
- **Q2: QC confidence threshold** - downstream of Q1. Test raising/lowering the confidence floor after clutter map is warm.

### Tier 4: Low priority / wrong direction for Penrith

- **T4: Min temporal frames** - currently 2 frames. Already minimal. Going to 1 removes tracking entirely. Going to 3+ worsens under-detection.
- **C1: Max storm speed** - currently 120 km/h. Rarely a factor.
- **C2: Max angular variance** - currently 0.5 radians. Tightening would hurt Penrith where cells change direction near mountains.
- **T3: Proximity radius** - currently 15km, aligned with verifier in V1. Further changes need paired analysis.

### Completed

- **V1: Align detection and verification proximity** - DONE. See V1 results below.

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

## Phase 6A: Intensity Trend Filter (NEUTRAL across all 8 locations)

Hypothesis: 73% of Penrith FAs are real cells that dissipate crossing the Blue
Mountains. If we suppress detection of cells with sharply declining intensity
(final/initial < 0.5), we should reduce these FAs.

**Result: no measurable effect at threshold 0.5 across all 8 locations.**

| Location | Δ POD | Δ FAR | Δ CSI | Δ Hits | Δ FAs |
|----------|-------|-------|-------|--------|-------|
| cairns_babinda | -0.001 | 0.000 | -0.001 | -1 | 0 |
| darwin | -0.012 | -0.001 | -0.008 | -3 | -1 |
| hilo | 0.000 | 0.000 | 0.000 | 0 | 0 |
| ketchikan | 0.000 | 0.000 | 0.000 | 0 | 0 |
| lake_margaret | 0.000 | 0.000 | 0.000 | 0 | 0 |
| mobile | 0.000 | -0.003 | +0.002 | 0 | -1 |
| penrith | 0.000 | -0.003 | +0.001 | 0 | -1 |
| quillayute | 0.000 | -0.003 | +0.002 | 0 | -2 |

Total across all locations: 4 fewer hits, 5 fewer FAs. Max CSI delta ±0.008
(Darwin loses 3 hits for 1 FA reduction - net negative).

**Status: REMOVED**. The structural mismatch (~82% of Penrith FAs are overhead
transients, not approaching cells that intensity trend can address) means more
data won't change the conclusion at this threshold. Future investigation could
sweep the threshold (0.6-0.8) but not at the cost of carrying dead config in
the meantime.

Why no effect:
- Of 55 Penrith FAs, only ~10 are truly approaching cells (positive predicted
  arrival). The other 45 are overhead-transient FAs that intensity trend
  doesn't address.
- The 0.5 threshold (intensity must halve across the track) may be too strict
  for the actual dissipation pattern. Cells in the radar tile may break apart
  rather than fade in peak intensity.
- The original 73% number may have come from a different baseline configuration.

**Status**: kept as opt-in (`use_intensity_trend=False` default,
`--use-intensity-trend` to enable in backtest). Tunable threshold could be
explored. Currently does not provide a path to reducing Penrith FAs.

**Next direction**: overhead transient FAs need a different approach -
satellite cloud check, forecast PoP, or smarter clutter detection.

## Phase 3B: Frame-Scaled min_temporal_frames (NEUTRAL across all 8 locations)

Hypothesis: longer-range predictions need more evidence. Scale `min_temporal_frames`
by lookahead horizon: 2 frames at <=20min, 3 at <=40min, 4 at >40min.

**Result across all 8 locations at 30min lookahead** (effective min becomes 3):

| Location | Δ POD | Δ FAR | Δ CSI |
|----------|-------|-------|-------|
| cairns_babinda | 0.000 | -0.001 | +0.001 |
| darwin | -0.004 | +0.001 | -0.003 |
| hilo | 0.000 | -0.005 | +0.003 |
| ketchikan | 0.000 | 0.000 | 0.000 |
| lake_margaret | -0.004 | -0.007 | +0.003 |
| mobile | 0.000 | 0.000 | 0.000 |
| penrith | -0.007 | -0.005 | -0.003 |
| quillayute | -0.003 | -0.001 | -0.002 |

Max CSI delta ±0.003. Slight wins at Hilo/Lake Margaret, slight losses at
Penrith/Darwin/Quillayute. Essentially noise.

**Status: REMOVED**. The tradeoff (filter short tracks, lose some real hits with
the noise) is structural; more data won't shift the direction. Could matter more
at 60min lookahead where scaling jumps to 4 frames, but default is 30min.

**Status**: kept as opt-in (`--frame-scale-by-lookahead`). Could matter more at
longer lookaheads (60min would require 4 frames) but current default is 30min
where the impact is tiny.

## Perth False Negative 2026-04-27 14:20 AEST (Tier 0 investigation)

**Context**: Live integration showed `Rain Incoming = Dry` at 14:20 AEST despite visible
approaching rain on radar. Window: `1777263600` (last 8 frames, 13:10–14:20 AEST).

**Finding: not a bug, a structural limitation.**

Window 2 (14:20 AEST - the false negative):
- 170 total tracks, 69 accepted (passed min_frames + ends-on-last-frame checks)
- `rain_incoming=False`, `max_approaching_intensity=0.0000`

### Root causes

**1. Persistent static clutter dominates accepted tracks (61 of 69)**

61 of 69 accepted tracks show exactly `intensity=0.376` (a fixed RainViewer palette step)
throughout all 8 frames (80 minutes). Bearings are random across all quadrants:
`N:16, E:24, S:18, W:10`. This is the signature of static ground/sea clutter - cells
that appear at a fixed location and intensity and are never filtered because the clutter
map is cold. None project to arrive at Perth within the lookahead window.

**2. Approaching band in only 1 frame (too_short)**

36 new cells appear in frame 7 only (last frame, 14:20 AEST). All are `too_short`
(single-frame, `MIN_TEMPORAL_FRAMES=2`). These are likely the actual approaching rain
band visible on radar. A single frame cannot be distinguished from noise.

**3. Two span-2 tracks (frames 6-7) are speed-rejected**

Two accepted tracks span frames 6-7 with velocities 156 km/h and 573 km/h. They pass
the structural track filter (2 frames ≥ `MIN_TEMPORAL_FRAMES`) but are correctly
rejected inside `_evaluate_approaching_cell`'s speed cap (`MAX_STORM_SPEED_KMH=120`).
Both have `intensity=0.376` - likely clutter-matching artifacts from unrelated cells
in adjacent frames.

### What would fix it

1. **Warm clutter map**: The 61 static-intensity cells would be filtered once the
   clutter map has learned Perth's ground clutter pattern (needs ~2 weeks of data).
   This is the most impactful fix - it eliminates the noise that's masking the signal.

2. **Single-frame approaching band detection with confidence signals** (Phase 3C):
   When a fresh batch of cells appears at the edge of the analysis window pointing
   toward the location, detect with 1 frame IF satellite says there are clouds there
   or forecast PoP is elevated. Without an external signal, single-frame = noise.

### No algorithm change warranted

The detector correctly rejected all accepted tracks (random bearings, fixed intensity,
speed violations). The false negative is correct given the data and thresholds. Fixing
it requires external confidence signals, not threshold tuning.

## Tier 0 Diagnostic Findings (early)

Inspecting Penrith FA window 1776388200 (predicted +11min, no rain arrived):
- 49 tracks built across 8 frames
- 8 marked "accepted" (passed structural checks)
- BUT 2 of those have impossible velocities (212 km/h, 322 km/h) - way over the
  120 km/h speed cap
- These get rejected later by `_evaluate_approaching_cell`'s speed check, but
  they were noise-matched cells the tracker linked across frames despite being
  physically impossible

**Implication**: noise tracking matches cells across frames creating phantom
high-speed tracks. The downstream speed cap catches them but they consume
detector cycles. Could pre-filter at the matching step or expose the
post-evaluation rejection in Tier 0 to make the noise visible.

## What's Actually Shipped (as of 2026-04-28)

All items below are in production on `main`. The plan doc was stale.

**Detection algorithm:**
- [x] 3A: Default lookahead reduced 60→30min (PR #158)
- [x] 3B: Frame-scaled min_frames - tested, NEUTRAL, kept as opt-in `--frame-scale-by-lookahead`
- [x] 3C: Overhead immediate detection - already implemented via `_check_rain_at_location`. Fires `rain_incoming=True` in a single frame when rain pixels are within the proximity radius. Gated by QC confidence (warm clutter map suppresses false positives). No further work needed.
- [x] 6A: Intensity trend filter - tested, NEUTRAL, REMOVED (opt-in `--use-intensity-trend` kept in backtest only)
- [x] Leading-edge arrival fallback (PR #169) - for large incoherent storm fronts whose centroid zigzags while edge approaches. Fixes Perth false negative 2026-04-27 class of events.

**Confidence signals:**
- [x] 5A: Forecast PoP (Open-Meteo, hourly, no API key). Multiplier: PoP<5% → 3x, PoP<30% → 2x, PoP≥30% → 1.0, missing → 1.0 (fail open).
  - **Open question**: the 3x multiplier at PoP<5% may be too aggressive for live use given hourly forecast lag. A stale "dry" forecast could suppress a real detection. Investigation: add `--pop-multiplier` to backtest CLI and run at 1.0/2.0/3.0 to bound the effect. Likely fix: soften to 1.5x/1.0x, or restrict gating to cells >50km away (hourly forecast has time to update before a distant cell arrives).
- [x] 5B: Satellite IR cloud confidence (PR #165). Himawari/GOES via RealEarth tiles. Clear sky → 3x multiplier. Missing satellite → 1.0 (fail open).

**Tooling:**
- [x] Compare feature (`--compare`)
- [x] Curated subset selection (`--generate-subset`, `--subset`)
- [x] Tile decode optimisation (15ms → 1.2ms)
- [x] Centroid extraction optimisation (12.5s → 25ms)
- [x] Frame caching in replay and verifier
- [x] Tier 0 diagnostic trace (`--inspect LOCATION TIMESTAMP`)
- [x] Tier 1 multi-window inspect (`--inspect-set MANIFEST`)
- [x] `--inspect-session SESSION_DIR` for golden_v2 fixtures (PR #170, GH #164)
- [x] `capture_golden_tiles.py --output-dir` for field captures (PR #170)

## Framework Improvements Remaining

### `--pop-multiplier` flag for backtest
Needed to run PoP sensitivity analysis. Add float flag (default 1.0) to the backtest CLI, thread into `_build_detector_config`. Run at 1.0/2.0/3.0 and compare scorecards to bound the live-cadence risk.

### Dry window skip
Consecutive dry windows with unchanged latest frame produce identical detection results. Skip detection when the new frame is also dry - major speedup for locations with long dry periods.

### Per-category comparison
Compare should show deltas per category (hit rate for hits, FA rate for FAs) not just aggregate POD/FAR. The curated set is balanced so aggregate scores are always ~0.5.

### Ground truth verifier
BOM/METAR observation correlation with cell-path-aware station matching. Separate from radar-only verification.
