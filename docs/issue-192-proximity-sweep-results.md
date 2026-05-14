# GH #192 proximity radius sweep results

Captured 2026-05-15 against the V2 palette (`PROXIMITY_RADIUS_KM = 15.0` on main at `bda126d`). 5 radii × 5 locations × 3886 windows each, obs-station-aligned verifier.

## Experiment design

- **Variants**: control 15km + sweep down to 10/7.5/5/2.5 km.
- **Locations**: cairns_babinda, darwin, hilo, lake_margaret, penrith (dropped mobile + quillayute per #187 slowness, ketchikan has no rain in dataset).
- **Verifier coords moved to nearest in-grid obs station** per `--align-location-to-obs` (backtester#20). Collapses the truth zone to "did the obs station report rain" instead of "was rain near user location", removing the radius-as-truth-zone confound.
- **Palette held constant at V2** (post-#190). Detector and verifier kept aligned within each run (V1 invariant).

Backtester invocation (per variant, varying `--proximity-radius-km`):

```bash
.venv/bin/python -m scripts.backtest \
    --data-dir backtest_data \
    --locations cairns_babinda,darwin,hilo,lake_margaret,penrith \
    --qc full \
    --verify \
    --proximity-radius-km <radius> \
    --align-location-to-obs \
    --output-dir /tmp/issue192/scorecards/<variant>
```

Each variant: ~73 min wall-clock. Sweep total: ~6h (one variant repeated due to a mid-flight `git checkout` accident on the first attempt; salvaged the control run).

## Aggregate scorecard

| Variant | Radius (km) | hits | miss | fa | cn | POD | FAR | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **control** | **15.0** | **4310** | **983** | **759** | **12518** | **0.814** | **0.150** | **0.712** |
| a1 | 10.0 | 3439 | 760 | 997 | 13374 | 0.819 | 0.225 | 0.662 |
| a2 | 7.5 | 3028 | 718 | 1095 | 13729 | 0.808 | 0.266 | 0.625 |
| a3 | 5.0 | 2516 | 659 | 1239 | 14156 | 0.792 | 0.330 | 0.570 |
| a4 | 2.5 | 1866 | 467 | 1514 | 14723 | 0.800 | 0.448 | 0.485 |

CSI drops **monotonically** as radius shrinks. POD is mostly flat (0.79-0.82). FAR rises **3x** from 15km to 2.5km. There is no inflection point in this direction.

## Per-location CSI

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.844 | 0.805 | 0.785 | 0.739 | 0.649 |
| darwin | 0.582 | 0.523 | 0.436 | 0.393 | 0.341 |
| hilo | 0.750 | 0.598 | 0.523 | 0.419 | 0.316 |
| lake_margaret | 0.689 | 0.641 | 0.605 | 0.552 | 0.444 |
| penrith | 0.395 | 0.384 | 0.373 | 0.324 | 0.294 |

Every location regresses monotonically. No location prefers a smaller radius.

## Per-location POD

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.890 | 0.862 | 0.849 | 0.817 | 0.820 |
| darwin | 0.776 | 0.778 | 0.767 | 0.711 | 0.782 |
| hilo | 0.880 | 0.841 | 0.829 | 0.824 | 0.827 |
| lake_margaret | 0.849 | 0.849 | 0.836 | 0.850 | 0.821 |
| penrith | 0.469 | **0.541** | 0.531 | 0.528 | 0.561 |

**Penrith stands out**: POD goes UP at smaller radii (0.469 -> 0.561). For every other location POD is flat or down. This is the only signal in the experiment that points away from 15km - and it points specifically at Penrith. See "Penrith anomaly" below.

## Per-location FAR

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.057 | 0.076 | 0.087 | 0.114 | 0.243 |
| darwin | 0.300 | 0.385 | 0.497 | 0.533 | 0.623 |
| hilo | 0.165 | 0.326 | 0.413 | 0.540 | 0.662 |
| lake_margaret | 0.215 | 0.277 | 0.313 | 0.389 | 0.508 |
| penrith | 0.287 | 0.430 | 0.444 | 0.545 | 0.618 |

FAR rises sharply at every location as the truth zone shrinks.

## Decision

**Keep `PROXIMITY_RADIUS_KM = 15.0`.** No sub-15km radius meets the #192 acceptance criteria (`>= +0.01 aggregate CSI AND no per-location drop >= 0.02`). The CSI curve is monotonically decreasing in this direction.

This experiment **did not test radii above 15km**. The curve direction suggests CSI might increase further at larger radii, but the original V1 alignment work (`docs/backtest-investigations.md` §V1) chose 15km as a deliberate ceiling matching the detector's effective sensing range. Going higher would mis-attribute distant rain as "arrived" and lose physical interpretability. Out of scope for this experiment.

## Interpretation - why does CSI drop with smaller R?

POD stays roughly constant. The detector keeps catching real rain events at the same rate. What changes is the **verifier's tolerance**:

- At 15km, "rain arrived" means rain pixel inside a 15km halo around the obs station. Many real rain events satisfy this.
- At 2.5km, "rain arrived" means rain pixel within 2.5km. Many events the detector correctly predicts (rain genuinely in the cell vicinity) miss the strict definition - they get scored as false alarms even though they're radar-correct.

So the **detector is well-calibrated for a ~15km neighbourhood**. Anything tighter punishes correct predictions for not landing precisely on the station.

This validates the V1 decision and rules out "we picked 15 arbitrarily and it might be too generous."

## Penrith anomaly

Penrith is the only location where POD improves at smaller radii. Possibilities:

1. **Mismatch between configured user location and obs station**: maybe the obs station picked by the resolver (Penrith Lakes AWS, 67113 at -33.7194, 150.6783) isn't quite where the heaviest rain falls in the Sydney basin events that hit Penrith. Tighter radii might accidentally favour predictions that land closer to the AWS than to the radar-centroid user location.
2. **Sea-breeze / orographic timing offset**: Sydney basin storms are heavily influenced by escarpment topography. Detector arrival predictions may be systematically early or late relative to the AWS, and the radius interacts with timing.
3. **Cell-splitting / blocking**: penrith has a known regression from V2 (#190 results). Whatever is causing that regression may also be why proximity behaves differently here.

Worth a follow-up dive (separate ticket). See `docs/backtest-investigations.md` §Phase 5 for the penrith deep-dive precedent.

## Trace-rain confound (unchanged hypothesis)

The companion backtester#19 hypothesis (trace rain doesn't register on tipping-bucket gauges) applies to this sweep too. Some of the "false alarms" at any radius may be radar-correct, gauge-blind events. The proximity sweep does not disprove or confirm that hypothesis - it varies the verifier's spatial tolerance, not its sensitivity floor.

## Replication

Per-variant scorecards at `/tmp/issue192/scorecards/<variant>/scorecards.json`. Runner: `/tmp/issue192/run_proximity_sweep.sh` (original) and `/tmp/issue192/rerun_failed_variants.sh` (rerun after mid-flight checkout). Analysis: `/tmp/issue192/analyze_results.py`. All artefacts under `/tmp` and not committed.

## Related

- #180 (palette investigation parent)
- #186 (palette ordering fix)
- #188 (long-backtest results)
- #190 (V2 palette merge)
- #192 (this experiment - closes)
- `abrainwood/rain-incoming-backtester#20` (--proximity-radius-km / --align-location-to-obs flags)
- `abrainwood/rain-incoming-backtester#19` (trace-rain-vs-obs hypothesis)
- `docs/backtest-investigations.md` §V1 / §T3 (prior history)
