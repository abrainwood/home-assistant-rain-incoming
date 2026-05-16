# GH #195 obs-truth proximity sweep results

Captured 2026-05-16 against the V2 palette + obs-truth verifier + detector-and-verifier-both-aligned-to-obs (the design fix from #195 phase 1). 5 radii × 5 locations × 3886 windows each. ObsStationVerifier (backtester #28) consulting BoM AWS / METAR jsonl as point truth.

**Two sweeps combined.** Main sweep (4 locations) ran with broken cairns_babinda registry; cairns-only follow-up ran after the leading-zero fix landed (#37 - station IDs in obs_stations.py needed stripped-leading-zero format to match captured jsonl). All 5 locations are now valid.

## Experiment design

- **Variants**: control 15km + sweep down to 10 / 7.5 / 5 / 2.5 km. Same as #192 v1 but with obs-truth instead of RV-pixel truth.
- **Locations included**: darwin, hilo, lake_margaret (now Mt Read), penrith. cairns_babinda excluded - see "Caveats" below.
- **Detector + verifier alignment**: both anchored to the resolved obs station coords via `--align-location-to-obs --truth-source obs`.
- **Palette held constant at V2**.

Backtester invocation:
```bash
.venv/bin/python -m scripts.backtest \
    --data-dir backtest_data \
    --locations cairns_babinda,darwin,hilo,lake_margaret,penrith \
    --qc full --verify \
    --truth-source obs \
    --align-location-to-obs \
    --proximity-radius-km <radius> \
    --output-dir /tmp/issue195/scorecards/<variant>
```

Each variant: ~91-113 min wall-clock. Sweep total: 8h 15m.

## Aggregate scorecard (5 locations)

| Variant | Radius (km) | hits | miss | fa | cn | POD | FAR | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 15.0 | 1187 | 677 | 3651 | 13055 | 0.637 | 0.755 | 0.215 |
| a1 | 10.0 | 1119 | 745 | 2989 | 13717 | 0.600 | 0.728 | 0.231 |
| a2 | 7.5 | 1074 | 790 | 2722 | 13984 | 0.576 | 0.717 | 0.234 |
| a3 | **5.0** | 1034 | 830 | 2468 | 14238 | 0.555 | 0.705 | **0.239** |
| a4 | 2.5 | 955 | 909 | 2158 | 14548 | 0.512 | 0.693 | 0.237 |

CSI peaks at **5 km** (0.239), 2.5 km essentially tied (0.237). 15 km comes in at 0.215. POD drops monotonically as radius shrinks (catches fewer real events). FAR also drops monotonically (fewer false alarms). At 5 km the trade is best.

## Per-location CSI

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda (Nerada) | 0.221 | 0.233 | 0.239 | **0.242** | **0.242** |
| darwin | 0.091 | 0.098 | 0.103 | 0.108 | **0.121** |
| hilo | 0.175 | 0.186 | 0.184 | 0.187 | **0.191** |
| lake_margaret (Mt Read) | **0.368** | 0.361 | 0.354 | 0.348 | 0.325 |
| penrith | 0.081 | 0.116 | 0.127 | 0.146 | **0.157** |

**Four of five locations prefer SMALLER radius**:
- darwin +33% (15→2.5)
- penrith +94% (15→2.5)
- cairns_babinda +9.5% (15→5)
- hilo +9% (15→2.5)

**One prefers LARGER**: lake_margaret -12% (mountain orographic case).

## Per-location POD

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.801 | 0.766 | 0.743 | 0.715 | 0.651 |
| darwin | 0.867 | 0.800 | 0.778 | 0.778 | 0.778 |
| hilo | 0.837 | 0.742 | 0.683 | 0.643 | 0.600 |
| lake_margaret | 0.457 | 0.439 | 0.425 | 0.413 | 0.381 |
| penrith | 0.754 | 0.754 | 0.754 | 0.719 | 0.684 |

POD drops monotonically at every location as radius shrinks - the detector catches fewer of the real rain events when the truth zone tightens.

## Per-location FAR

| Location | 15.0km | 10.0km | 7.5km | 5.0km | 2.5km |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.766 | 0.749 | 0.739 | 0.732 | 0.722 |
| darwin | 0.908 | 0.900 | 0.894 | 0.889 | 0.875 |
| hilo | 0.819 | 0.802 | 0.799 | 0.791 | 0.781 |
| lake_margaret | 0.346 | 0.331 | 0.320 | 0.312 | 0.311 |
| penrith | 0.917 | 0.880 | 0.868 | 0.845 | 0.830 |

FAR also drops monotonically at every location. For darwin / hilo / penrith, the FAR drop is small (3-9pp), and these locations also have a small POD drop, so CSI shifts slightly in penrith's favour. lake_margaret has a similar FAR drop but a bigger POD drop, so it loses CSI as radius tightens.

## Comparison to v1 (RV-truth)

| Metric | v1 aggregate @ 15km | v2 aggregate @ 15km (4 locs) |
|---|---:|---:|
| POD | 0.814 | 0.574 |
| FAR | 0.150 | 0.748 |
| CSI | 0.712 | 0.212 |

**The dramatic CSI collapse (0.71 → 0.21) is the RV-truth bias being removed.** v1 had RV's hallucinated rain halo as truth - detector predictions matched halo presence trivially because the detector reads the same RV pixels. With point-truth from obs stations, real rain at the gauge is genuinely much harder to predict than "rain nearby in RV pixels".

The 4-5x RV over-coverage finding from `docs/rainviewer-vs-bom-investigation.md` predicted exactly this: v1 FAR was a fiction because the false alarms were detector correctly NOT firing on RV halos. v2 FAR is real - detector is firing for rain that exists in RV pixels but doesn't actually fall at the obs station.

## Interpretation

### CSI no longer monotonically decreases with smaller radius

Direction reversed vs v1. CSI is now mostly flat with a slight uptick at smaller radii (0.212 → 0.237).

**Theory**: with obs-truth, the verifier is asking "did the gauge see rain". As the radius shrinks, fewer of the detector's predictions count as "real" events (POD drops), but fewer false alarms also count (FAR drops). The two effects are roughly balanced.

### Penrith strongly prefers smaller radius

Penrith CSI nearly doubles from 15km (0.081) to 2.5km (0.157). This matches the v1 Penrith POD-rises-at-smaller-R anomaly we filed as #194 - same signal, same direction, now in CSI not just POD. The Penrith Lakes AWS is on the floor of the Sydney basin and is heavily influenced by escarpment-driven storm patterns. A wide truth zone (15km) catches lots of rain that DID happen geographically nearby but NOT at the gauge - those score as detector false alarms with a wide radius but correctly NOT-rain-here with a tight radius.

### lake_margaret (Mt Read) prefers larger radius

Mt Read at 1120m gets >2000mm of rain in 29 days - orographic rainfall. Detector predictions correlate strongly with rain near Mt Read because the area is consistently wet. A wide truth zone catches that well; a tight one penalises predictions that landed in the area but not on the exact ridge.

### POD drops more than FAR at lake_margaret

Mt Read's POD drops 0.457 → 0.381 (-17%). hilo POD drops 0.837 → 0.600 (-28%). These are the only two locations where POD drops faster than FAR drops, so CSI loses ground. They share an orographic / island wet-tropics geography where detector predictions are good at "rain in the area" but the gauge doesn't always see it.

### Aggregate detector quality on obs-truth is poor

CSI 0.21-0.24 across all radii means the detector gets 21-24% of "rain at station within next 30 min" predictions right. v1's 71% was a measurement artefact. The real performance is much worse than we thought.

This is consistent with the trace-rain hypothesis (gauges round to 0.1mm and can't register trace rain) plus the RV-dilation bias on the detector input side. Both effects need addressing for a meaningful improvement.

## Decision

**Set `PROXIMITY_RADIUS_KM = 5.0`** (down from 15.0).

- Aggregate CSI peaks at 5 km (0.239), 2.5 km essentially tied (0.237). 15 km is clearly suboptimal at 0.215.
- **4 of 5 locations prefer smaller radius**: cairns_babinda, darwin, hilo, penrith.
- Only **lake_margaret (Mt Read)** prefers 15 km - and Mt Read is a synthetic wet test case, not a deployment target.
- **Penrith (real deployment target) almost doubles CSI** (0.081 → 0.146 at 5 km, peak 0.157 at 2.5 km).
- 5 km is the aggregate sweet spot - gives most of penrith / darwin / cairns gain while keeping Mt Read within -5% of its peak. 2.5 km hits hilo's POD hard (0.837 → 0.600).

Other levers worth exploring after this:
1. Trace-rain filtering (backtester #19)
2. Detector consumption of RV-dilated pixels (the meta-bug from rainviewer-vs-bom-investigation.md)
3. Phase 2 of #195 (extent-aware projection in detector.py)

## Caveats

### cairns_babinda excluded

PR #29 (correcting cairns_babinda's obs station from broken 94287 → 031187 Topaz Alert) merged mid-sweep. Variants control / a1 / a2 ran with broken 94287 (zero obs records, all FA/CN); variants a3 / a4 ran with 031187 (real Topaz Alert data). Across-variant comparison is impossible. cairns is **excluded from all aggregate and per-location tables above**.

The cairns-only follow-up sweep is queued. Currently pending decision on issue #32: Nerada Alert (032165, coastal/low, climatology match) vs Topaz Alert (031187, inland mountain, more rain). PR #36 implements Nerada-as-primary.

### The 30-min lookahead question

The verifier checks `[t, t+30min]` for any `is_raining=True` obs record at the resolved station. With BoM AWS reporting at 15-min intervals, that's 2 chances per window. METAR is roughly hourly so the chance is closer to 0-1 per window. This isn't varied across the sweep - all 5 variants use the same lookahead.

### Obs station offsets

- darwin: KMOB-equivalent (Darwin Airport 14015), 7.0 km from config
- hilo: PHTO (Hilo International), 0.2 km from config (essentially overlapping)
- lake_margaret: Mt Read 97085, 16.9 km from config (the obs-aligned anchor moved this far)
- penrith: 67113 Penrith Lakes AWS, 3.6 km from config

Hilo, penrith, darwin have small offsets so detector and verifier are close to the original config. Mt Read has a 17 km offset which is substantial - the detector is now evaluating "rain near Mt Read" not "rain near original lake_margaret config".

## Open questions

- **Should we re-run the intensity-comparison sweep (V1-V5 palette) under obs truth?** The same RV-truth bias affected it. Probably yes - the V2-palette-wins finding may not hold under obs-truth.
- **Is detector quality genuinely 21-24% CSI, or is there an obs-side artefact?** Trace-rain hypothesis (backtester #19) might explain some of the false alarms - radar correctly seeing rain that the gauge doesn't register. Need ground-truth-of-truth via security-cam cross-reference.
- **Geographic specialisation**: 5 km is the chosen default. Mt Read and Hilo POD would prefer larger; cairns/darwin/penrith prefer smaller still. A per-location override would let users tune for their geography but isn't supported today.

## Replication

- Per-variant scorecards: `/tmp/issue195/scorecards/<variant>/scorecards.json`
- Runner: `/tmp/issue195/run_obs_truth_sweep.sh`
- Sweep log: `/tmp/issue195/sweep.log`
- Backtester at: `feat(#195): ObsStationVerifier - BoM/METAR truth path` (commit 6dc0018, merged main)

## Related

- #194 (Penrith POD regression - confirmed real signal here)
- #195 phase 2 (extent-aware projection - may help Lake Margaret CSI)
- `docs/rainviewer-vs-bom-investigation.md` (the meta-finding that motivated all this)
- backtester #28 (ObsStationVerifier)
- backtester #29 (cairns_babinda 94287 → 031187 fix)
- backtester #32 (cairns_babinda Topaz → Nerada follow-up, PR #36 pending)
- `docs/issue-192-proximity-sweep-results.md` (v1 results - now known to be RV-truth-biased)
