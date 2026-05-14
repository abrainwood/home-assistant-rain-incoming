# GH #180 long backtest results

Captured 2026-05-14 against full dataset (7 locations × 3886 windows × 5 variants).
Mobile excluded due to performance issue (see #187 - the slow-location investigation).

## Variant definitions

All variants share the corrected palette ordering from PR #186 (cyan/blue inversion fixed,
counter-intuitive sub-palette orderings documented). They differ only in which RainViewer
scheme 2 khaki trace tiers are included:

| Variant | Branch | Trace tiers included | Palette entries |
|---|---|---|---:|
| before_fix | `a0b893c` (pre-merge) | none (and broken cyan/blue ordering) | 9 |
| V1 | `experiment/180-postfix-v1` | none (corrected palette) | 9 |
| V2 | `experiment/180-postfix-v2` | `(218,204,147)` innermost only | 10 |
| V3 | `experiment/180-postfix-v3` | `(218,204,147)` + `(206,192,135)` | 11 |
| V4 | `main` (current) | all three trace tiers | 12 |

Trace tier dBZ values (from RainViewer published colour scheme 2 table):

| RGB | Intensity | dBZ | Position in cell |
|---|---:|---:|---|
| `(218,204,147)` | 0.09 | ~13 | innermost - adjacent to blue core |
| `(206,192,135)` | 0.05 | ~10 | middle ring |
| `(170,158,121)` | 0.02 | ~7  | outermost ring |

## Backtester invocation

```bash
.venv/bin/python -m scripts.backtest \
    --data-dir backtest_data \
    --locations cairns_babinda,darwin,hilo,ketchikan,lake_margaret,penrith,quillayute \
    --qc full \
    --verify \
    --output-dir /tmp/issue180/scorecards_full/<variant>
```

Each variant: 7 locations × 3886 windows each (172 gaps skipped per location).
Wall-clock per variant: 3-7 hours depending on quillayute's tracking cost.

## Aggregate scorecard

| Variant | hits | miss | fa | cn | POD | FAR | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| before_fix | 5698 | 893 | 1437 | 17970 | 0.865 | 0.201 | **0.710** |
| V1 (no trace) | 5360 | 1231 | 1047 | 18360 | 0.813 | 0.163 | 0.702 |
| **V2 (+218)** | **5375** | **1216** | **994** | **18413** | **0.816** | **0.156** | **0.709** |
| V3 (+218,+206) | 5353 | 1238 | 1004 | 18403 | 0.812 | 0.158 | 0.705 |
| V4 (+218,+206,+170, current main) | 5353 | 1238 | 1001 | 18406 | 0.812 | 0.158 | 0.705 |

## Delta vs before_fix

| Variant | dPOD | dFAR | dCSI |
|---|---:|---:|---:|
| V1 | -0.051 | -0.038 | -0.008 |
| V2 | -0.049 | **-0.045** | **-0.001** |
| V3 | -0.052 | -0.043 | -0.005 |
| V4 | -0.052 | -0.044 | -0.005 |

**All post-fix variants trade ~5% POD for ~5% FAR.** Net CSI essentially unchanged. The
palette fix is correctness-preserving and CSI-neutral on the full dataset (the apparent
CSI regression seen on the curated subset was a small-sample artefact - 290 windows
vs 27,200).

## Per-location breakdown

### CSI

| Location | before | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.873 | 0.843 | 0.847 | 0.840 | 0.840 |
| darwin | 0.559 | 0.570 | **0.618** | 0.595 | 0.606 |
| hilo | 0.745 | 0.751 | 0.750 | 0.753 | 0.753 |
| ketchikan | n/a (no rain in dataset - 0 hits/miss/fa, 3714 cn) |
| lake_margaret | 0.830 | 0.814 | 0.819 | 0.816 | 0.818 |
| penrith | **0.423** | **0.386** | 0.399 | 0.396 | 0.394 |
| quillayute | 0.608 | 0.611 | 0.615 | 0.612 | 0.610 |

### POD

| Location | before | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.929 | 0.891 | 0.892 | 0.888 | 0.887 |
| darwin | 0.832 | 0.796 | 0.821 | 0.798 | 0.798 |
| hilo | 0.919 | 0.878 | 0.880 | 0.882 | 0.884 |
| lake_margaret | 0.907 | 0.852 | 0.849 | 0.846 | 0.848 |
| penrith | 0.552 | 0.473 | 0.484 | 0.480 | 0.479 |
| quillayute | 0.856 | 0.784 | 0.780 | 0.778 | 0.776 |

### FAR

| Location | before | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|---:|
| cairns_babinda | 0.066 | 0.060 | 0.056 | 0.060 | 0.059 |
| darwin | 0.369 | 0.332 | **0.285** | 0.300 | 0.284 |
| hilo | 0.203 | 0.161 | 0.165 | 0.162 | 0.165 |
| lake_margaret | 0.093 | 0.053 | **0.041** | 0.041 | 0.041 |
| penrith | 0.356 | 0.321 | 0.307 | 0.307 | 0.311 |
| quillayute | 0.323 | 0.265 | 0.257 | 0.259 | 0.260 |

## Key findings

1. **The palette fix is CSI-neutral on the full dataset.** The trade is POD ↓~5%
   vs FAR ↓~5%. UX-wise this favours the corrected palette (fewer false alarms
   at minor sensitivity cost).

2. **V2 is the clear winner among post-fix variants** (CSI 0.709, lowest FAR
   at 0.156, virtually identical to before_fix CSI 0.710). Adding just the
   innermost trace tier `(218,204,147)` captures the practical benefit; the
   outer two trace tiers are not needed.

3. **V3 ≈ V4 within noise** (both CSI 0.705). The `(170,158,121)` outermost
   trace tier has zero measurable impact.

4. **Per-location winners and losers:**
   - **darwin: biggest improvement post-fix.** CSI 0.559 → 0.618 in V2 (+0.059).
     FAR drops from 0.369 → 0.285. The fix significantly helps locations with
     heavy convective storm regimes.
   - **penrith: regresses.** CSI 0.423 → 0.394 in V4. POD drops 0.552 → 0.479.
     Sydney basin specifically loses sensitivity. Worth a follow-up
     investigation (separate ticket).
   - **lake_margaret: FAR halved** (0.093 → 0.041 in V2) - much cleaner detection.
   - cairns_babinda, hilo, quillayute: stable.
   - ketchikan: no rain in dataset, no meaningful metrics.

5. **Performance side-effect**: the palette fix is also a performance win on
   high-cell-density locations. quillayute took ~4h 32m on before_fix vs ~1h 54m
   on V1 (no trace tiers). Over-counted pixels in the broken palette inflated
   cell counts which dominate O(N²) tracking cost. See abrainwood/rain-incoming-backtester#18
   for the reporting splits that surfaced this.

## Recommended next action

Drop two palette entries from production:

- `(170, 158, 121, 0.02)` outermost trace
- `(206, 192, 135, 0.05)` middle trace

Keep `(218, 204, 147, 0.09)` as the only trace tier. This matches V2.

Implementation tracked in a separate issue.

## Replication

Per-variant scorecards live at `/tmp/issue180/scorecards_full/<variant>/`. Each
directory contains:

- `scorecards.json` - per-location aggregate
- `<location>.csv` - per-window predictions and verifier verdicts
- `scorecard.md` - human-readable report

The runner script that produced these is `/tmp/issue180/run_long_backtest.sh`.
Both the scorecards and the runner are throwaway artefacts and are not committed.

## Related

- PR #186 (palette intensity ordering fix)
- #180 (parent issue)
- #187 (slow-location investigation - mobile and quillayute)
- abrainwood/rain-incoming-backtester#18 (split reporting by clutter maturity and wet/dry truth)
