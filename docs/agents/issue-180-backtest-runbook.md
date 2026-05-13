# GH #180 backtest runbook - palette extension experiment

Self-contained procedure for backtesting four palette variants and picking a winner. A future session (likely from the parent directory `~/src/projects/homeassistant/`) should be able to follow this end-to-end without referencing other docs.

> **Status:** experiment in flight. Variant branches pushed. Backtester not yet run. Delete this file (and the diagnostic scripts under `scripts/`) when #180 closes.

## Goal

The current `PRECIP_COLOURS` palette in `custom_components/rain_incoming/providers/rainviewer.py` is missing three previously-undocumented RainViewer scheme 2 trace tiers (khaki tones at `(170,158,121)`, `(206,192,135)`, `(218,204,147)`) that spatially form concentric halos around precipitation cells. Adding them increases detector sensitivity to trace precipitation; the question is whether that helps, hurts, or is neutral for detection performance.

Backtest four variants and pick the one that maximises detection performance (POD / FAR / CSI) without regressing on the curated subset. Independently, render the same frames against each variant and visually compare to BOM imagery (the user does this visual step).

## Variants

| Variant | Branch | Trace tiers added (RGB, intensity) | Rationale |
|---|---|---|---|
| V1 | `main` | none | Baseline. |
| V2 | `experiment/180-palette-v2` | `(170, 158, 121, 0.09)` | Most conservative - one new tier. |
| V3 | `experiment/180-palette-v3` | + `(206, 192, 135, 0.05)` | User's stated hypothesis ("kill only the lowest of the three khaki tones"). |
| V4 | `experiment/180-palette-v4` | + `(218, 204, 147, 0.02)` | Most aggressive - all three trace tiers. |

Each experiment branch is exactly one commit on top of `main` (palette change + two `pytest.mark.skip` markers on tests that encode the old khaki-is-land-mask assumption + one offset fix in `test_parser_rejects_beyond_max_colour_distance` where prepending a high-R trace tier to `PRECIP_COLOURS[0]` causes overflow in the existing offset constant).

## Prerequisites

- Two repos checked out side-by-side as siblings:
  - `~/src/projects/homeassistant/home-assistant-rain-incoming` (this repo)
  - `~/src/projects/homeassistant/rain-incoming-backtester` (private)
- The backtester treats this repo as an editable local dep via `pip install -e ../home-assistant-rain-incoming`. **Whatever branch is checked out here is the code the backtester sees.**
- Both `.venv` environments set up (this repo: `pip install -e ".[dev]"`; backtester: see its own README).
- Backtester data: `backtest_data/captures/` and `backtest_data/observations/` populated (the user already has these).

## Procedure

### Step 1: Run the backtester on each variant

For each variant, the procedure is identical except for which branch is checked out in *this* repo.

```bash
# From rain-incoming-backtester:
cd ~/src/projects/homeassistant/rain-incoming-backtester
mkdir -p /tmp/issue180/scorecards

# V1 baseline
( cd ../home-assistant-rain-incoming && git checkout main )
.venv/bin/python -m scripts.backtest \
    --locations all \
    --qc full \
    --verify \
    --output-dir /tmp/issue180/scorecards/v1 \
    2>&1 | tee /tmp/issue180/scorecards/v1.log

# V2
( cd ../home-assistant-rain-incoming && git checkout experiment/180-palette-v2 )
.venv/bin/python -m scripts.backtest \
    --locations all \
    --qc full \
    --verify \
    --output-dir /tmp/issue180/scorecards/v2 \
    2>&1 | tee /tmp/issue180/scorecards/v2.log

# V3
( cd ../home-assistant-rain-incoming && git checkout experiment/180-palette-v3 )
.venv/bin/python -m scripts.backtest \
    --locations all \
    --qc full \
    --verify \
    --output-dir /tmp/issue180/scorecards/v3 \
    2>&1 | tee /tmp/issue180/scorecards/v3.log

# V4
( cd ../home-assistant-rain-incoming && git checkout experiment/180-palette-v4 )
.venv/bin/python -m scripts.backtest \
    --locations all \
    --qc full \
    --verify \
    --output-dir /tmp/issue180/scorecards/v4 \
    2>&1 | tee /tmp/issue180/scorecards/v4.log

# Back to main when done
( cd ../home-assistant-rain-incoming && git checkout main )
```

**Time estimate:** unknown - depends on dataset size. Run V1 first, time it, multiply by four for total budget.

**If using a curated subset instead of all data,** add `--subset path/to/manifest.json` to each command. The Phase 0.5 / Phase -2 curated subsets in the backtester repo are the standard A/B knobs - see `docs/backtest-investigations.md` in this repo for what's been used historically.

### Step 2: Compile the comparison table

Each variant produces a `scorecards.json` file under `/tmp/issue180/scorecards/<variant>/`. Read all four and tabulate POD / FAR / CSI per location, plus aggregate.

```bash
for v in v1 v2 v3 v4; do
    echo "=== $v ==="
    cat /tmp/issue180/scorecards/$v/scorecards.json | jq '.'
    echo
done
```

The backtester also supports `--compare`, which produces a per-variant delta against a prior run. Useful for V2/V3/V4 vs V1:

```bash
# Compare V2 against V1 baseline
.venv/bin/python -m scripts.backtest \
    --locations all --qc full --verify \
    --output-dir /tmp/issue180/scorecards/v2-vs-v1 \
    --compare /tmp/issue180/scorecards/v1 \
    2>&1 | tee /tmp/issue180/scorecards/v2-vs-v1.log
# (Requires checking out v2 first)
```

### Step 3: Visual compare (user-driven, but render fixtures here)

For the visual side, pick 3-5 fixture timestamps where chunks-vanishing was observed (Sydney basin during May 2026 storms is a known good source - see GH #180 image attachments). For each timestamp, render the same composite under each variant:

```bash
# From this repo (home-assistant-rain-incoming):
for variant in main experiment/180-palette-v2 experiment/180-palette-v3 experiment/180-palette-v4; do
    git checkout "$variant"
    branch_name=$(echo "$variant" | tr '/' '_')
    mkdir -p /tmp/issue180/render/"$branch_name"
    # Use existing render helper - precise invocation depends on what's available;
    # the user has `make dev` for a live integration, or you can call
    # render_animated_composite() directly from a Python REPL.
done
git checkout main
```

The user reviews these side-by-side against BOM imagery for the same timestamps. Visual judgement: does the new variant look more like BOM's coherent precipitation, or has it added false signal?

### Step 4: Decide and clean up

**Decision criteria:**
- **Pick V<n> if:** its CSI is ≥ V1 baseline (no detection regression) AND its visual compare looks better than V1 (closer to BOM). If multiple variants meet both bars, pick the most conservative.
- **Stick with V1 (close #180 with "no change") if:** all variants regress detection OR none visually improve.
- **Split palettes (revisit ADR-0002) if:** a variant clearly improves the visual but regresses detection. That's the one outcome that breaks the shared-palette assumption. Don't do this lightly - the ADR explicitly notes it requires revisiting.

After deciding, on **this** repo:

```bash
# Delete throwaway experiment branches both locally and on origin
for v in v2 v3 v4; do
    git push origin --delete experiment/180-palette-$v
    git branch -D experiment/180-palette-$v
done

# Delete the temp scorecard dir
rm -rf /tmp/issue180

# Open Implementation PR #2: TDD-driven implementation of the winning palette
git checkout main && git pull
git checkout -b feat/issue-180-extend-precip-palette
# ... TDD cycle in the implementation PR, see GH #180 for what tests need writing ...
```

If V1 wins (no change), close GH #180 with a comment summarising the negative finding - the diagnostic scripts will still have documented the bug for future readers, and ADR-0002 still applies.

## Open variables this runbook can't pin down

- **Intensity values for V2/V3/V4** (`0.09 / 0.05 / 0.02`) are extrapolated guesses below the existing `0.10` floor. If the backtest signal is weak (small delta either direction), a follow-up sensitivity analysis may be needed to tune intensities. Tracked as a "potential follow-up" in the planning conversation - not part of this experiment.
- **Subset vs full dataset.** Runbook defaults to `--locations all`. If runtime is prohibitive, switch to the curated subset used in prior backtests.
- **Per-location vs aggregate.** Aggregate POD/FAR/CSI can mask regressions in specific locations (e.g. coastal vs inland). Inspect per-location deltas before concluding.

## References

- GH #180 (issue with corrected diagnosis + comment thread)
- PR #184 (this PR - investigation context, ADR-0002, glossary, diagnostic scripts)
- `docs/adr/0002-shared-precipitation-palette.md`
- `CONTEXT.md` glossary (precipitation tier, trace tier, land mask, detector/renderer palette alignment)
- `scripts/diagnose_threshold_mismatch.py` / `inspect_unmatched_colours.py` / `spatial_unmatched.py` (the diagnostic that disproved the original hypothesis)
- `../rain-incoming-backtester/CONTEXT.md` (backtester domain language - clutter map, noise taxonomy etc.)
