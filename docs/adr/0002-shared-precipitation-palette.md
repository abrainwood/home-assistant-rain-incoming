# 2. Detector and renderer share a single precipitation palette

Date: 2026-05-13

## Status

Accepted

## Context

Two components in the integration classify RainViewer tile pixels as precipitation:

- The **detector** (`providers/rainviewer.py:_colour_to_intensity` and its vectorised cousin `_tile_to_intensity_array`) maps each pixel to a 0.0-1.0 intensity value used by the rain-incoming sensor and the QC pipeline.
- The **renderer** (`radar/composite.py:filter_precipitation_pixels`) zeroes the alpha channel of pixels that aren't precipitation, so the camera entity / animated GIF only shows real returns over the user's basemap.

Both components currently share `PRECIP_COLOURS` (the documented palette in `providers/rainviewer.py`) as the single source of truth for "which RGB values count as precipitation." They differ in one constant: the L2-distance threshold for matching (detector: 60, renderer: 30). This delta was investigated in GH #180 and found to be empirically irrelevant - real RainViewer tiles emit zero pixels in the `(30, 60]` band, so the threshold difference never reclassifies a pixel.

While extending `PRECIP_COLOURS` with the previously-undocumented trace tiers identified in GH #180, the question arose: should the renderer carry a *richer* palette than the detector? Specifically, low-dBZ trace tiers that improve the rendered image's resemblance to BOM imagery without affecting (or possibly hurting) the detector's POD/FAR.

The alternative considered was a split palette: `PRECIP_COLOURS` for the detector, a separate `RENDER_PALETTE` (a superset) for the renderer.

## Decision

The detector and renderer continue to share a single palette - the renderer renders **exactly** what the detector classifies as precipitation, and nothing more. When the palette is extended (or trimmed), both consumers move in lock-step.

## Consequences

- **Diagnosability:** the rendered camera entity is a faithful visualisation of what the detector is acting on. When the rain-incoming sensor fires unexpectedly (or fails to fire), a user can inspect the rendered loop and trust that "what they see is what the detector saw." A split palette would break this invariant - the user could be looking at trace pixels the detector ignored, or vice versa, and chase phantom bugs.
- **Backtest coupling:** any palette extension changes detection sensitivity. Extensions must be validated via the rain-incoming-backtester (POD / FAR / CSI on the curated subset) before merging. There is no longer a way to ship "renderer-only" palette changes without engaging the backtest discipline.
- **Rendering ceiling:** the rendered output cannot be richer than the detector's classification. If a future requirement is "show trace precipitation in the camera image *without* making the detector trigger on it," this ADR needs revising and the palette must be split. That's a known tradeoff, not a hidden cost.
- **Two constants, one truth:** `MAX_COLOUR_DISTANCE` (detector) and `_FILTER_MAX_COLOUR_DISTANCE` (renderer) remain separate values. They were preserved during this decision because their empirical disagreement set is empty, so unifying them is cosmetic; if they ever start meaningfully diverging on real data, treat that as a defect in this ADR's premise and revisit.
