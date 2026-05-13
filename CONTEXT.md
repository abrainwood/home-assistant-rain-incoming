# Context: rain_incoming

Home Assistant custom integration that detects approaching rain from RainViewer radar tiles and exposes binary + arrival-time sensors for HA automations.

## Glossary

### Precipitation tier

A discrete colour in the RainViewer scheme 2 palette that maps to a dBZ range. The detector and renderer share a single palette (`PRECIP_COLOURS` in `providers/rainviewer.py`); any colour matched to a tier within the L2-distance threshold is treated as precipitation. Tiers below the "very light blue" `(0, 91, 142)` documented as ~16 dBZ are emitted by RainViewer as khaki tones (e.g. `(170, 158, 121)`) and are referred to as **trace tiers**.

Avoid the term **"intensity level"** as a synonym - it's overloaded with `_INTENSITY_THRESHOLD` (a luminance threshold for confidence-map dimming) and the `intensity` *value* (the 0.0-1.0 float each tier maps to). "Tier" specifically means the palette entry; "intensity" means the value attached to it.

### Trace tier / trace precipitation

The undocumented sub-blue RainViewer scheme 2 palette entries `(218, 204, 147)` / `(206, 192, 135)` / `(170, 158, 121)`, representing roughly 3-15 dBZ trace returns. Historically we treated these as land mask and dropped them; in fact they hug precipitation cells as concentric halos (see GH #180). Bringing them into `PRECIP_COLOURS` is the fix tracked under that issue.

### Land mask

Pixels in a RainViewer tile that represent *terrain colour* rather than radar return, with no overlapping precipitation. Both the detector's `_colour_to_intensity` and the renderer's `filter_precipitation_pixels` drop these by treating any pixel with L2 distance > `MAX_COLOUR_DISTANCE` from the palette as land mask. **Note:** the three trace tiers (above) are not land mask, despite their khaki appearance and our previous misclassification.

### Detector vs renderer (palette alignment)

The detector (rain-incoming sensor + QC pipeline) and the renderer (camera entity / animated GIF) share `PRECIP_COLOURS` as a single source of truth. They use the same L2-distance threshold concept but with different constants (`MAX_COLOUR_DISTANCE = 60` for detector, `_FILTER_MAX_COLOUR_DISTANCE = 30` for renderer). The threshold *delta* is empirically irrelevant: live RainViewer tiles emit zero pixels in the `(30, 60]` band - both colours match cleanly (`d ≤ 30`) or are unmatched (`d > 60`).

The rendering is intentionally a faithful visualisation of what the detector sees - if they disagreed, unexpected detection results would be impossible to diagnose visually.

<!--
Add terms here as they're resolved. Suggested format:

### TermName

What it means in this codebase. Note any synonyms the codebase **avoids**, so future agents and contributors don't drift back to them.
-->
