# Golden Test Data - 8 Apr 2026 ~21:30 AEST

## Ground truth
- **Sydney/Terry Hills**: Clear sky, stars visible, NO rain. All radar returns are AP noise.
- **SW quadrant (512km)**: Real rain moving W->E, visible on BOM 512km radar
- **BOM FTP**: Worst noise night seen - heavy AP/ground clutter on all products
- **Open-Meteo**: 0.0mm at Terry Hills

## What this captures
Perfect test case: noise + real rain in the same data. Use for QC tuning.

## Known issue discovered
The QC system obliterates the real SW rain in the 512km view because:
1. Confidence maps are computed on the analysis grid (256x256, ~3 degree box around location)
2. The 512km render viewport extends far beyond the analysis grid
3. Pixels outside the analysis grid get no confidence data -> default to invisible
4. The real rain in the SW is outside the analysis bounds

The fix: rendering-time confidence should default to 1.0 (show everything) for pixels
outside the QC analysis area, not 0.0 (hide everything). The QC can only vouch for
pixels it actually analyzed.

## Files
- `manifest_20260408.json` - frame metadata + ground truth notes
- `grids_20260408_terry_hills.npz` - 8 intensity grids (256x256, analysis bounds)
- `confidence_20260408_terry_hills.npz` - 8 confidence maps from full QC pipeline
