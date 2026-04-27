"""DataLoader for the backtesting framework.

Scans captured radar data directories, parses metadata, and returns structured
records for replay.  Also loads ground-truth observation JSONL files.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureRecord:
    """A single radar capture from disk."""

    capture_utc: datetime
    frame_ts: int                            # Unix timestamp of the radar frame
    tile_paths: dict[tuple[int, int], Path]  # (x, y) -> absolute path to PNG
    location_name: str
    lat: float
    lon: float
    zoom: int


@dataclass(frozen=True)
class ObservationRecord:
    """A single ground truth observation."""

    station_id: str
    obs_utc: datetime
    rain_mm: float | None  # None for METAR without precip data
    is_raining: bool        # True if rain detected


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def _parse_utc(ts_str: str) -> datetime:
    """Parse an ISO 8601 string (with Z or +00:00) into an aware UTC datetime."""
    # datetime.fromisoformat() handles '+00:00' natively in Python 3.11+,
    # but not the 'Z' suffix until Python 3.11.  Normalise 'Z' to '+00:00'
    # to be safe across 3.12 (which does support it, but belt-and-braces).
    normalised = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(normalised).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Capture loading
# ---------------------------------------------------------------------------


def _parse_capture(meta_path: Path) -> CaptureRecord:
    """Parse a single *_meta.json file into a CaptureRecord."""
    raw = json.loads(meta_path.read_text())

    location = raw["location"]
    tiles_raw: list[dict] = raw["tiles"]

    tile_paths: dict[tuple[int, int], Path] = {
        (t["x"], t["y"]): (meta_path.parent / t["file"]).resolve()
        for t in tiles_raw
    }

    return CaptureRecord(
        capture_utc=_parse_utc(raw["capture_utc"]),
        frame_ts=raw["frame_ts"],
        tile_paths=tile_paths,
        location_name=location["name"],
        lat=location["lat"],
        lon=location["lon"],
        zoom=raw["zoom"],
    )


def load_captures(location_dir: Path) -> list[CaptureRecord]:
    """Scan a location directory and return sorted CaptureRecord list.

    Auto-detects format: if a bronze/capture_info.json exists inside
    location_dir, delegates to load_captures_v2(location_dir / "bronze").
    Otherwise scans *_meta.json files (backtest format).

    Returns records sorted by frame_ts ascending.
    Skips duplicate frame_ts values (keeps the first seen, in filesystem order).
    """
    if (location_dir / "bronze" / "capture_info.json").exists():
        return load_captures_v2(location_dir / "bronze")

    records: list[CaptureRecord] = []
    seen_frame_ts: set[int] = set()

    for meta_path in sorted(location_dir.rglob("*_meta.json")):
        try:
            record = _parse_capture(meta_path)
        except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
            _LOGGER.warning("Skipping corrupt meta file %s: %s", meta_path, exc)
            continue
        if record.frame_ts in seen_frame_ts:
            continue
        seen_frame_ts.add(record.frame_ts)
        records.append(record)

    records.sort(key=lambda r: r.frame_ts)
    return records


def load_captures_v2(bronze_dir: Path) -> list[CaptureRecord]:
    """Load captures from a golden_v2 bronze/ directory.

    Reads `bronze_dir/capture_info.json` for location/zoom/tile_coords and
    `bronze_dir/manifest.json` for the list of past radar frame timestamps.
    Builds tile_paths as `bronze_dir/tiles/{frame_ts}/{zoom}_{x}_{y}_s{scheme}.png`.
    Returns CaptureRecord list sorted by frame_ts ascending.
    """
    info = json.loads((bronze_dir / "capture_info.json").read_text())
    manifest = json.loads((bronze_dir / "manifest.json").read_text())

    location = info["location"]
    zoom = info["zoom"]
    scheme = info["detection_colour_scheme"]
    tile_coords: list[dict] = info["tile_coords"]
    capture_utc = _parse_utc(info["capture_time_utc"])

    tiles_root = bronze_dir / "tiles"
    records: list[CaptureRecord] = []
    for frame in manifest["radar"]["past"]:
        frame_ts = frame["time"]
        frame_dir = tiles_root / str(frame_ts)
        tile_paths = {
            (t["x"], t["y"]): frame_dir / f"{zoom}_{t['x']}_{t['y']}_s{scheme}.png"
            for t in tile_coords
        }
        records.append(
            CaptureRecord(
                capture_utc=capture_utc,
                frame_ts=frame_ts,
                tile_paths=tile_paths,
                location_name=location["name"],
                lat=location["lat"],
                lon=location["lon"],
                zoom=zoom,
            )
        )

    records.sort(key=lambda r: r.frame_ts)
    return records


# ---------------------------------------------------------------------------
# Observation loading
# ---------------------------------------------------------------------------


def _parse_observation_line(raw: dict) -> ObservationRecord:
    """Parse a single JSON object (dict) into an ObservationRecord.

    Supports both BOM format (rain_mm field) and METAR format
    (is_raining + precip_mm fields).
    """
    obs_utc = _parse_utc(raw["obs_utc"])

    if "rain_mm" in raw:
        # BOM format
        rain_mm: float | None = raw["rain_mm"]
        is_raining = rain_mm is not None and rain_mm > 0.0
    else:
        # METAR format
        rain_mm = raw.get("precip_mm")  # may be None
        is_raining = bool(raw["is_raining"])

    return ObservationRecord(
        station_id=raw["station_id"],
        obs_utc=obs_utc,
        rain_mm=rain_mm,
        is_raining=is_raining,
    )


def load_observations(obs_dir: Path) -> list[ObservationRecord]:
    """Load observation JSONL files from a directory.

    Reads all .jsonl files, parses each line as JSON.  Handles both BOM
    (rain_mm field) and METAR (is_raining + precip_mm fields) formats.
    Returns records sorted by obs_utc ascending.
    """
    records: list[ObservationRecord] = []

    for jsonl_path in sorted(obs_dir.rglob("*.jsonl")):
        try:
            text = jsonl_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            _LOGGER.warning("Skipping unreadable obs file %s: %s", jsonl_path, exc)
            continue
        for line_num, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(_parse_observation_line(raw))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                _LOGGER.warning(
                    "Skipping malformed line %d in %s: %s", line_num, jsonl_path, exc
                )

    records.sort(key=lambda r: r.obs_utc)
    return records
