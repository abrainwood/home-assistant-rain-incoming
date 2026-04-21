"""DataLoader for the backtesting framework.

Scans captured radar data directories, parses metadata, and returns structured
records for replay.  Also loads ground-truth observation JSONL files.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    """Scan a location directory, parse meta.json files, return sorted records.

    Scans all date subdirectories, finds *_meta.json files, parses each, and
    builds tile_paths by resolving tile filenames relative to the meta.json's
    directory.  Returns records sorted by frame_ts ascending.

    Skips duplicate frame_ts values (keeps the first seen, in filesystem order).
    """
    records: list[CaptureRecord] = []
    seen_frame_ts: set[int] = set()

    for meta_path in sorted(location_dir.rglob("*_meta.json")):
        record = _parse_capture(meta_path)
        if record.frame_ts in seen_frame_ts:
            continue
        seen_frame_ts.add(record.frame_ts)
        records.append(record)

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
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(_parse_observation_line(raw))

    records.sort(key=lambda r: r.obs_utc)
    return records
