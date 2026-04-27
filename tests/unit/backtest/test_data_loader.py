"""Unit tests for scripts.backtest.data_loader."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.backtest.data_loader import (
    CaptureRecord,
    ObservationRecord,
    load_captures,
    load_captures_v2,
    load_observations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_meta(
    date_dir: Path,
    hhmm: str,
    frame_ts: int,
    location_name: str = "cairns_babinda",
    lat: float = -17.35,
    lon: float = 145.92,
    zoom: int = 7,
    tiles: list[dict] | None = None,
    capture_utc: str = "2026-04-21T10:25:34.123456Z",
) -> Path:
    """Write a synthetic _meta.json into date_dir and return its path."""
    date_dir.mkdir(parents=True, exist_ok=True)
    if tiles is None:
        tiles = [
            {"x": 127, "y": 89, "file": f"{hhmm}_127_89.png"},
            {"x": 128, "y": 89, "file": f"{hhmm}_128_89.png"},
        ]
    meta = {
        "capture_utc": capture_utc,
        "frame_ts": frame_ts,
        "frame_path": f"/nowcast/{frame_ts}",
        "location": {"name": location_name, "lat": lat, "lon": lon},
        "zoom": zoom,
        "tiles": tiles,
        "bytes_total": 12345,
    }
    meta_path = date_dir / f"{hhmm}_meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path


def _touch_tile(date_dir: Path, filename: str) -> Path:
    """Create an empty PNG stub so tile_paths can resolve to real files."""
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / filename
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# CaptureRecord tests
# ---------------------------------------------------------------------------


class TestLoadCapturesParsesMeta:
    def test_load_captures_parses_meta_json(self, tmp_path: Path) -> None:
        """Single meta.json produces a CaptureRecord with correct fields."""
        date_dir = tmp_path / "2026-04-21"
        _touch_tile(date_dir, "1025_127_89.png")
        _touch_tile(date_dir, "1025_128_89.png")
        _write_meta(date_dir, "1025", frame_ts=1771573200)

        records = load_captures(tmp_path)

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, CaptureRecord)
        assert rec.frame_ts == 1771573200
        assert rec.location_name == "cairns_babinda"
        assert rec.lat == pytest.approx(-17.35)
        assert rec.lon == pytest.approx(145.92)
        assert rec.zoom == 7
        expected_utc = datetime(2026, 4, 21, 10, 25, 34, 123456, tzinfo=timezone.utc)
        assert rec.capture_utc == expected_utc


class TestLoadCapturesSortsByFrameTs:
    def test_load_captures_sorts_by_frame_ts(self, tmp_path: Path) -> None:
        """Records are returned sorted by frame_ts ascending regardless of disk order."""
        date_dir = tmp_path / "2026-04-21"
        timestamps = [1771573400, 1771573200, 1771573600]
        for i, ts in enumerate(timestamps):
            hhmm = f"10{30 + i:02d}"
            tiles = [{"x": 127, "y": 89, "file": f"{hhmm}_127_89.png"}]
            _touch_tile(date_dir, f"{hhmm}_127_89.png")
            _write_meta(date_dir, hhmm, frame_ts=ts, tiles=tiles)

        records = load_captures(tmp_path)

        assert len(records) == 3
        assert [r.frame_ts for r in records] == sorted(timestamps)


class TestLoadCapturesSkipsDuplicateFrameTs:
    def test_load_captures_skips_duplicate_frame_ts(self, tmp_path: Path) -> None:
        """When two captures share a frame_ts, only the first seen is kept."""
        date_dir = tmp_path / "2026-04-21"
        shared_ts = 1771573200
        for hhmm in ("1025", "1026"):
            tiles = [{"x": 127, "y": 89, "file": f"{hhmm}_127_89.png"}]
            _touch_tile(date_dir, f"{hhmm}_127_89.png")
            _write_meta(date_dir, hhmm, frame_ts=shared_ts, tiles=tiles)

        records = load_captures(tmp_path)

        assert len(records) == 1
        assert records[0].frame_ts == shared_ts


class TestLoadCapturesResolvesTilePaths:
    def test_load_captures_resolves_tile_paths(self, tmp_path: Path) -> None:
        """tile_paths has correct (x, y) keys and absolute Path values."""
        date_dir = tmp_path / "2026-04-21"
        tiles = [
            {"x": 127, "y": 89, "file": "1025_127_89.png"},
            {"x": 128, "y": 89, "file": "1025_128_89.png"},
            {"x": 127, "y": 90, "file": "1025_127_90.png"},
            {"x": 128, "y": 90, "file": "1025_128_90.png"},
        ]
        for t in tiles:
            _touch_tile(date_dir, t["file"])
        _write_meta(date_dir, "1025", frame_ts=1771573200, tiles=tiles)

        records = load_captures(tmp_path)

        assert len(records) == 1
        tile_paths = records[0].tile_paths
        assert set(tile_paths.keys()) == {(127, 89), (128, 89), (127, 90), (128, 90)}
        for (x, y), path in tile_paths.items():
            assert path.is_absolute()
            assert path == date_dir / f"1025_{x}_{y}.png"


class TestLoadCapturesAcrossDateDirs:
    def test_load_captures_across_date_dirs(self, tmp_path: Path) -> None:
        """Captures in two different date directories are all discovered and sorted."""
        for date_str, hhmm, ts in [
            ("2026-04-20", "2350", 1771500000),
            ("2026-04-21", "0010", 1771501000),
            ("2026-04-21", "0020", 1771502000),
        ]:
            date_dir = tmp_path / date_str
            tiles = [{"x": 127, "y": 89, "file": f"{hhmm}_127_89.png"}]
            _touch_tile(date_dir, f"{hhmm}_127_89.png")
            _write_meta(date_dir, hhmm, frame_ts=ts, tiles=tiles)

        records = load_captures(tmp_path)

        assert len(records) == 3
        assert [r.frame_ts for r in records] == [1771500000, 1771501000, 1771502000]


class TestLoadCapturesEmptyDir:
    def test_load_captures_empty_dir(self, tmp_path: Path) -> None:
        """An empty location directory returns an empty list."""
        records = load_captures(tmp_path)
        assert records == []


class TestLoadCapturesIgnoresNonMetaFiles:
    def test_load_captures_ignores_non_meta_files(self, tmp_path: Path) -> None:
        """A directory containing only PNGs (no *_meta.json) returns empty list."""
        date_dir = tmp_path / "2026-04-21"
        _touch_tile(date_dir, "1025_127_89.png")
        _touch_tile(date_dir, "1025_128_89.png")

        records = load_captures(tmp_path)

        assert records == []


# ---------------------------------------------------------------------------
# ObservationRecord tests
# ---------------------------------------------------------------------------


class TestLoadObservationsParsesBom:
    def test_load_observations_parses_bom_jsonl(self, tmp_path: Path) -> None:
        """BOM-format JSONL line is parsed with correct fields."""
        obs_file = tmp_path / "2026-04-13.jsonl"
        line = json.dumps({
            "station_id": "94087",
            "obs_utc": "2026-04-13T17:45:00Z",
            "rain_mm": 2.4,
            "quality": 3,
            "source_file": "IDT65900_20260413195300.hcs",
        })
        obs_file.write_text(line + "\n")

        records = load_observations(tmp_path)

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, ObservationRecord)
        assert rec.station_id == "94087"
        expected_utc = datetime(2026, 4, 13, 17, 45, 0, tzinfo=timezone.utc)
        assert rec.obs_utc == expected_utc
        assert rec.rain_mm == pytest.approx(2.4)
        assert rec.is_raining is True


class TestLoadObservationsParsesMetar:
    def test_load_observations_parses_metar_jsonl(self, tmp_path: Path) -> None:
        """METAR-format JSONL line is parsed; is_raining and rain_mm are set correctly."""
        obs_file = tmp_path / "2026-04-14.jsonl"
        line = json.dumps({
            "station_id": "KMOB",
            "obs_utc": "2026-04-14T12:00:00.000Z",
            "is_raining": True,
            "wx_string": "RA",
            "precip_mm": 1.2,
            "lat": 30.6882,
            "lon": -88.2459,
            "elev_m": 67,
        })
        obs_file.write_text(line + "\n")

        records = load_observations(tmp_path)

        assert len(records) == 1
        rec = records[0]
        assert rec.station_id == "KMOB"
        assert rec.is_raining is True
        assert rec.rain_mm == pytest.approx(1.2)
        expected_utc = datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc)
        assert rec.obs_utc == expected_utc


class TestLoadObservationsSortsByObsUtc:
    def test_load_observations_sorts_by_obs_utc(self, tmp_path: Path) -> None:
        """Records from multiple JSONL files are merged and sorted by obs_utc."""
        file_a = tmp_path / "2026-04-13.jsonl"
        file_a.write_text(
            json.dumps({"station_id": "A", "obs_utc": "2026-04-13T18:00:00Z", "rain_mm": 0.0, "quality": 3, "source_file": "x.hcs"}) + "\n"
            + json.dumps({"station_id": "A", "obs_utc": "2026-04-13T17:00:00Z", "rain_mm": 1.0, "quality": 3, "source_file": "x.hcs"}) + "\n"
        )
        file_b = tmp_path / "2026-04-14.jsonl"
        file_b.write_text(
            json.dumps({"station_id": "B", "obs_utc": "2026-04-14T06:00:00Z", "rain_mm": 0.5, "quality": 3, "source_file": "y.hcs"}) + "\n"
        )

        records = load_observations(tmp_path)

        assert len(records) == 3
        utcs = [r.obs_utc for r in records]
        assert utcs == sorted(utcs)


class TestLoadObservationsEmptyDir:
    def test_load_observations_empty_dir(self, tmp_path: Path) -> None:
        """An empty directory returns an empty list."""
        records = load_observations(tmp_path)
        assert records == []


class TestLoadObservationsNullPrecipMm:
    def test_load_observations_handles_null_precip_mm(self, tmp_path: Path) -> None:
        """METAR with precip_mm=null produces rain_mm=None on the record."""
        obs_file = tmp_path / "2026-04-14.jsonl"
        line = json.dumps({
            "station_id": "KMOB",
            "obs_utc": "2026-04-14T12:00:00.000Z",
            "is_raining": False,
            "wx_string": "BCFG",
            "precip_mm": None,
            "lat": 30.6882,
            "lon": -88.2459,
            "elev_m": 67,
        })
        obs_file.write_text(line + "\n")

        records = load_observations(tmp_path)

        assert len(records) == 1
        assert records[0].rain_mm is None
        assert records[0].is_raining is False


# ---------------------------------------------------------------------------
# Corrupt file handling
# ---------------------------------------------------------------------------


class TestLoadCapturesCorruptMeta:
    """Corrupt or malformed meta.json must be skipped with a WARNING."""

    def test_corrupt_meta_json_skipped(self, tmp_path, caplog):
        day_dir = tmp_path / "2026-04-21"
        day_dir.mkdir()
        meta_file = day_dir / "1200_meta.json"
        meta_file.write_text("this is not json{{{")

        with caplog.at_level(logging.WARNING, logger="scripts.backtest.data_loader"):
            records = load_captures(tmp_path)

        assert records == [], "Corrupt meta.json should be skipped, not crash"
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "Corrupt meta.json must log a WARNING"
        )

    def test_missing_fields_in_meta_json_skipped(self, tmp_path, caplog):
        day_dir = tmp_path / "2026-04-21"
        day_dir.mkdir()
        meta_file = day_dir / "1200_meta.json"
        meta_file.write_text(json.dumps({"incomplete": True}))

        with caplog.at_level(logging.WARNING, logger="scripts.backtest.data_loader"):
            records = load_captures(tmp_path)

        assert records == [], "Meta with missing fields should be skipped"
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


class TestLoadObservationsCorruptJsonl:
    """Corrupt JSONL lines must be skipped with a WARNING."""

    def test_malformed_jsonl_line_skipped(self, tmp_path, caplog):
        obs_file = tmp_path / "2026-04-21.jsonl"
        good_line = json.dumps({
            "station_id": "94087",
            "obs_utc": "2026-04-21T10:00:00Z",
            "rain_mm": 0.0,
            "quality": 3,
        })
        obs_file.write_text(good_line + "\n" + "not valid json\n")

        with caplog.at_level(logging.WARNING, logger="scripts.backtest.data_loader"):
            records = load_observations(tmp_path)

        assert len(records) == 1, "Good line should be kept, bad line skipped"
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "Malformed JSONL line must log a WARNING"
        )

    def test_unreadable_obs_file_skipped(self, tmp_path, caplog):
        obs_file = tmp_path / "2026-04-21.jsonl"
        obs_file.write_bytes(b"\x80\x81\x82\x83")  # invalid UTF-8

        with caplog.at_level(logging.WARNING, logger="scripts.backtest.data_loader"):
            records = load_observations(tmp_path)

        assert records == [], "Unreadable file should be skipped"
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# load_captures_v2 (golden_v2 bronze format)
# ---------------------------------------------------------------------------


def _write_v2_bronze(
    bronze_dir: Path,
    *,
    location_name: str = "Perth_false_negative_20260427",
    lat: float = -31.95,
    lon: float = 115.86,
    zoom: int = 7,
    detection_colour_scheme: int = 6,
    render_colour_scheme: int = 2,
    capture_utc: str = "2026-04-27T04:34:15.558606+00:00",
    center_tile: tuple[int, int] = (105, 75),
    grid_half: int = 1,
    frames: list[tuple[int, str]] | None = None,
    create_tile_files: bool = True,
) -> Path:
    """Write a synthetic golden_v2 bronze/ directory with capture_info, manifest, and tile stubs.

    `frames` is a list of (frame_ts, path) tuples. Defaults to two frames.
    Returns the bronze directory.
    """
    if frames is None:
        frames = [
            (1777257000, "/v2/radar/b1a106dd76d3"),
            (1777257600, "/v2/radar/5a338ef5fafc"),
        ]

    cx, cy = center_tile
    tile_coords = [
        {"x": x, "y": y}
        for x in range(cx - grid_half, cx + grid_half + 1)
        for y in range(cy - grid_half, cy + grid_half + 1)
    ]

    bronze_dir.mkdir(parents=True, exist_ok=True)
    capture_info = {
        "location": {"lat": lat, "lon": lon, "name": location_name},
        "capture_time_utc": capture_utc,
        "zoom": zoom,
        "center_tile": {"x": cx, "y": cy},
        "grid_half": grid_half,
        "tile_coords": tile_coords,
        "detection_colour_scheme": detection_colour_scheme,
        "render_colour_scheme": render_colour_scheme,
        "frame_count": len(frames),
    }
    (bronze_dir / "capture_info.json").write_text(json.dumps(capture_info))

    manifest = {
        "version": "2.0",
        "radar": {
            "past": [{"time": ts, "path": path} for ts, path in frames],
        },
    }
    (bronze_dir / "manifest.json").write_text(json.dumps(manifest))

    if create_tile_files:
        tiles_root = bronze_dir / "tiles"
        for ts, _path in frames:
            frame_dir = tiles_root / str(ts)
            frame_dir.mkdir(parents=True, exist_ok=True)
            for t in tile_coords:
                tile_file = frame_dir / f"{zoom}_{t['x']}_{t['y']}_s{detection_colour_scheme}.png"
                tile_file.write_bytes(b"")

    return bronze_dir


class TestLoadCapturesV2HappyPath:
    def test_load_captures_v2_returns_records_with_correct_fields(
        self, tmp_path: Path
    ) -> None:
        """`load_captures_v2` reads bronze/{capture_info,manifest}.json and returns
        CaptureRecord list sorted by frame_ts ascending with correct fields."""
        bronze = tmp_path / "bronze"
        _write_v2_bronze(
            bronze,
            location_name="Perth_false_negative_20260427",
            lat=-31.95,
            lon=115.86,
            zoom=7,
            capture_utc="2026-04-27T04:34:15.558606+00:00",
            frames=[
                (1777257000, "/v2/radar/b1a106dd76d3"),
                (1777257600, "/v2/radar/5a338ef5fafc"),
            ],
        )

        records = load_captures_v2(bronze)

        assert len(records) == 2
        assert all(isinstance(r, CaptureRecord) for r in records)
        assert [r.frame_ts for r in records] == [1777257000, 1777257600]
        for rec in records:
            assert rec.location_name == "Perth_false_negative_20260427"
            assert rec.lat == pytest.approx(-31.95)
            assert rec.lon == pytest.approx(115.86)
            assert rec.zoom == 7
            expected_utc = datetime(
                2026, 4, 27, 4, 34, 15, 558606, tzinfo=timezone.utc
            )
            assert rec.capture_utc == expected_utc

    def test_tile_paths_have_correct_format(self, tmp_path: Path) -> None:
        """tile_paths keys are (x, y) tuples; values point to the right filename."""
        bronze = tmp_path / "bronze"
        _write_v2_bronze(bronze, zoom=7, detection_colour_scheme=6, grid_half=1, center_tile=(105, 75))

        records = load_captures_v2(bronze)

        rec = records[0]  # frame_ts=1777257000
        assert (105, 75) in rec.tile_paths
        expected = bronze / "tiles" / "1777257000" / "7_105_75_s6.png"
        assert rec.tile_paths[(105, 75)] == expected

    def test_out_of_order_manifest_sorted_by_frame_ts(self, tmp_path: Path) -> None:
        """Records are sorted ascending by frame_ts even if manifest lists them out of order."""
        bronze = tmp_path / "bronze"
        _write_v2_bronze(bronze, frames=[
            (1777257600, "/v2/radar/second"),
            (1777257000, "/v2/radar/first"),
        ])

        records = load_captures_v2(bronze)

        assert [r.frame_ts for r in records] == [1777257000, 1777257600]

    def test_missing_capture_info_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when capture_info.json is absent."""
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        (bronze / "manifest.json").write_text('{"version":"2.0","radar":{"past":[]}}')

        with pytest.raises((FileNotFoundError, OSError)):
            load_captures_v2(bronze)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when manifest.json is absent."""
        bronze = tmp_path / "bronze"
        _write_v2_bronze(bronze, create_tile_files=False)
        (bronze / "manifest.json").unlink()

        with pytest.raises((FileNotFoundError, OSError)):
            load_captures_v2(bronze)


class TestLoadCapturesAutoDetect:
    def test_auto_detects_v2_format(self, tmp_path: Path) -> None:
        """load_captures delegates to load_captures_v2 when bronze/capture_info.json exists."""
        session_dir = tmp_path / "Perth_false_negative_20260427"
        bronze = session_dir / "bronze"
        _write_v2_bronze(bronze, frames=[(1777257000, "/v2/radar/x")])

        records = load_captures(session_dir)

        assert len(records) == 1
        assert records[0].frame_ts == 1777257000
        assert records[0].location_name == "Perth_false_negative_20260427"
