"""Unit tests for scripts.backtest.manifest - manifest save/load/filter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


class TestManifestRoundtrip:
    def test_save_and_load_preserves_entries(self, tmp_path):
        """Manifest must roundtrip: save then load produces identical entries."""
        from scripts.backtest.manifest import ManifestEntry, save_manifest, load_manifest

        entries = [
            ManifestEntry(
                location="darwin",
                window_end_ts=1776345600,
                category="false_alarm",
                subcategory="overhead_noise",
            ),
            ManifestEntry(
                location="darwin",
                window_end_ts=1776346200,
                category="hit",
                subcategory="strong",
            ),
        ]
        path = tmp_path / "test.json"
        save_manifest(entries, "test-v1", "Test manifest", "reports/baseline", path)

        loaded = load_manifest(path)
        assert loaded.name == "test-v1"
        assert loaded.description == "Test manifest"
        assert loaded.created_from == "reports/baseline"
        assert len(loaded.entries) == 2
        assert loaded.entries[0].location == "darwin"
        assert loaded.entries[0].window_end_ts == 1776345600
        assert loaded.entries[0].category == "false_alarm"
        assert loaded.entries[0].subcategory == "overhead_noise"
