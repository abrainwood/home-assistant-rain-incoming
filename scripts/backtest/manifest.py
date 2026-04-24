"""Manifest: a curated list of windows for quick backtesting.

A manifest selects specific windows from a full baseline run that
represent the important categories (hits, misses, false alarms,
correct negatives) for fast algorithm iteration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    """One selected window in the manifest."""

    location: str
    window_end_ts: int
    category: str       # hit, miss, false_alarm, correct_negative
    subcategory: str    # strong, marginal, overhead_noise, dissipated, etc.


@dataclass
class Manifest:
    """A curated set of windows for quick backtesting."""

    name: str
    description: str
    created_from: str
    created_at: str
    entries: list[ManifestEntry]


def save_manifest(
    entries: list[ManifestEntry],
    name: str,
    description: str,
    source_dir: str,
    path: Path,
) -> None:
    """Save a manifest to a JSON file."""
    data = {
        "name": name,
        "description": description,
        "created_from": source_dir,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "windows": [
            {
                "location": e.location,
                "window_end_ts": e.window_end_ts,
                "category": e.category,
                "subcategory": e.subcategory,
            }
            for e in entries
        ],
    }
    path.write_text(json.dumps(data, indent=2))


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from a JSON file."""
    data = json.loads(path.read_text())
    entries = [
        ManifestEntry(
            location=w["location"],
            window_end_ts=w["window_end_ts"],
            category=w["category"],
            subcategory=w["subcategory"],
        )
        for w in data["windows"]
    ]
    return Manifest(
        name=data["name"],
        description=data["description"],
        created_from=data["created_from"],
        created_at=data["created_at"],
        entries=entries,
    )
