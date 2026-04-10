"""Tests for BOM-style GIF labels.

TDD: these tests are written FIRST, before the implementation.

The label at the bottom of each GIF frame should show:
  Location  DD/MM/YY  HH:MM TZ  Range

Examples:
  "Sydney  10/04/26  20:40 AEST  128km"
  "10/04/26  20:40 UTC  64km"  (no location name set)
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from custom_components.rain_incoming.radar.composite import build_frame_label


class TestBuildFrameLabel:
    """Test the label string builder."""

    def test_full_label_with_location(self):
        ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
        label = build_frame_label(
            timestamp=ts, tz_name="UTC", radius_km=128, location_name="Sydney",
        )
        assert label == "Sydney  10/04/26  20:40 UTC  128km"

    def test_label_without_location(self):
        ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
        label = build_frame_label(
            timestamp=ts, tz_name="UTC", radius_km=64,
        )
        assert label == "10/04/26  20:40 UTC  64km"

    def test_label_with_local_timezone(self):
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Australia/Sydney")
        ts = datetime(2026, 4, 10, 9, 40, 0, tzinfo=tz)
        label = build_frame_label(
            timestamp=ts, tz_name="Australia/Sydney", radius_km=256,
            location_name="Lake Margaret",
        )
        # strftime %Z on zoneinfo gives "AEST" for +10
        tz_str = ts.strftime("%Z") or "AEST"
        assert "Lake Margaret" in label
        assert "10/04/26" in label
        assert "09:40" in label
        assert "256km" in label

    def test_label_with_512km(self):
        ts = datetime(2026, 4, 10, 20, 40, 0, tzinfo=timezone.utc)
        label = build_frame_label(
            timestamp=ts, tz_name="UTC", radius_km=512, location_name="Babinda",
        )
        assert label == "Babinda  10/04/26  20:40 UTC  512km"

    def test_no_timestamp_returns_range_only(self):
        label = build_frame_label(
            timestamp=None, tz_name=None, radius_km=128, location_name="Sydney",
        )
        assert "Sydney" in label
        assert "128km" in label
