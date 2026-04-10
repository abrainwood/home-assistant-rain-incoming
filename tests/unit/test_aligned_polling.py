"""Tests for aligned polling schedule.

TDD: these tests are written FIRST, before the implementation.

Polling should align to minutes 01, 11, 21, 31, 41, 51 past the hour
so that data is always ~1 minute behind RainViewer's 10-minute updates,
regardless of when HA started.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.rain_incoming.coordinator import next_aligned_poll_time


class TestNextAlignedPollTime:
    """next_aligned_poll_time returns the next time at minute 01/11/21/31/41/51."""

    def test_just_after_the_hour(self):
        """At 10:00:30, next poll should be 10:01:00."""
        now = datetime(2026, 4, 10, 10, 0, 30, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 10, 1, 0, tzinfo=timezone.utc)

    def test_at_01_should_return_11(self):
        """At 10:01:00 exactly, next poll should be 10:11:00."""
        now = datetime(2026, 4, 10, 10, 1, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 10, 11, 0, tzinfo=timezone.utc)

    def test_mid_window(self):
        """At 10:05:00, next poll should be 10:11:00."""
        now = datetime(2026, 4, 10, 10, 5, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 10, 11, 0, tzinfo=timezone.utc)

    def test_at_51_should_wrap_to_next_hour(self):
        """At 10:51:00 exactly, next poll should be 11:01:00."""
        now = datetime(2026, 4, 10, 10, 51, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 11, 1, 0, tzinfo=timezone.utc)

    def test_at_55_should_wrap_to_next_hour(self):
        """At 10:55:00, next poll should be 11:01:00."""
        now = datetime(2026, 4, 10, 10, 55, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 11, 1, 0, tzinfo=timezone.utc)

    def test_at_59_should_wrap_to_next_hour(self):
        """At 23:59:30, next poll should be 00:01:00 next day."""
        now = datetime(2026, 4, 10, 23, 59, 30, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 11, 0, 1, 0, tzinfo=timezone.utc)

    def test_returns_utc(self):
        """Result should always be UTC."""
        now = datetime(2026, 4, 10, 10, 5, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result.tzinfo == timezone.utc

    def test_seconds_until_next_poll(self):
        """At 10:09:45, next poll is 10:11:00 = 75 seconds away."""
        now = datetime(2026, 4, 10, 10, 9, 45, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        delta = (result - now).total_seconds()
        assert delta == 75.0

    def test_at_exactly_aligned_time_returns_next_slot(self):
        """At 10:21:00.000, should return 10:31:00 not 10:21:00."""
        now = datetime(2026, 4, 10, 10, 21, 0, tzinfo=timezone.utc)
        result = next_aligned_poll_time(now)
        assert result == datetime(2026, 4, 10, 10, 31, 0, tzinfo=timezone.utc)
