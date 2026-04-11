"""Integration tests for the incremental frame cache in RainDetectorCoordinator.

These tests exercise the real _async_update_data() rather than a copy of it.
Each scenario patches get_frames (network) and per-frame _fetch_stitched_grid
(tile HTTP calls) so the production cache-resolution code runs exactly as it
does in production - the tests just control the inputs and observe
coordinator._frame_cache after each call.

Why this file is in tests/integration/ instead of tests/unit/:
  The tests construct a real RainDetectorCoordinator with a real HomeAssistant
  instance. That requires the hass fixture from pytest-homeassistant-custom-
  component, which makes this an integration boundary.

The conftest.py autouse fixture skips modules whose __name__ contains
"test_coordinator", so double-patching is not a concern here.

IMPORTANT: the coordinator's _fetch_frame() swallows all exceptions from
_fetch_stitched_grid (logging them as warnings). This means we can't detect
unwanted fetch calls by raising from patched methods - we must use tracking
lists instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from homeassistant.core import HomeAssistant

from custom_components.rain_incoming.coordinator import RainDetectorCoordinator
from custom_components.rain_incoming.providers.rainviewer import RainViewerFrame
from custom_components.rain_incoming.radar.detector import Confidence, DetectionResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_entry(lat: float = -33.701, lon: float = 151.209, lookahead: int = 60):
    """Minimal config entry mock - same pattern as test_coordinator_rate_limit."""
    entry = MagicMock()
    entry.data = {
        "latitude": lat,
        "longitude": lon,
        "lookahead_minutes": lookahead,
    }
    entry.entry_id = "frame_cache_test"
    return entry


def _make_fresh_frame(path: str, ts: int = 0) -> RainViewerFrame:
    """Create a RainViewerFrame as returned by get_frames - no cached grid yet."""
    return RainViewerFrame(
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
        path=path,
        zoom=7,
        colour_scheme=2,
    )


def _patch_frame_fetch_succeeds(frame: RainViewerFrame, fetch_log: list[str] | None = None):
    """Patch _fetch_stitched_grid to populate the frame's cache with a non-zero grid.

    If fetch_log is provided, appends frame.path when called so tests can
    assert which frames were actually fetched over the network.

    The fake stores bounds + size exactly as the coordinator will pass them, so
    get_intensity_grid() cache-hits when the coordinator calls it later.
    """
    async def _fake_fetch(bounds, width, height, session):
        if fetch_log is not None:
            fetch_log.append(frame.path)
        frame._cached_grid = np.full((height, width), 0.5, dtype=np.float32)
        frame._cached_bounds = bounds
        return frame._cached_grid

    frame._fetch_stitched_grid = _fake_fetch


def _patch_frame_fetch_fails(frame: RainViewerFrame, fetch_log: list[str] | None = None):
    """Patch _fetch_stitched_grid to raise without populating the cache.

    Simulates a rate-limit, timeout, or other fetch failure. The frame's
    _cached_grid stays None. The coordinator's _fetch_frame() swallows the
    exception and logs a warning.
    """
    async def _fake_fetch(bounds, width, height, session):
        if fetch_log is not None:
            fetch_log.append(frame.path)
        raise RuntimeError("simulated fetch failure")

    frame._fetch_stitched_grid = _fake_fetch


_EMPTY_RESULT = DetectionResult(
    rain_incoming=False,
    arrival_time=None,
    confidence=Confidence.UNAVAILABLE,
    frame_count=0,
)


def _network_patches(coordinator, frames):
    """Return the 3 context managers needed to isolate the coordinator from the network."""
    return (
        patch.object(coordinator._provider, "get_frames", new=AsyncMock(return_value=frames)),
        patch(
            "custom_components.rain_incoming.coordinator.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.rain_incoming.coordinator.detect",
            return_value=_EMPTY_RESULT,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1: empty cache -> all frames fetched
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_poll_fetches_all_frames(hass: HomeAssistant):
    """On the first poll the cache is empty so all frames must be fetched."""
    fetch_log: list[str] = []
    frames = [_make_fresh_frame(f"/v2/radar/frame{i}", ts=i) for i in range(8)]
    for f in frames:
        _patch_frame_fetch_succeeds(f, fetch_log)

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    with patch.object(
        coordinator._provider, "get_frames", new=AsyncMock(return_value=frames)
    ), patch(
        "custom_components.rain_incoming.coordinator.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.rain_incoming.coordinator.detect",
        return_value=_EMPTY_RESULT,
    ):
        await coordinator._async_update_data()

    # All 8 paths must have been fetched
    assert set(fetch_log) == {f.path for f in frames}, (
        f"Expected all 8 frames to be fetched. Fetched: {fetch_log}"
    )
    assert len(fetch_log) == 8, (
        f"Expected exactly 8 fetches, got {len(fetch_log)}: {fetch_log}"
    )

    # All 8 paths must be in the cache
    assert set(coordinator._frame_cache.keys()) == {f.path for f in frames}, (
        f"Cache should hold all 8 frames after first poll. Keys: {list(coordinator._frame_cache.keys())}"
    )

    # Each cached frame must have a populated grid
    for cached in coordinator._frame_cache.values():
        assert cached._cached_grid is not None, (
            f"Frame {cached.path} should have a populated grid after fetch"
        )


# ---------------------------------------------------------------------------
# Scenario 2: second poll reuses unchanged frames, only fetches new ones
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_poll_reuses_unchanged_frames(hass: HomeAssistant):
    """Second poll with 6 unchanged + 2 new paths only fetches the 2 new ones."""
    old_paths = [f"/v2/radar/frame{i}" for i in range(6)]
    new_paths = [f"/v2/radar/frame{i}" for i in range(6, 8)]

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    # --- Poll 1: all 6 frames, all succeed ---
    poll1_log: list[str] = []
    poll1_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(old_paths)]
    for f in poll1_frames:
        _patch_frame_fetch_succeeds(f, poll1_log)

    pg1, ps1, pd1 = _network_patches(coordinator, poll1_frames)
    with pg1, ps1, pd1:
        await coordinator._async_update_data()

    assert set(coordinator._frame_cache.keys()) == set(old_paths), (
        "Cache after poll 1 should hold the 6 old frames"
    )

    # --- Poll 2: manifest = 6 old + 2 new ---
    # Track ALL fetches on poll 2 - both old (should not be called) and new (must be called)
    poll2_log: list[str] = []
    poll2_frames = (
        [_make_fresh_frame(p, ts=i) for i, p in enumerate(old_paths)]
        + [_make_fresh_frame(p, ts=i + 6) for i, p in enumerate(new_paths)]
    )
    for f in poll2_frames:
        _patch_frame_fetch_succeeds(f, poll2_log)

    pg2, ps2, pd2 = _network_patches(coordinator, poll2_frames)
    with pg2, ps2, pd2:
        await coordinator._async_update_data()

    # Only the 2 new frames should have been fetched
    assert set(poll2_log) == set(new_paths), (
        f"Expected only new frames to be fetched on poll 2.\n"
        f"Fetched: {poll2_log}\n"
        f"New paths: {new_paths}"
    )

    # Cache should contain all 8 (6 old + 2 new)
    assert set(coordinator._frame_cache.keys()) == set(old_paths) | set(new_paths), (
        "Cache after poll 2 should hold all 8 frames"
    )


# ---------------------------------------------------------------------------
# Scenario 3: eviction - stale paths removed from cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eviction_removes_frames_not_in_current_manifest(hass: HomeAssistant):
    """Frames from previous polls that are no longer in the manifest are evicted."""
    stale_paths = [f"/v2/radar/stale{i}" for i in range(3)]
    current_paths = [f"/v2/radar/current{i}" for i in range(8)]
    overlap_paths = current_paths[:3]  # 3 paths appear in both stale cache and manifest

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    # --- Poll 1: seed the cache with 3 stale + 3 overlap frames ---
    poll1_frames = [
        _make_fresh_frame(p, ts=i)
        for i, p in enumerate(stale_paths + overlap_paths)
    ]
    for f in poll1_frames:
        _patch_frame_fetch_succeeds(f)

    pg1, ps1, pd1 = _network_patches(coordinator, poll1_frames)
    with pg1, ps1, pd1:
        await coordinator._async_update_data()

    assert len(coordinator._frame_cache) == 6, (
        "Cache after poll 1 should have 6 frames (3 stale + 3 overlap)"
    )

    # --- Poll 2: manifest contains only current paths (stale paths gone) ---
    poll2_log: list[str] = []
    poll2_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(current_paths)]
    for f in poll2_frames:
        _patch_frame_fetch_succeeds(f, poll2_log)

    pg2, ps2, pd2 = _network_patches(coordinator, poll2_frames)
    with pg2, ps2, pd2:
        await coordinator._async_update_data()

    # Stale frames must have been evicted
    for stale in stale_paths:
        assert stale not in coordinator._frame_cache, (
            f"Stale frame {stale} should have been evicted from cache"
        )

    # Cache must contain exactly the current manifest paths
    assert set(coordinator._frame_cache.keys()) == set(current_paths), (
        f"Cache should hold exactly current manifest paths.\n"
        f"Extra: {set(coordinator._frame_cache.keys()) - set(current_paths)}\n"
        f"Missing: {set(current_paths) - set(coordinator._frame_cache.keys())}"
    )

    # 3 overlap frames were in cache and should not have been re-fetched
    assert set(poll2_log).isdisjoint(set(overlap_paths)), (
        f"Overlap frames should have been reused from cache, not re-fetched.\n"
        f"Unexpectedly fetched: {set(poll2_log) & set(overlap_paths)}"
    )

    # 5 new frames (current_paths[3:]) must have been fetched
    assert set(poll2_log) == set(current_paths[3:]), (
        f"Expected 5 new frames to be fetched. Fetched: {poll2_log}"
    )


# ---------------------------------------------------------------------------
# Scenario 4: resolved frame order matches manifest order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolved_frame_order_matches_manifest(hass: HomeAssistant):
    """Cached + uncached frames interleaved must be resolved in manifest order.

    The coordinator stores frame_paths in the order frames are resolved.
    That order must match the manifest (time order) regardless of which
    frames came from cache and which were freshly fetched.
    """
    paths = [f"/v2/radar/frame{i}" for i in range(8)]
    cached_indices = [2, 4, 6]

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    # --- Poll 1: seed cache for indices 2, 4, 6 ---
    poll1_frames = [_make_fresh_frame(paths[i], ts=i) for i in cached_indices]
    for f in poll1_frames:
        _patch_frame_fetch_succeeds(f)

    pg1, ps1, pd1 = _network_patches(coordinator, poll1_frames)
    with pg1, ps1, pd1:
        await coordinator._async_update_data()

    # --- Poll 2: full 8-frame manifest with cached frames interleaved ---
    poll2_frames = [_make_fresh_frame(p, ts=j) for j, p in enumerate(paths)]
    for f in poll2_frames:
        _patch_frame_fetch_succeeds(f)

    pg2, ps2, pd2 = _network_patches(coordinator, poll2_frames)
    with pg2, ps2, pd2:
        await coordinator._async_update_data()

    # frame_paths is set in _async_update_data from the resolved frames list
    assert coordinator.frame_paths == paths, (
        f"Resolved frame order must match manifest order.\n"
        f"Expected: {paths}\n"
        f"Got:      {coordinator.frame_paths}"
    )


# ---------------------------------------------------------------------------
# Scenario 5: _fetch_stitched_grid not called for cached frames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_not_called_for_cached_frames(hass: HomeAssistant):
    """Fetching must not be attempted for frames already in the cache.

    Uses a tracking list to detect whether _fetch_stitched_grid is called
    for cached frames. We do NOT raise from the patched method because the
    coordinator's _fetch_frame() swallows all exceptions - raising would
    produce a warning log but the test would still pass.
    """
    paths = [f"/v2/radar/frame{i}" for i in range(4)]
    cached_paths = paths[:2]
    new_paths = paths[2:]

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    # --- Poll 1: seed cache for the first 2 frames ---
    poll1_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(cached_paths)]
    for f in poll1_frames:
        _patch_frame_fetch_succeeds(f)

    pg1, ps1, pd1 = _network_patches(coordinator, poll1_frames)
    with pg1, ps1, pd1:
        await coordinator._async_update_data()

    # --- Poll 2: 4-frame manifest, first 2 from cache ---
    # Track ALL fetch calls in poll 2 - both cached and new
    poll2_log: list[str] = []
    poll2_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(paths)]
    for f in poll2_frames:
        _patch_frame_fetch_succeeds(f, poll2_log)

    pg2, ps2, pd2 = _network_patches(coordinator, poll2_frames)
    with pg2, ps2, pd2:
        await coordinator._async_update_data()

    # Only new frames (not in cache) should have been fetched
    assert set(poll2_log) == set(new_paths), (
        f"_fetch_stitched_grid should only be called for new frames.\n"
        f"Called for: {poll2_log}\n"
        f"New paths:  {new_paths}"
    )
    assert set(poll2_log).isdisjoint(set(cached_paths)), (
        f"_fetch_stitched_grid must NOT be called for cached frames.\n"
        f"Called for cached: {set(poll2_log) & set(cached_paths)}"
    )


# ---------------------------------------------------------------------------
# Scenario 6: failed fetch is not cached - re-fetched on next poll
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_fetch_not_cached(hass: HomeAssistant):
    """Frames whose fetch failed must not be cached, and must be re-fetched next poll.

    If a frame is rate-limited or times out, its _cached_grid stays None.
    Caching it would mean we never retry - the broken data persists forever.
    On the SECOND poll, the failed frame must appear in to_fetch again, not
    be treated as a cache hit.
    """
    paths = [f"/v2/radar/frame{i}" for i in range(4)]

    entry = _make_entry()
    coordinator = RainDetectorCoordinator(hass, entry)

    # --- Poll 1: frames 0-2 succeed, frame 3 fails ---
    poll1_log: list[str] = []
    poll1_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(paths)]
    for f in poll1_frames[:3]:
        _patch_frame_fetch_succeeds(f, poll1_log)
    _patch_frame_fetch_fails(poll1_frames[3], poll1_log)

    pg1, ps1, pd1 = _network_patches(coordinator, poll1_frames)
    with pg1, ps1, pd1:
        # Should succeed - 3 of 4 frames have data
        await coordinator._async_update_data()

    # All 4 frames were attempted (3 succeed, 1 fails)
    assert set(poll1_log) == set(paths), (
        f"All 4 frames should have been attempted on poll 1. Attempted: {poll1_log}"
    )

    # Frame 3 failed - must not be in cache
    assert paths[3] not in coordinator._frame_cache, (
        f"Frame with failed fetch must not be cached. Cache keys: {list(coordinator._frame_cache.keys())}"
    )

    # Frames 0-2 succeeded - must be in cache
    for p in paths[:3]:
        assert p in coordinator._frame_cache, (
            f"Successfully fetched frame {p} must be in cache"
        )

    # --- Poll 2: same manifest, all succeed this time ---
    poll2_log: list[str] = []
    poll2_frames = [_make_fresh_frame(p, ts=i) for i, p in enumerate(paths)]
    for f in poll2_frames:
        _patch_frame_fetch_succeeds(f, poll2_log)

    pg2, ps2, pd2 = _network_patches(coordinator, poll2_frames)
    with pg2, ps2, pd2:
        await coordinator._async_update_data()

    # Frame 3 was not in cache - it must have been re-fetched
    assert paths[3] in poll2_log, (
        "Previously failed frame must be re-fetched on the second poll"
    )

    # Frames 0-2 were cached - they must NOT have been re-fetched
    assert set(poll2_log).isdisjoint(set(paths[:3])), (
        f"Previously successful frames must be reused from cache, not re-fetched.\n"
        f"Unexpectedly re-fetched: {set(poll2_log) & set(paths[:3])}"
    )

    # After poll 2, all 4 frames should now be cached
    assert set(coordinator._frame_cache.keys()) == set(paths), (
        "All 4 frames should be cached after poll 2 succeeds fully"
    )
