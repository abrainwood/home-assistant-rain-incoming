"""
Golden-data radar scenarios for use in unit, integration, and E2E tests.

Each factory function returns a list of mock RadarFrame objects simulating a
specific weather situation at the Terry Hills test location. The frames are
10-minute intervals, oldest first, matching the RainViewer API cadence.

Scenarios:
- approaching_from_west: rain cell moving east toward the location (should trigger)
- rain_overhead:         rain cell already directly at the location (should trigger, arrival=now)
- rain_receding:         rain cell moving west away from the location (should not trigger)
- no_rain:               no precipitation in any frame (should not trigger)
- beyond_lookahead:      slow cell too far away to arrive within the lookahead window (should not trigger)
- parallel_miss:         cell moving east but far north - coherent motion, never intersects location (should not trigger)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np

from custom_components.incoming_rain.providers.base import BoundingBox, RadarFrame

# Standard test location: Terry Hills radar
LAT: float = -33.701
LON: float = 151.209

# 64x64 grid covering ±1.5° around the test location
# Location pixel: row=32, col=32 (centre of grid)
GRID_SIZE: int = 64
BOUNDS: BoundingBox = BoundingBox(
    lat_min=LAT - 1.5,
    lat_max=LAT + 1.5,
    lon_min=LON - 1.5,
    lon_max=LON + 1.5,
)

_BASE_TIME = datetime(2026, 4, 7, 10, 0, 0, tzinfo=timezone.utc)


def _frame(grid: np.ndarray, offset_minutes: int) -> RadarFrame:
    frame = MagicMock(spec=RadarFrame)
    frame.timestamp = _BASE_TIME + timedelta(minutes=offset_minutes)
    frame.get_intensity_grid.return_value = grid.astype(np.float32)
    frame.get_intensity_at.return_value = float(grid.mean())
    return frame


def approaching_from_west() -> list[RadarFrame]:
    """
    3x3 rain cell at row 30-32, moving east by 8 pixels per frame.
    Starts at col 10, reaches the location (col 32) within the lookahead window.
    Expected: rain_incoming=True, arrival_time > last frame timestamp.
    """
    frames = []
    for i, col in enumerate([10, 18, 26]):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[30:33, col:col + 3] = 0.8
        frames.append(_frame(grid, -20 + i * 10))
    return frames


def rain_overhead() -> list[RadarFrame]:
    """
    3x3 rain cell centred at the location pixel (row 32, col 32), stationary across all frames.
    Expected: rain_incoming=True, arrival_time == last frame timestamp (rain is here now).
    """
    frames = []
    for i in range(3):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[31:34, 31:34] = 0.8  # centroid at (32, 32) = location
        frames.append(_frame(grid, -20 + i * 10))
    return frames


def rain_receding() -> list[RadarFrame]:
    """
    3x3 rain cell at row 30-32, moving west away from the location.
    Starts near col 32 and moves toward col 16 - already past the location heading away.
    Expected: rain_incoming=False.
    """
    frames = []
    for i, col in enumerate([32, 24, 16]):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[30:33, col:col + 3] = 0.8
        frames.append(_frame(grid, -20 + i * 10))
    return frames


def no_rain() -> list[RadarFrame]:
    """
    Three frames with zero precipitation everywhere.
    Expected: rain_incoming=False, arrival_time=None.
    """
    frames = []
    for i in range(3):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        frames.append(_frame(grid, -20 + i * 10))
    return frames


def beyond_lookahead() -> list[RadarFrame]:
    """
    3x3 rain cell at row 30-32, barely moving east (1 px/frame from col 5).
    The cell would take hours to reach the location - well beyond any reasonable lookahead.
    Expected: rain_incoming=False when lookahead_seconds is short (e.g. 300s).
    """
    frames = []
    for i, col in enumerate([5, 6, 7]):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[30:33, col:col + 3] = 0.8
        frames.append(_frame(grid, -20 + i * 10))
    return frames


def parallel_miss() -> list[RadarFrame]:
    """
    3x3 rain cell at rows 5-7 (far north of the location at row 32), moving east.
    Motion is directionally coherent but the trajectory never intersects the proximity circle.
    Expected: rain_incoming=False, arrival_time=None.
    """
    frames = []
    for i, col in enumerate([20, 28, 36]):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[5:8, col:col + 3] = 0.8
        frames.append(_frame(grid, -20 + i * 10))
    return frames
