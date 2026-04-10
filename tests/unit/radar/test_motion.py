import math

import numpy as np
import pytest

from custom_components.rain_incoming.radar.motion import (
    estimate_velocity,
    extract_cell_centroids,
    is_directionally_coherent,
    match_cells_across_frames,
)


class TestExtractCellCentroids:
    def test_empty_labeled_grid_returns_empty(self):
        labeled = np.zeros((5, 5), dtype=int)
        result = extract_cell_centroids(labeled)
        assert result == {}

    def test_single_pixel_component(self):
        labeled = np.zeros((5, 5), dtype=int)
        labeled[2, 3] = 1
        result = extract_cell_centroids(labeled)
        assert result == {1: (2.0, 3.0)}

    def test_square_component_centroid_is_centre(self):
        labeled = np.zeros((6, 6), dtype=int)
        labeled[2:4, 2:4] = 1  # 2x2 block at rows 2-3, cols 2-3
        result = extract_cell_centroids(labeled)
        assert result[1] == pytest.approx((2.5, 2.5))

    def test_multiple_components_returned(self):
        labeled = np.zeros((8, 8), dtype=int)
        labeled[1, 1] = 1
        labeled[6, 6] = 2
        result = extract_cell_centroids(labeled)
        assert 1 in result
        assert 2 in result
        assert result[1] == (1.0, 1.0)
        assert result[2] == (6.0, 6.0)

    def test_label_zero_is_ignored(self):
        labeled = np.zeros((4, 4), dtype=int)
        labeled[0, 0] = 0  # background - must not appear
        labeled[2, 2] = 1
        result = extract_cell_centroids(labeled)
        assert 0 not in result
        assert 1 in result


class TestMatchCellsAcrossFrames:
    def test_single_cell_matched_within_distance(self):
        t0 = {1: (2.0, 2.0)}
        t1 = {1: (3.0, 3.0)}  # moved sqrt(2) ≈ 1.41 pixels
        matches = match_cells_across_frames(t0, t1, max_distance=3.0)
        assert len(matches) == 1
        assert matches[0] == (1, 1)

    def test_cell_too_far_is_not_matched(self):
        t0 = {1: (0.0, 0.0)}
        t1 = {1: (10.0, 10.0)}
        matches = match_cells_across_frames(t0, t1, max_distance=3.0)
        assert matches == []

    def test_nearest_cell_wins_when_multiple_candidates(self):
        t0 = {1: (5.0, 5.0)}
        t1 = {10: (5.5, 5.0), 20: (8.0, 8.0)}  # label 10 is closer
        matches = match_cells_across_frames(t0, t1, max_distance=5.0)
        assert len(matches) == 1
        assert matches[0] == (1, 10)

    def test_empty_frames_return_no_matches(self):
        assert match_cells_across_frames({}, {}, max_distance=5.0) == []
        assert match_cells_across_frames({1: (0.0, 0.0)}, {}, max_distance=5.0) == []
        assert match_cells_across_frames({}, {1: (0.0, 0.0)}, max_distance=5.0) == []

    def test_each_t1_cell_matched_at_most_once(self):
        # Two t0 cells competing for the same t1 cell - only closer one wins
        t0 = {1: (5.0, 5.0), 2: (5.1, 5.0)}  # label 2 is slightly closer to t1[1]
        t1 = {1: (5.2, 5.0)}
        matches = match_cells_across_frames(t0, t1, max_distance=5.0)
        t1_labels = [m[1] for m in matches]
        assert len(set(t1_labels)) == len(t1_labels), "t1 cell matched more than once"


class TestEstimateVelocity:
    def test_stationary_cell_has_zero_velocity(self):
        vy, vx = estimate_velocity((5.0, 5.0), (5.0, 5.0), time_delta_seconds=600)
        assert vy == pytest.approx(0.0)
        assert vx == pytest.approx(0.0)

    def test_rightward_motion(self):
        # Moved 12 pixels right in 600 seconds
        vy, vx = estimate_velocity((5.0, 0.0), (5.0, 12.0), time_delta_seconds=600)
        assert vy == pytest.approx(0.0)
        assert vx == pytest.approx(12.0 / 600)

    def test_diagonal_motion(self):
        vy, vx = estimate_velocity((0.0, 0.0), (3.0, 4.0), time_delta_seconds=100)
        assert vy == pytest.approx(3.0 / 100)
        assert vx == pytest.approx(4.0 / 100)

    def test_upward_motion_is_negative_vy(self):
        # Moving north = decreasing row index
        vy, vx = estimate_velocity((10.0, 5.0), (5.0, 5.0), time_delta_seconds=500)
        assert vy == pytest.approx(-5.0 / 500)
        assert vx == pytest.approx(0.0)


class TestIsDirectionallyCoherent:
    def test_identical_velocities_are_coherent(self):
        velocities = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        assert is_directionally_coherent(velocities, max_angular_variance=0.5)

    def test_opposite_velocities_are_not_coherent(self):
        velocities = [(0.0, 1.0), (0.0, -1.0)]
        assert not is_directionally_coherent(velocities, max_angular_variance=0.5)

    def test_perpendicular_velocities_are_not_coherent(self):
        velocities = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        assert not is_directionally_coherent(velocities, max_angular_variance=0.5)

    def test_small_direction_variation_is_coherent(self):
        # Slightly varying but all roughly eastward
        velocities = [(0.0, 1.0), (0.1, 1.0), (-0.1, 1.0)]
        assert is_directionally_coherent(velocities, max_angular_variance=0.5)

    def test_single_velocity_is_always_coherent(self):
        assert is_directionally_coherent([(1.0, 0.0)], max_angular_variance=0.1)

    def test_all_zero_velocities_returns_false(self):
        # Stationary noise has no direction - not coherent movement
        velocities = [(0.0, 0.0), (0.0, 0.0)]
        assert not is_directionally_coherent(velocities, max_angular_variance=0.5)
