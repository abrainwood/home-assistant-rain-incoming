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

    # -- mutmut survivors below --

    def test_one_empty_dict_returns_no_matches(self):
        """Mutant #15: `or` -> `and` in the empty-guard clause.
        EQUIVALENT MUTANT - the nested loop over an empty dict produces no
        candidates regardless, so the guard is an optimization not a behavior.
        This test documents the expected behavior but cannot kill the mutant."""
        t0 = {1: (0.0, 0.0)}
        t1: dict[int, tuple[float, float]] = {}
        assert match_cells_across_frames(t0, t1, max_distance=10.0) == []
        assert match_cells_across_frames({}, {1: (0.0, 0.0)}, max_distance=10.0) == []

    def test_distance_calculation_uses_squared_components(self):
        """Kill mutants #18-23: `** 2` -> `* 2` or `** 3` in distance calc.
        Place cells 5px apart on one axis. Euclidean = 5.0.
        With `* 2` the distance would be sqrt(10) ~ 3.16 (wrong match).
        With `** 3` the distance would be sqrt(125) ~ 11.18 (wrong reject).
        Use max_distance=5.0 so only the correct `** 2` produces a match."""
        t0 = {1: (0.0, 0.0)}
        t1 = {1: (5.0, 0.0)}
        # Exact Euclidean distance = 5.0, so max_distance=5.0 should match (<=)
        matches = match_cells_across_frames(t0, t1, max_distance=5.0)
        assert len(matches) == 1, "5px cell must match at max_distance=5.0"

        # Now test the column axis too (kills the (c1-c0) mutations)
        t0_col = {1: (0.0, 0.0)}
        t1_col = {1: (0.0, 5.0)}
        matches_col = match_cells_across_frames(t0_col, t1_col, max_distance=5.0)
        assert len(matches_col) == 1, "5px column cell must match at max_distance=5.0"

        # And a case that should NOT match: just beyond 5.0
        t1_far = {1: (5.1, 0.0)}
        matches_far = match_cells_across_frames(t0, t1_far, max_distance=5.0)
        assert matches_far == [], "5.1px cell must not match at max_distance=5.0"

    def test_distance_column_axis_tight_threshold(self):
        """Kill mutant #22: `(c1-c0) ** 2` -> `(c1-c0) * 2` in column distance.
        Cell 4px apart on column axis only. Correct: sqrt(16) = 4.0.
        With `* 2`: sqrt(8) = 2.83. Use max_distance=3.5 so the correct distance
        exceeds it (no match) but the mutated distance fits (wrong match)."""
        t0 = {1: (0.0, 0.0)}
        t1 = {1: (0.0, 4.0)}
        matches = match_cells_across_frames(t0, t1, max_distance=3.5)
        assert matches == [], "4px column distance must exceed max_distance=3.5"

    def test_boundary_distance_is_inclusive(self):
        """Kill mutant #25: `<=` -> `<` in distance comparison.
        Cell at exactly max_distance must be included."""
        t0 = {1: (0.0, 0.0)}
        t1 = {1: (3.0, 4.0)}  # distance = exactly 5.0
        matches = match_cells_across_frames(t0, t1, max_distance=5.0)
        assert len(matches) == 1, "Cell at exactly max_distance must match (<=, not <)"

    def test_continue_past_claimed_cells_to_find_later_matches(self):
        """Kill mutant #32: `continue` -> `break` in the matching loop.
        Three cells: the closest pair claims t1[10], then the loop must
        CONTINUE past the claimed entry to find the next valid match for t0[2].
        With `break`, the loop exits early and t0[2] never gets matched."""
        t0 = {1: (0.0, 0.0), 2: (10.0, 0.0)}
        t1 = {10: (0.5, 0.0), 20: (10.5, 0.0)}
        # Sorted candidates: (0.5, 1, 10), (0.5, 2, 20), (10.5, 1, 20), (9.5, 2, 10)
        # Actually let me recalculate...
        # (1->10): dist = 0.5  (1->20): dist = 10.5  (2->10): dist = 9.5  (2->20): dist = 0.5
        # Sorted: (0.5, 1, 10), (0.5, 2, 20), (9.5, 2, 10), (10.5, 1, 20)
        # First match: (1, 10). Then (2, 20) is next - both unclaimed, match.
        # With break: after (1, 10) matches, loop sees (2, 20) which is unclaimed... hmm.
        # Need a case where a claimed entry appears BETWEEN two valid matches.
        # Let me restructure: three t0 cells, two t1 cells, with an intermediate
        # distance that blocks.
        t0 = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (20.0, 0.0)}
        t1 = {10: (0.5, 0.0), 20: (20.5, 0.0)}
        # Candidates sorted by distance:
        # (0.5, 1, 10), (0.5, 2, 10)... wait, t1[10] is at 0.5 from t0[1] and 0.5 from t0[2]
        # Actually: dist(1,10)=0.5, dist(2,10)=0.5, dist(3,20)=0.5, dist(1,20)=20.5, dist(2,20)=19.5, dist(3,10)=19.5
        # Sorted: (0.5, 1, 10), (0.5, 2, 10), (0.5, 3, 20), (19.5, 2, 20), (19.5, 3, 10), (20.5, 1, 20)
        # Match (1, 10). Next: (2, 10) - t1[10] claimed, CONTINUE. Next: (3, 20) - unclaimed, match.
        # With BREAK at (2, 10): loop exits, (3, 20) never checked. Only 1 match instead of 2.
        matches = match_cells_across_frames(t0, t1, max_distance=25.0)
        matched_t0_labels = sorted(m[0] for m in matches)
        assert len(matches) == 2, (
            f"Expected 2 matches (skipping claimed t1 cell), got {len(matches)}: {matches}"
        )
        assert 1 in matched_t0_labels
        assert 3 in matched_t0_labels


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

    # -- mutmut survivors below --

    def test_empty_velocities_returns_false(self):
        """Kill mutant #44: `return False` -> `return True` for empty input."""
        assert not is_directionally_coherent([], max_angular_variance=0.5)

    def test_circular_variance_math_coherent_case(self):
        """Kill mutants #62, #65: `** 2` -> `** 3` in R calculation.

        Two velocities at ~45 degrees apart. Correct circular variance ~ 0.076.
        With `** 3`, R changes enough to flip the result.

        Velocities: (1, 0) = 90 deg and (0.7, 0.7) ~ 45 deg.
        Mean resultant R ~ 0.924, circular_variance ~ 0.076.
        With max_angular_variance = 0.1, this should be coherent."""
        velocities = [(1.0, 0.0), (0.7, 0.7)]
        assert is_directionally_coherent(velocities, max_angular_variance=0.1), (
            "Two velocities ~45 degrees apart should be coherent at variance=0.1"
        )

    def test_circular_variance_incoherent_rejects_inflated_r(self):
        """Kill mutants #57, #61: `/ len` -> `* len` and `** 2` -> `* 2` in R calc.

        Two velocities ~117 degrees apart. Correct variance ~ 0.475 (incoherent).
        Mutations that inflate sin_mean (multiplying by len instead of dividing,
        or using `* 2` instead of `** 2`) artificially boost R, making variance
        negative - wrongly appearing coherent.

        With max_angular_variance = 0.4, this MUST be incoherent."""
        velocities = [(0.0, 1.0), (1.0, -0.5)]
        assert not is_directionally_coherent(velocities, max_angular_variance=0.4), (
            "Two velocities ~117 degrees apart must be incoherent at variance=0.4"
        )

    def test_circular_variance_boundary_rejects_at_threshold(self):
        """Kill mutant #70: `<=` -> `<` in threshold comparison.

        Construct velocities where circular_variance == max_angular_variance exactly.
        The `<=` operator means this should return True; `<` would return False.

        Two velocities pointing in directions that produce a known circular variance.
        We compute the exact variance and use it as the threshold."""
        # Two velocities: east (0, 1) and slightly north-of-east (0.2, 1.0)
        velocities = [(0.0, 1.0), (0.2, 1.0)]

        # Compute expected circular variance to use as exact threshold
        import math
        angles = [math.atan2(vy, vx) for vy, vx in velocities]
        sin_mean = sum(math.sin(a) for a in angles) / len(angles)
        cos_mean = sum(math.cos(a) for a in angles) / len(angles)
        r = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
        exact_variance = 1.0 - r

        # At exactly the threshold, <= should return True
        assert is_directionally_coherent(velocities, max_angular_variance=exact_variance), (
            f"Circular variance {exact_variance:.6f} at exact threshold must be coherent (<=)"
        )
