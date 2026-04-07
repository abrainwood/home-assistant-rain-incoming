import numpy as np
import pytest

from custom_components.incoming_rain.radar.filters import (
    filter_by_area,
    filter_by_temporal_persistence,
    threshold_intensity,
)


class TestThresholdIntensity:
    def test_zeros_returns_all_false(self):
        grid = np.zeros((4, 4), dtype=float)
        result = threshold_intensity(grid, threshold=0.1)
        assert not result.any()

    def test_values_above_threshold_return_true(self):
        grid = np.array([[0.0, 0.5], [0.2, 0.8]])
        result = threshold_intensity(grid, threshold=0.3)
        expected = np.array([[False, True], [False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_exact_threshold_is_included(self):
        grid = np.array([[0.3]])
        result = threshold_intensity(grid, threshold=0.3)
        assert result[0, 0]

    def test_all_above_threshold_returns_all_true(self):
        grid = np.ones((3, 3), dtype=float)
        result = threshold_intensity(grid, threshold=0.5)
        assert result.all()


class TestFilterByArea:
    def test_empty_grid_returns_empty(self):
        grid = np.zeros((4, 4), dtype=bool)
        result = filter_by_area(grid, min_area=2)
        assert not result.any()

    def test_large_component_is_kept(self):
        grid = np.zeros((5, 5), dtype=bool)
        grid[1:4, 1:4] = True  # 3x3 = 9 pixels
        result = filter_by_area(grid, min_area=5)
        assert result[2, 2]

    def test_small_component_is_removed(self):
        grid = np.zeros((5, 5), dtype=bool)
        grid[0, 0] = True  # 1 pixel
        result = filter_by_area(grid, min_area=2)
        assert not result.any()

    def test_large_and_small_components_separated(self):
        grid = np.zeros((8, 8), dtype=bool)
        grid[0, 0] = True       # 1 pixel - small
        grid[3:6, 3:6] = True   # 9 pixels - large
        result = filter_by_area(grid, min_area=5)
        assert not result[0, 0]
        assert result[4, 4]

    def test_exactly_min_area_is_kept(self):
        grid = np.zeros((5, 5), dtype=bool)
        grid[1, 1] = True
        grid[1, 2] = True  # 2-pixel component
        result = filter_by_area(grid, min_area=2)
        assert result[1, 1]
        assert result[1, 2]

    def test_min_area_one_keeps_all(self):
        grid = np.zeros((4, 4), dtype=bool)
        grid[0, 0] = True
        grid[3, 3] = True
        result = filter_by_area(grid, min_area=1)
        assert result[0, 0]
        assert result[3, 3]


class TestFilterByTemporalPersistence:
    def test_pixel_present_in_all_frames_is_kept(self):
        frames = [np.ones((3, 3), dtype=bool)] * 3
        result = filter_by_temporal_persistence(frames, min_frames=3)
        assert result.all()

    def test_pixel_present_in_no_frames_is_removed(self):
        frames = [np.zeros((3, 3), dtype=bool)] * 3
        result = filter_by_temporal_persistence(frames, min_frames=2)
        assert not result.any()

    def test_pixel_meeting_threshold_and_in_last_frame_is_kept(self):
        f0 = np.zeros((3, 3), dtype=bool)
        f1 = np.zeros((3, 3), dtype=bool)
        f2 = np.zeros((3, 3), dtype=bool)
        f0[1, 1] = True
        f1[1, 1] = True
        f2[1, 1] = True  # present in all 3, including the last
        frames = [f0, f1, f2]
        result = filter_by_temporal_persistence(frames, min_frames=2)
        assert result[1, 1]

    def test_pixel_below_threshold_is_removed(self):
        f0 = np.zeros((3, 3), dtype=bool)
        f1 = np.zeros((3, 3), dtype=bool)
        f2 = np.zeros((3, 3), dtype=bool)
        f0[1, 1] = True  # present in only 1 of 3
        frames = [f0, f1, f2]
        result = filter_by_temporal_persistence(frames, min_frames=2)
        assert not result[1, 1]

    def test_pixel_not_in_last_frame_is_excluded(self):
        """Pixel present in early frames but absent in last frame is not reported."""
        f0 = np.zeros((3, 3), dtype=bool)
        f1 = np.zeros((3, 3), dtype=bool)
        f2 = np.zeros((3, 3), dtype=bool)
        f0[1, 1] = True
        f1[1, 1] = True  # disappeared by last frame
        frames = [f0, f1, f2]
        result = filter_by_temporal_persistence(frames, min_frames=2)
        assert not result[1, 1]  # not in last frame, so not reported

    def test_returns_intersection_with_last_frame(self):
        """Pixels not in the last frame are excluded even if historically persistent."""
        early = np.zeros((3, 3), dtype=bool)
        early[0, 0] = True
        early[1, 1] = True
        late = np.zeros((3, 3), dtype=bool)
        late[1, 1] = True  # [0,0] gone, [1,1] persists
        frames = [early, early, late]
        result = filter_by_temporal_persistence(frames, min_frames=2)
        assert not result[0, 0]  # not in last frame
        assert result[1, 1]      # persistent and in last frame

    def test_single_frame_with_min_one(self):
        frame = np.zeros((3, 3), dtype=bool)
        frame[2, 2] = True
        result = filter_by_temporal_persistence([frame], min_frames=1)
        assert result[2, 2]
