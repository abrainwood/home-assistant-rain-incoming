from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from custom_components.incoming_rain.radar.composite import (
    calculate_map_zoom,
    draw_crosshair,
    draw_range_rings,
    filter_precipitation_pixels,
    km_per_pixel,
)


class TestKmPerPixel:
    def test_equator_zoom_0(self):
        # At zoom 0 the whole world fits in 256 pixels
        result = km_per_pixel(0.0, 0)
        expected = 40075.0 / 256
        assert result == pytest.approx(expected, rel=1e-6)

    def test_higher_zoom_halves_distance(self):
        z0 = km_per_pixel(0.0, 0)
        z1 = km_per_pixel(0.0, 1)
        assert z1 == pytest.approx(z0 / 2, rel=1e-6)

    def test_latitude_reduces_km(self):
        equator = km_per_pixel(0.0, 7)
        mid_lat = km_per_pixel(45.0, 7)
        assert mid_lat < equator
        assert mid_lat == pytest.approx(equator * math.cos(math.radians(45.0)), rel=1e-6)


class TestCalculateMapZoom:
    def test_small_radius_gives_higher_zoom(self):
        z_small = calculate_map_zoom(-33.7, 64, 640)
        z_large = calculate_map_zoom(-33.7, 256, 640)
        assert z_small > z_large

    def test_default_radius_128_reasonable_zoom(self):
        z = calculate_map_zoom(-33.7, 128, 640)
        # Should be somewhere between 7 and 12 for a 128km radius
        assert 7 <= z <= 12

    def test_returns_int(self):
        z = calculate_map_zoom(0.0, 128, 640)
        assert isinstance(z, int)


class TestFilterPrecipitationPixels:
    def test_transparent_pixels_stay_transparent(self):
        # A fully transparent image should remain transparent
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = filter_precipitation_pixels(img)
        assert result.shape == (10, 10, 4)
        assert (result[:, :, 3] == 0).all()

    def test_precipitation_colour_preserved(self):
        # Create an image with a known precipitation colour
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        # Use the first precip colour: (0, 91, 142) with full alpha
        img[5, 5] = [0, 91, 142, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 255  # alpha preserved

    def test_non_precip_colour_removed(self):
        # Create a pixel with a non-precipitation colour (land mask)
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        # Pure white with full alpha - not a precip colour
        img[5, 5] = [255, 255, 255, 255]
        result = filter_precipitation_pixels(img)
        assert result[5, 5, 3] == 0  # should be made transparent


class TestDrawCrosshair:
    def test_draws_red_at_circle_edge(self):
        from custom_components.incoming_rain.radar.composite import _CROSSHAIR_RADIUS
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        draw_crosshair(img, 50, 50)
        pixels = np.array(img)
        # The circle edge should have red pixels
        edge_pixel = pixels[50 - _CROSSHAIR_RADIUS, 50]
        assert edge_pixel[0] > 100  # red channel present from circle outline

    def test_crosshair_lines_extend_from_centre(self):
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 255))
        draw_crosshair(img, 100, 100)
        pixels = np.array(img)
        # Check that some pixels along the horizontal line are red
        # Offset from centre to avoid the circle area
        line_pixel = pixels[100, 100 + 20]
        assert line_pixel[0] > 100  # red channel present


class TestDrawRangeRings:
    def test_draws_at_expected_radius(self):
        size = 200
        img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
        centre_x, centre_y = size // 2, size // 2
        radius_pixels = 50
        draw_range_rings(img, centre_x, centre_y, radius_pixels)
        pixels = np.array(img)
        # Check a point on the full-radius ring (top of circle)
        ring_pixel = pixels[centre_y - radius_pixels, centre_x]
        # Should have some white/grey marking
        assert ring_pixel[0] > 20 or ring_pixel[1] > 20 or ring_pixel[2] > 20
        # Verify it differs from the pure black background
        assert not (ring_pixel[0] == 0 and ring_pixel[1] == 0 and ring_pixel[2] == 0 and ring_pixel[3] == 255)
