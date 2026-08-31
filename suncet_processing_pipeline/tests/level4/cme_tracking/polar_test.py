"""Tests for the explicit SunCET solar-position-angle remap."""

import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.input import SequenceGeometry
from suncet_processing_pipeline.level4.cme_tracking.polar import (
    PolarConfig,
    build_polar_grid,
    remap_image_to_polar,
)


def test_cardinal_position_angles_are_north_east_south_west():
    image = np.zeros((51, 51), dtype=np.float32)
    center = 25
    radius_px = 20
    image[center - radius_px, center] = 1.0  # north
    image[center, center - radius_px] = 2.0  # solar east (image left)
    image[center + radius_px, center] = 3.0  # south
    image[center, center + radius_px] = 4.0  # solar west (image right)
    geometry = SequenceGeometry(
        image_shape_yx=image.shape,
        center_x_px=float(center),
        center_y_px=float(center),
        solar_radius_px=10.0,
        pixel_scales_arcsec_xy=None,
        pixel_scale_arcsec=None,
        pixel_scale_anisotropy_fraction=None,
        north_vector_yx=(-1.0, 0.0),
        east_vector_yx=(0.0, -1.0),
        orientation_source="explicit_override",
    )
    grid = build_polar_grid(
        geometry,
        PolarConfig(
            position_angle_step_deg=90.0,
            radial_step_px=1.0,
            minimum_radius_rsun=2.0,
            maximum_radius_rsun=2.0,
            interpolation_order=1,
        ),
    )

    polar = remap_image_to_polar(image, grid)

    np.testing.assert_array_equal(grid.position_angle_deg, [0.0, 90.0, 180.0, 270.0])
    np.testing.assert_allclose(polar[:, 0], [1.0, 2.0, 3.0, 4.0], atol=1e-6)


def test_cardinal_angles_follow_explicit_fits_pixel_basis():
    """Positive CDELT2 can put solar north at increasing array row."""

    image = np.zeros((51, 51), dtype=np.float32)
    center = 25
    radius_px = 20
    image[center + radius_px, center] = 11.0  # north is down in this FITS WCS
    image[center, center - radius_px] = 22.0  # solar east remains image left
    geometry = SequenceGeometry(
        image_shape_yx=image.shape,
        center_x_px=float(center),
        center_y_px=float(center),
        solar_radius_px=10.0,
        pixel_scales_arcsec_xy=None,
        pixel_scale_arcsec=None,
        pixel_scale_anisotropy_fraction=None,
        north_vector_yx=(1.0, 0.0),
        east_vector_yx=(0.0, -1.0),
        orientation_source="fits_wcs",
    )
    grid = build_polar_grid(
        geometry,
        PolarConfig(
            position_angle_step_deg=90.0,
            radial_step_px=1.0,
            minimum_radius_rsun=2.0,
            maximum_radius_rsun=2.0,
        ),
    )

    polar = remap_image_to_polar(image, grid)

    assert polar[0, 0] == 11.0
    assert polar[1, 0] == 22.0
