"""Cartesian-to-polar geometry for CME front tracking.

The polar coordinate convention is deliberately explicit: solar position
angle is zero at solar north and increases through solar east.  A FITS image
does not necessarily display north at decreasing array row, so the mapping
uses pixel-space north and east unit vectors resolved by the input boundary,
not a screen-orientation assumption.

The initial synthetic adapter supplies a fixed center and apparent solar
radius.  Keeping this geometry separate from the detector makes it possible to
replace the adapter with per-frame WCS geometry later without changing the
front-likelihood or tracking interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import map_coordinates

from .input import SequenceGeometry


class PolarGeometryError(ValueError):
    """Raised when a requested polar grid is inconsistent with the image."""


@dataclass(frozen=True)
class PolarConfig:
    """Sampling choices for a SunCET polar image.

    Parameters are deliberately expressed in either pixels or solar radii;
    there is no cadence or velocity assumption in this layer.
    """

    position_angle_step_deg: float = 2.0
    radial_step_px: float = 1.0
    minimum_radius_rsun: float = 1.05
    maximum_radius_rsun: float | None = None
    interpolation_order: int = 1

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.position_angle_step_deg)
            or self.position_angle_step_deg <= 0
            or self.position_angle_step_deg > 360
        ):
            raise PolarGeometryError(
                "position_angle_step_deg must be finite and in (0, 360]."
            )
        angle_count = round(360.0 / self.position_angle_step_deg)
        if angle_count < 1 or not math.isclose(
            angle_count * self.position_angle_step_deg,
            360.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise PolarGeometryError(
                "position_angle_step_deg must divide 360 degrees exactly."
            )
        if not math.isfinite(self.radial_step_px) or self.radial_step_px <= 0:
            raise PolarGeometryError("radial_step_px must be finite and positive.")
        if (
            not math.isfinite(self.minimum_radius_rsun)
            or self.minimum_radius_rsun < 0
        ):
            raise PolarGeometryError(
                "minimum_radius_rsun must be finite and nonnegative."
            )
        if self.maximum_radius_rsun is not None:
            if (
                not math.isfinite(self.maximum_radius_rsun)
                or self.maximum_radius_rsun < self.minimum_radius_rsun
            ):
                raise PolarGeometryError(
                    "maximum_radius_rsun must be finite and no smaller than "
                    "minimum_radius_rsun."
                )
        if self.interpolation_order not in range(0, 6):
            raise PolarGeometryError("interpolation_order must be between 0 and 5.")


@dataclass(frozen=True)
class PolarGrid:
    """Precomputed sampling coordinates shared by every fixed-geometry frame."""

    position_angle_deg: np.ndarray
    radius_px: np.ndarray
    radius_rsun: np.ndarray
    sample_y_px: np.ndarray
    sample_x_px: np.ndarray
    valid_mask: np.ndarray
    image_shape_yx: tuple[int, int]
    interpolation_order: int

    def __post_init__(self) -> None:
        angle_count = self.position_angle_deg.size
        radius_count = self.radius_px.size
        expected = (angle_count, radius_count)
        for name, values in (
            ("sample_y_px", self.sample_y_px),
            ("sample_x_px", self.sample_x_px),
            ("valid_mask", self.valid_mask),
        ):
            if values.shape != expected:
                raise PolarGeometryError(
                    f"{name} has shape {values.shape}; expected {expected}."
                )
        if self.radius_rsun.shape != self.radius_px.shape:
            raise PolarGeometryError("Pixel and solar-radius axes must agree.")
        if self.position_angle_deg.ndim != 1 or self.radius_px.ndim != 1:
            raise PolarGeometryError("Polar coordinate axes must be one-dimensional.")
        if angle_count == 0 or radius_count == 0:
            raise PolarGeometryError("A polar grid must contain at least one sample.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(position_angle, radius)`` grid shape."""

        return self.valid_mask.shape

    @property
    def radial_step_px(self) -> float:
        """Return the constant radial sampling step in pixels."""

        if self.radius_px.size == 1:
            return math.nan
        return float(self.radius_px[1] - self.radius_px[0])

    @property
    def position_angle_step_deg(self) -> float:
        """Return the constant circular position-angle sampling step."""

        return 360.0 / float(self.position_angle_deg.size)


def _default_maximum_radius_rsun(geometry: SequenceGeometry) -> float:
    """Reach the most distant horizontal or vertical edge, not a corner."""

    height, width = geometry.image_shape_yx
    axial_edge_distances = (
        geometry.center_x_px,
        width - 1.0 - geometry.center_x_px,
        geometry.center_y_px,
        height - 1.0 - geometry.center_y_px,
    )
    return max(axial_edge_distances) / geometry.solar_radius_px


def build_polar_grid(
    geometry: SequenceGeometry,
    config: PolarConfig | None = None,
) -> PolarGrid:
    """Build a reusable fixed-geometry polar sampling grid.

    The returned validity mask retains the rectangular field-of-view boundary.
    In particular, large radii can remain available near the long image axis
    without pretending that every position angle has the same coverage.
    """

    if config is None:
        config = PolarConfig()

    maximum_radius_rsun = (
        _default_maximum_radius_rsun(geometry)
        if config.maximum_radius_rsun is None
        else config.maximum_radius_rsun
    )
    if maximum_radius_rsun < config.minimum_radius_rsun:
        raise PolarGeometryError(
            "The requested minimum radius lies outside the available image radius."
        )

    angle_count = round(360.0 / config.position_angle_step_deg)
    position_angle_deg = np.linspace(
        0.0,
        360.0,
        angle_count,
        endpoint=False,
        dtype=np.float64,
    )

    minimum_radius_px = config.minimum_radius_rsun * geometry.solar_radius_px
    maximum_radius_px = maximum_radius_rsun * geometry.solar_radius_px
    radius_count = (
        int(math.floor((maximum_radius_px - minimum_radius_px) / config.radial_step_px))
        + 1
    )
    radius_px = minimum_radius_px + np.arange(radius_count) * config.radial_step_px
    radius_rsun = radius_px / geometry.solar_radius_px

    angles_rad = np.deg2rad(position_angle_deg)[:, np.newaxis]
    radii = radius_px[np.newaxis, :]
    north_y, north_x = geometry.north_vector_yx
    east_y, east_x = geometry.east_vector_yx
    sample_y_px = geometry.center_y_px + radii * (
        np.cos(angles_rad) * north_y + np.sin(angles_rad) * east_y
    )
    sample_x_px = geometry.center_x_px + radii * (
        np.cos(angles_rad) * north_x + np.sin(angles_rad) * east_x
    )

    height, width = geometry.image_shape_yx
    valid_mask = (
        (sample_x_px >= 0.0)
        & (sample_x_px <= width - 1.0)
        & (sample_y_px >= 0.0)
        & (sample_y_px <= height - 1.0)
    )

    return PolarGrid(
        position_angle_deg=position_angle_deg,
        radius_px=np.asarray(radius_px, dtype=np.float64),
        radius_rsun=np.asarray(radius_rsun, dtype=np.float64),
        sample_y_px=np.asarray(sample_y_px, dtype=np.float64),
        sample_x_px=np.asarray(sample_x_px, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        image_shape_yx=geometry.image_shape_yx,
        interpolation_order=config.interpolation_order,
    )


def remap_image_to_polar(image: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Interpolate one Cartesian image onto ``(position_angle, radius)``.

    Invalid rectangular-FOV samples are represented by ``NaN`` and remain
    distinguishable from real zero-valued pixels throughout tracking.
    """

    values = np.asarray(image)
    if values.ndim != 2 or values.shape != grid.image_shape_yx:
        raise PolarGeometryError(
            f"Image has shape {values.shape}; expected {grid.image_shape_yx}."
        )
    if not np.issubdtype(values.dtype, np.number):
        raise PolarGeometryError("Image values must be numeric.")

    polar = map_coordinates(
        np.asarray(values, dtype=np.float64),
        (grid.sample_y_px, grid.sample_x_px),
        order=grid.interpolation_order,
        mode="constant",
        cval=np.nan,
        prefilter=grid.interpolation_order > 1,
    )
    polar[~grid.valid_mask] = np.nan
    return np.asarray(polar, dtype=np.float32)


def remap_sequence_to_polar(images: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Remap an in-memory ``(time, y, x)`` sequence using one cached grid."""

    values = np.asarray(images)
    if values.ndim != 3 or values.shape[1:] != grid.image_shape_yx:
        raise PolarGeometryError(
            "Images must have shape (time, y, x) matching the polar grid."
        )
    if values.shape[0] < 1:
        raise PolarGeometryError("At least one image is required.")

    polar = np.empty((values.shape[0], *grid.shape), dtype=np.float32)
    for frame_index, image in enumerate(values):
        polar[frame_index] = remap_image_to_polar(image, grid)
    return polar
