"""Deterministic front evidence for the first SunCET CME baseline.

The score produced here is an interpretable detection statistic, not a
calibrated probability.  It combines three complementary signals in polar
images: excess above a robust temporal background, positive running
difference, and the outward-facing edge of that excess.  All normalizations
are robust and radius dependent so the steep coronal brightness gradient does
not dominate the outer field of view.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from .polar import PolarGrid


class LikelihoodError(ValueError):
    """Raised when polar images or evidence parameters are invalid."""


@dataclass(frozen=True)
class EvidenceConfig:
    """Configuration for deterministic CME-front evidence."""

    temporal_median_window_frames: int = 1
    temporal_background_percentile: float = 25.0
    smooth_position_angle_deg: float = 2.0
    smooth_radius_px: float = 1.0
    excess_weight: float = 0.15
    running_difference_weight: float = 0.30
    leading_edge_weight: float = 0.55
    minimum_leading_edge_z: float = 0.75
    z_score_clip: float = 12.0
    scale_floor_fraction: float = 1e-6
    retain_diagnostics: bool = False

    def __post_init__(self) -> None:
        window = self.temporal_median_window_frames
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, Integral)
            or window < 1
            or window % 2 != 1
        ):
            raise LikelihoodError(
                "temporal_median_window_frames must be a positive odd integer."
            )
        if not 0.0 <= self.temporal_background_percentile <= 100.0:
            raise LikelihoodError(
                "temporal_background_percentile must be between 0 and 100."
            )
        for name, value in (
            ("smooth_position_angle_deg", self.smooth_position_angle_deg),
            ("smooth_radius_px", self.smooth_radius_px),
            ("excess_weight", self.excess_weight),
            ("running_difference_weight", self.running_difference_weight),
            ("leading_edge_weight", self.leading_edge_weight),
            ("minimum_leading_edge_z", self.minimum_leading_edge_z),
        ):
            if not math.isfinite(value) or value < 0:
                raise LikelihoodError(f"{name} must be finite and nonnegative.")
        if (
            self.excess_weight
            + self.running_difference_weight
            + self.leading_edge_weight
            <= 0
        ):
            raise LikelihoodError("At least one evidence weight must be positive.")
        if not math.isfinite(self.z_score_clip) or self.z_score_clip <= 0:
            raise LikelihoodError("z_score_clip must be finite and positive.")
        if (
            not math.isfinite(self.scale_floor_fraction)
            or self.scale_floor_fraction <= 0
        ):
            raise LikelihoodError(
                "scale_floor_fraction must be finite and positive."
            )


@dataclass(frozen=True)
class LikelihoodResult:
    """Front score and optional evidence channels on a common polar grid."""

    score: np.ndarray
    temporal_background: np.ndarray
    supported_frame_mask: np.ndarray
    base_excess_z: np.ndarray | None = None
    running_difference_z: np.ndarray | None = None
    leading_edge_z: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.score.ndim != 3:
            raise LikelihoodError(
                "Likelihood score must have shape (time, position_angle, radius)."
            )
        if self.temporal_background.shape != self.score.shape[1:]:
            raise LikelihoodError(
                "Temporal background shape does not match the polar score."
            )
        if self.supported_frame_mask.shape != (self.score.shape[0],):
            raise LikelihoodError(
                "supported_frame_mask must have one value per score frame."
            )
        if self.supported_frame_mask.dtype != np.bool_:
            raise LikelihoodError("supported_frame_mask must be boolean.")
        for name, values in (
            ("base_excess_z", self.base_excess_z),
            ("running_difference_z", self.running_difference_z),
            ("leading_edge_z", self.leading_edge_z),
        ):
            if values is not None and values.shape != self.score.shape:
                raise LikelihoodError(f"{name} shape does not match the score.")


def _nan_gaussian_filter(
    image: np.ndarray,
    sigma_position_angle: float,
    sigma_radius: float,
) -> np.ndarray:
    """Smooth a polar image without bleeding invalid-FOV NaNs into the data."""

    valid = np.isfinite(image)
    if not np.any(valid):
        return np.full(image.shape, np.nan, dtype=np.float64)
    if sigma_position_angle == 0 and sigma_radius == 0:
        return np.asarray(image, dtype=np.float64).copy()

    sigma = (sigma_position_angle, sigma_radius)
    numerator = gaussian_filter(
        np.where(valid, image, 0.0),
        sigma=sigma,
        mode=("wrap", "nearest"),
    )
    denominator = gaussian_filter(
        valid.astype(np.float64),
        sigma=sigma,
        mode=("wrap", "nearest"),
    )
    smoothed = np.full(image.shape, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=smoothed,
        where=denominator > 1e-8,
    )
    smoothed[~valid] = np.nan
    return smoothed


def _nan_statistic(function, values: np.ndarray, **kwargs) -> np.ndarray:
    """Run a NaN statistic while suppressing expected all-NaN FOV warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom", category=RuntimeWarning)
        return function(values, **kwargs)


def _centered_temporal_nanmedian(
    values: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """Median-filter the first axis without inventing endpoint samples.

    A strict majority of the centered window must be finite at a polar sample.
    Frames without a complete centered window are returned as NaN.  The
    window-one result is an exact copy of the input and therefore disables the
    filter.
    """

    if (
        isinstance(window_frames, (bool, np.bool_))
        or not isinstance(window_frames, Integral)
        or window_frames < 1
        or window_frames % 2 != 1
    ):
        raise LikelihoodError("Temporal-median window must be a positive odd integer.")
    if values.ndim < 1:
        raise LikelihoodError("Temporal-median input must have a time axis.")
    if window_frames > values.shape[0]:
        raise LikelihoodError(
            "temporal_median_window_frames cannot exceed the frame count."
        )

    source = np.asarray(values, dtype=np.float64)
    if window_frames == 1:
        return source.copy()

    half_width = window_frames // 2
    minimum_finite = half_width + 1
    filtered = np.full(source.shape, np.nan, dtype=np.float64)
    for frame_index in range(half_width, source.shape[0] - half_width):
        window = source[
            frame_index - half_width : frame_index + half_width + 1
        ]
        finite_count = np.count_nonzero(np.isfinite(window), axis=0)
        median = _nan_statistic(np.nanmedian, window, axis=0)
        filtered[frame_index] = np.where(
            finite_count >= minimum_finite,
            median,
            np.nan,
        )
    return filtered


def _robust_zscore_by_radius(
    values: np.ndarray,
    scale_floor_fraction: float,
) -> np.ndarray:
    """Robustly center and scale over time and position angle at each radius."""

    center = _nan_statistic(np.nanmedian, values, axis=(0, 1))
    absolute_deviation = np.abs(values - center[np.newaxis, np.newaxis, :])
    mad = _nan_statistic(np.nanmedian, absolute_deviation, axis=(0, 1))
    robust_scale = 1.4826 * mad

    # A perfectly static analytic background legitimately has zero MAD.  A
    # conventional standard deviation supplies a useful fallback for sparse
    # nonzero evidence, while the final floor keeps the all-zero case finite.
    standard_scale = _nan_statistic(np.nanstd, values, axis=(0, 1))
    absolute_reference = _nan_statistic(
        np.nanpercentile,
        np.abs(values),
        q=75.0,
        axis=(0, 1),
    )
    finite_values = np.abs(values[np.isfinite(values)])
    global_reference = (
        float(np.percentile(finite_values, 75.0)) if finite_values.size else 1.0
    )
    floor = np.maximum(
        np.nan_to_num(absolute_reference, nan=0.0) * scale_floor_fraction,
        max(global_reference * scale_floor_fraction, np.finfo(np.float64).eps),
    )
    scale = np.where(
        np.isfinite(robust_scale) & (robust_scale > floor),
        robust_scale,
        np.where(
            np.isfinite(standard_scale) & (standard_scale > floor),
            standard_scale,
            floor,
        ),
    )
    z_score = (values - center[np.newaxis, np.newaxis, :]) / scale[
        np.newaxis, np.newaxis, :
    ]
    z_score[~np.isfinite(values)] = np.nan
    return z_score


def score_front_likelihood(
    polar_images: np.ndarray,
    grid: PolarGrid,
    config: EvidenceConfig | None = None,
) -> LikelihoodResult:
    """Construct a deterministic CME leading-front score cube.

    Parameters
    ----------
    polar_images
        Array with shape ``(time, position_angle, radius)``.
    grid
        Geometry used to generate the polar images.
    config
        Evidence and robust-normalization choices.

    Notes
    -----
    The first supported frame has no predecessor, so its running-difference
    channel is initialized to zero.  With a centered temporal-median window
    greater than one, the half-window frames at each global endpoint are
    deliberately unsupported rather than padded; a three-frame window thus
    adds one frame of latency without shifting the retained timestamps.
    """

    if config is None:
        config = EvidenceConfig()
    values = np.asarray(polar_images, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != grid.shape:
        raise LikelihoodError(
            "polar_images must have shape (time, position_angle, radius) "
            "matching the supplied grid."
        )
    if values.shape[0] < 2:
        raise LikelihoodError("At least two polar images are required.")
    if not np.any(np.isfinite(values)):
        raise LikelihoodError("The polar sequence contains no finite image values.")
    if config.temporal_median_window_frames > values.shape[0]:
        raise LikelihoodError(
            "temporal_median_window_frames cannot exceed the frame count."
        )

    angle_sigma_bins = (
        config.smooth_position_angle_deg / grid.position_angle_step_deg
    )
    radial_step_px = grid.radial_step_px
    if not math.isfinite(radial_step_px) or radial_step_px <= 0:
        raise LikelihoodError("Likelihood scoring requires at least two radial bins.")
    radial_sigma_bins = config.smooth_radius_px / radial_step_px

    temporal_window = int(config.temporal_median_window_frames)
    half_window = temporal_window // 2
    supported_frame_mask = np.ones(values.shape[0], dtype=bool)
    if temporal_window == 1:
        # Keep the established science path exactly unchanged when disabled.
        evidence_input = values
    else:
        evidence_input = _centered_temporal_nanmedian(values, temporal_window)
        supported_frame_mask[:half_window] = False
        supported_frame_mask[-half_window:] = False

    smoothed = np.empty_like(evidence_input)
    for frame_index, frame in enumerate(evidence_input):
        smoothed[frame_index] = _nan_gaussian_filter(
            frame,
            angle_sigma_bins,
            radial_sigma_bins,
        )
    evidence_images = smoothed

    temporal_background = _nan_statistic(
        np.nanpercentile,
        evidence_images,
        q=config.temporal_background_percentile,
        axis=0,
    )
    base_excess = evidence_images - temporal_background[np.newaxis, :, :]

    if temporal_window == 1:
        # This branch intentionally preserves the original floating-point and
        # NaN behavior for a scientifically clean A/B control.
        running_difference = np.zeros_like(evidence_images)
        running_difference[0, np.isfinite(evidence_images[0])] = 0.0
        running_difference[1:] = evidence_images[1:] - evidence_images[:-1]
    else:
        running_difference = np.full_like(evidence_images, np.nan)
        first_supported = half_window
        first_finite = np.isfinite(evidence_images[first_supported])
        running_difference[first_supported, first_finite] = 0.0
        supported_stop = evidence_images.shape[0] - half_window
        running_difference[first_supported + 1 : supported_stop] = (
            evidence_images[first_supported + 1 : supported_stop]
            - evidence_images[first_supported : supported_stop - 1]
        )

    # For increasing radius, the outer edge of a bright front is a negative
    # intensity gradient.  Negating the derivative makes it positive evidence.
    leading_edge = -np.gradient(base_excess, radial_step_px, axis=-1)

    base_z = _robust_zscore_by_radius(
        base_excess,
        config.scale_floor_fraction,
    )
    running_z = _robust_zscore_by_radius(
        running_difference,
        config.scale_floor_fraction,
    )
    edge_z = _robust_zscore_by_radius(
        leading_edge,
        config.scale_floor_fraction,
    )

    positive_base = np.clip(base_z, 0.0, config.z_score_clip)
    positive_running = np.clip(running_z, 0.0, config.z_score_clip)
    positive_edge = np.clip(edge_z, 0.0, config.z_score_clip)

    weight_total = (
        config.excess_weight
        + config.running_difference_weight
        + config.leading_edge_weight
    )
    score = (
        config.excess_weight * positive_base
        + config.running_difference_weight * positive_running
        + config.leading_edge_weight * positive_edge
    ) / weight_total

    score[edge_z < config.minimum_leading_edge_z] = 0.0
    score[~np.isfinite(evidence_images)] = np.nan

    diagnostics = config.retain_diagnostics
    return LikelihoodResult(
        score=np.asarray(score, dtype=np.float32),
        temporal_background=np.asarray(temporal_background, dtype=np.float32),
        supported_frame_mask=supported_frame_mask,
        base_excess_z=np.asarray(base_z, dtype=np.float32) if diagnostics else None,
        running_difference_z=(
            np.asarray(running_z, dtype=np.float32) if diagnostics else None
        ),
        leading_edge_z=np.asarray(edge_z, dtype=np.float32) if diagnostics else None,
    )
