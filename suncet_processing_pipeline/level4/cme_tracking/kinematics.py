"""Provisional height-time fitting and CME kinematic estimation.

The routines in this module operate on measured, projected CME heights at
their *actual* elapsed observation times.  The returned speed and acceleration
are analytic derivatives of one fitted spline; they are deliberately not raw
finite differences, which would amplify front-location noise.

This is an initial method, not yet a frozen Level 4 product definition.  In
particular, endpoint samples are flagged because spline derivatives are less
well constrained there, and the uncertainty calculation is a deterministic
Monte Carlo/bootstrap estimate rather than a final science error model.

No height unit is imposed.  If input heights are in solar radii, output speed
and acceleration are in solar radii per second and solar radii per second
squared, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import UnivariateSpline


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class KinematicsResult:
    """A fitted height-time track and its derived projected kinematics.

    ``valid_mask`` is the initial science-quality mask for the derivative
    products.  It includes the fit domain and excludes ``endpoint_mask`` plus
    intervals marked by ``unsupported_gap_mask``; later uncertainty thresholds
    can further restrict it.  ``measurement_mask`` distinguishes
    observed heights from values interpolated across an internal gap.
    Arrays of fitted uncertainties contain ``NaN`` when uncertainty sampling
    was not requested.
    """

    elapsed_seconds: FloatArray
    raw_height: FloatArray
    raw_height_sigma: FloatArray
    measurement_mask: BoolArray
    fit_domain_mask: BoolArray
    fitted_height: FloatArray
    speed: FloatArray
    acceleration: FloatArray
    fitted_height_sigma: FloatArray
    speed_sigma: FloatArray
    acceleration_sigma: FloatArray
    endpoint_mask: BoolArray
    unsupported_gap_mask: BoolArray
    valid_mask: BoolArray
    method_metadata: dict[str, Any]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _height_vector(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("raw_height must be a one-dimensional array")
    if np.any(np.isinf(array)):
        raise ValueError("raw_height may contain finite values or NaN, not infinity")
    return array


def _fit_spline(
    normalized_time: FloatArray,
    height: FloatArray,
    weights: FloatArray | None,
    smoothing_factor: float,
    degree: int,
) -> UnivariateSpline:
    return UnivariateSpline(
        normalized_time,
        height,
        w=weights,
        k=degree,
        s=smoothing_factor,
        ext="raise",
        check_finite=False,
    )


def _evaluate_spline(
    spline: UnivariateSpline,
    normalized_time: FloatArray,
    time_scale_seconds: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    height = np.asarray(spline(normalized_time), dtype=np.float64)
    speed = np.asarray(
        spline.derivative(1)(normalized_time) / time_scale_seconds,
        dtype=np.float64,
    )
    acceleration = np.asarray(
        spline.derivative(2)(normalized_time) / time_scale_seconds**2,
        dtype=np.float64,
    )
    return height, speed, acceleration


def fit_kinematics(
    elapsed_seconds: ArrayLike,
    raw_height: ArrayLike,
    height_sigma: ArrayLike | None = None,
    *,
    smoothing_factor: float | None = None,
    endpoint_samples: int = 1,
    uncertainty_samples: int = 0,
    random_seed: int = 0,
    maximum_gap_seconds: float | None = None,
) -> KinematicsResult:
    """Fit projected CME height and analytically derive speed and acceleration.

    Parameters
    ----------
    elapsed_seconds
        Actual elapsed observation time in seconds.  Samples may be irregularly
        spaced but must be finite and strictly increasing.
    raw_height
        Measured projected height in any consistent unit.  ``NaN`` denotes a
        missing front measurement and is retained in the result.  The spline
        is fit to finite measurements only and may bridge internal gaps.
    height_sigma
        Optional positive one-sigma measurement uncertainty for each height.
        ``NaN`` is permitted only where ``raw_height`` is missing.  When
        supplied, the spline uses inverse-sigma weights.  FITPACK defines its
        smoothing objective using squared weights, so this corresponds to
        inverse-variance residual weighting.
    smoothing_factor
        FITPACK smoothing target ``s``.  With supplied uncertainties, the
        default is the number of samples, the expected order of the weighted
        chi-square.  Without supplied uncertainties, the transparent default
        is zero (interpolation) rather than inventing a noise scale.  A
        positive caller-selected value should be used for noisy unweighted
        data.
    endpoint_samples
        Number of samples flagged at each edge of the fitted time domain.  The
        fitted values are still returned there, but ``valid_mask`` is false.
    uncertainty_samples
        Number of deterministic Monte Carlo/bootstrap refits.  Zero disables
        uncertainty propagation; otherwise at least two are required.  With
        ``height_sigma``, Gaussian measurement perturbations are used.  Without
        it, centered fit residuals are resampled with replacement.
    random_seed
        Seed used for uncertainty sampling.
    maximum_gap_seconds
        Optional largest supported interval between consecutive measurements.
        Rows bordering a larger interval, plus any intervening rows, remain in
        the inspectable fit but are excluded from ``valid_mask``.

    Notes
    -----
    Time is normalized internally before fitting and derivatives are converted
    back to per-second units.  This improves conditioning and guarantees that
    merely changing an assumed cadence rescales speed and acceleration by the
    expected first- and second-power factors.

    The spline is evaluated only from the first through the last finite height
    measurement.  Returned fitted values and derivatives remain ``NaN``
    outside that domain; this routine does not extrapolate a CME track.
    """

    time = _finite_vector(elapsed_seconds, name="elapsed_seconds")
    height = _height_vector(raw_height)
    if time.size != height.size:
        raise ValueError("elapsed_seconds and raw_height must have equal length")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("elapsed_seconds must be strictly increasing")

    measurement_mask = np.isfinite(height)
    measurement_count = int(np.count_nonzero(measurement_mask))
    if measurement_count < 4:
        raise ValueError("at least four finite height measurements are required")
    measurement_indices = np.flatnonzero(measurement_mask)
    first_measurement = int(measurement_indices[0])
    last_measurement = int(measurement_indices[-1])
    fit_domain_mask = np.zeros(time.shape, dtype=np.bool_)
    fit_domain_mask[first_measurement : last_measurement + 1] = True
    fit_domain_indices = np.flatnonzero(fit_domain_mask)

    if isinstance(endpoint_samples, bool) or not isinstance(
        endpoint_samples, (int, np.integer)
    ):
        raise ValueError("endpoint_samples must be an integer")
    if endpoint_samples < 0 or 2 * endpoint_samples >= fit_domain_indices.size:
        raise ValueError(
            "endpoint_samples must be nonnegative and leave at least one "
            "interior sample"
        )
    if isinstance(uncertainty_samples, bool) or not isinstance(
        uncertainty_samples, (int, np.integer)
    ):
        raise ValueError("uncertainty_samples must be an integer")
    if uncertainty_samples == 1 or uncertainty_samples < 0:
        raise ValueError("uncertainty_samples must be zero or at least two")
    if isinstance(random_seed, bool) or not isinstance(
        random_seed, (int, np.integer)
    ):
        raise ValueError("random_seed must be an integer")
    if maximum_gap_seconds is not None:
        maximum_gap_seconds = float(maximum_gap_seconds)
        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("maximum_gap_seconds must be finite and positive")

    sigma: FloatArray | None
    if height_sigma is None:
        sigma = None
        raw_sigma = np.full(time.shape, np.nan, dtype=np.float64)
        weights = None
    else:
        supplied_sigma = np.asarray(height_sigma, dtype=np.float64)
        if supplied_sigma.ndim != 1:
            raise ValueError("height_sigma must be a one-dimensional array")
        if supplied_sigma.size != time.size:
            raise ValueError(
                "height_sigma and elapsed_seconds must have equal length"
            )
        if np.any(np.isinf(supplied_sigma)):
            raise ValueError("height_sigma may contain positive values or NaN")
        if np.any(np.isnan(supplied_sigma) & measurement_mask):
            raise ValueError(
                "height_sigma must be finite and positive at every measurement"
            )
        finite_sigma_mask = np.isfinite(supplied_sigma)
        if np.any(supplied_sigma[finite_sigma_mask] <= 0.0):
            raise ValueError("finite height_sigma values must all be positive")
        raw_sigma = supplied_sigma.copy()
        raw_sigma[~measurement_mask] = np.nan
        sigma = supplied_sigma[measurement_mask]
        weights = 1.0 / sigma

    if smoothing_factor is None:
        if sigma is None:
            smoothing = 0.0
            smoothing_source = "interpolating_default_without_uncertainties"
        else:
            smoothing = float(measurement_count)
            smoothing_source = "expected_weighted_chi_square"
    else:
        smoothing = float(smoothing_factor)
        if not np.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("smoothing_factor must be finite and nonnegative")
        smoothing_source = "caller"

    measured_time = time[measurement_mask]
    measured_height = height[measurement_mask]
    time_origin_seconds = float(measured_time[0])
    time_scale_seconds = float(measured_time[-1] - measured_time[0])
    normalized_measured_time = (
        measured_time - time_origin_seconds
    ) / time_scale_seconds
    domain_time = time[fit_domain_mask]
    normalized_domain_time = (
        domain_time - time_origin_seconds
    ) / time_scale_seconds
    degree = min(3, measurement_count - 1)

    spline = _fit_spline(
        normalized_measured_time,
        measured_height,
        weights,
        smoothing,
        degree,
    )
    domain_height, domain_speed, domain_acceleration = _evaluate_spline(
        spline,
        normalized_domain_time,
        time_scale_seconds,
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (domain_height, domain_speed, domain_acceleration)
    ):
        raise RuntimeError("spline fit produced non-finite kinematics")

    fitted_height = np.full(time.shape, np.nan, dtype=np.float64)
    speed = np.full(time.shape, np.nan, dtype=np.float64)
    acceleration = np.full(time.shape, np.nan, dtype=np.float64)
    fitted_height[fit_domain_mask] = domain_height
    speed[fit_domain_mask] = domain_speed
    acceleration[fit_domain_mask] = domain_acceleration

    fitted_height_sigma = np.full(time.shape, np.nan, dtype=np.float64)
    speed_sigma = np.full(time.shape, np.nan, dtype=np.float64)
    acceleration_sigma = np.full(time.shape, np.nan, dtype=np.float64)
    uncertainty_method = "none"

    if uncertainty_samples:
        rng = np.random.default_rng(int(random_seed))
        sampled_heights = np.empty(
            (uncertainty_samples, fit_domain_indices.size), dtype=np.float64
        )
        sampled_speeds = np.empty_like(sampled_heights)
        sampled_accelerations = np.empty_like(sampled_heights)

        if sigma is not None:
            uncertainty_method = "gaussian_measurement_monte_carlo"

            def draw_height() -> FloatArray:
                return measured_height + rng.normal(loc=0.0, scale=sigma)

        else:
            centered_residuals = measured_height - fitted_height[measurement_mask]
            centered_residuals -= np.mean(centered_residuals)
            residual_tolerance = np.finfo(np.float64).eps * max(
                1.0, float(np.max(np.abs(measured_height)))
            )
            if np.max(np.abs(centered_residuals)) <= residual_tolerance:
                raise ValueError(
                    "residual bootstrap cannot estimate uncertainty from an "
                    "exact/interpolating fit; supply height_sigma or choose "
                    "positive smoothing_factor"
                )
            uncertainty_method = "centered_residual_bootstrap"

            def draw_height() -> FloatArray:
                residual_draw = rng.choice(
                    centered_residuals,
                    size=measurement_count,
                    replace=True,
                )
                return fitted_height[measurement_mask] + residual_draw

        for sample_index in range(uncertainty_samples):
            sample_spline = _fit_spline(
                normalized_measured_time,
                draw_height(),
                weights,
                smoothing,
                degree,
            )
            (
                sampled_heights[sample_index],
                sampled_speeds[sample_index],
                sampled_accelerations[sample_index],
            ) = _evaluate_spline(
                sample_spline,
                normalized_domain_time,
                time_scale_seconds,
            )

        fitted_height_sigma[fit_domain_mask] = np.std(
            sampled_heights, axis=0, ddof=1
        )
        speed_sigma[fit_domain_mask] = np.std(
            sampled_speeds, axis=0, ddof=1
        )
        acceleration_sigma[fit_domain_mask] = np.std(
            sampled_accelerations,
            axis=0,
            ddof=1,
        )

    endpoint_mask = np.zeros(time.shape, dtype=np.bool_)
    if endpoint_samples:
        endpoint_mask[fit_domain_indices[:endpoint_samples]] = True
        endpoint_mask[fit_domain_indices[-endpoint_samples:]] = True
    unsupported_gap_mask = np.zeros(time.shape, dtype=np.bool_)
    if maximum_gap_seconds is not None:
        for left, right in zip(
            measurement_indices[:-1], measurement_indices[1:], strict=True
        ):
            if time[right] - time[left] > maximum_gap_seconds:
                unsupported_gap_mask[left : right + 1] = True
    valid_mask = fit_domain_mask & ~endpoint_mask & ~unsupported_gap_mask

    residuals = measured_height - fitted_height[measurement_mask]
    metadata: dict[str, Any] = {
        "method": "scipy.interpolate.UnivariateSpline",
        "provisional": True,
        "spline_degree": degree,
        "smoothing_factor": smoothing,
        "smoothing_factor_source": smoothing_source,
        "weighting": "inverse_measurement_sigma" if sigma is not None else "none",
        "time_origin_seconds": time_origin_seconds,
        "time_scale_seconds": time_scale_seconds,
        "measurement_count": measurement_count,
        "fit_domain_count": int(fit_domain_indices.size),
        "endpoint_samples": int(endpoint_samples),
        "uncertainty_method": uncertainty_method,
        "uncertainty_samples": int(uncertainty_samples),
        "random_seed": int(random_seed) if uncertainty_samples else None,
        "maximum_gap_seconds": maximum_gap_seconds,
        "unsupported_gap_count": int(np.count_nonzero(unsupported_gap_mask)),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
    }

    return KinematicsResult(
        elapsed_seconds=time.copy(),
        raw_height=height.copy(),
        raw_height_sigma=raw_sigma,
        measurement_mask=measurement_mask,
        fit_domain_mask=fit_domain_mask,
        fitted_height=fitted_height,
        speed=speed,
        acceleration=acceleration,
        fitted_height_sigma=fitted_height_sigma,
        speed_sigma=speed_sigma,
        acceleration_sigma=acceleration_sigma,
        endpoint_mask=endpoint_mask,
        unsupported_gap_mask=unsupported_gap_mask,
        valid_mask=valid_mask,
        method_metadata=metadata,
    )
