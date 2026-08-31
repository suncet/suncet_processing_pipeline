"""Reduce an angularly resolved front to provisional event-level samples."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tracking import FrontTrack


@dataclass(frozen=True)
class FrontSummaryConfig:
    """Explicit provisional choices for the headline scalar track."""

    height_percentile: float = 90.0
    minimum_angles_per_frame: int = 3
    score_reference: float = 2.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.height_percentile <= 100.0:
            raise ValueError("height_percentile must be between zero and 100.")
        if self.minimum_angles_per_frame < 1:
            raise ValueError("minimum_angles_per_frame must be positive.")
        if not math.isfinite(self.score_reference) or self.score_reference <= 0:
            raise ValueError("score_reference must be finite and positive.")


@dataclass(frozen=True)
class FrontSummary:
    """Provisional robust-apex summary, one row per input frame."""

    height_rsun: np.ndarray
    height_sigma_rsun: np.ndarray
    position_angle_deg: np.ndarray
    angular_width_deg: np.ndarray
    coverage_fraction: np.ndarray
    confidence: np.ndarray
    observed_angle_count: np.ndarray


def _circular_center_width(angles_deg: np.ndarray, step_deg: float) -> tuple[float, float]:
    """Return circular mean and smallest sampled arc containing all angles."""

    angles = np.mod(np.asarray(angles_deg, dtype=np.float64), 360.0)
    if angles.size == 0:
        return math.nan, math.nan
    radians = np.deg2rad(angles)
    vector = np.mean(np.exp(1j * radians))
    center = (
        float(np.mod(np.rad2deg(np.angle(vector)), 360.0))
        if abs(vector) >= 1e-6
        else math.nan
    )
    if angles.size == 1:
        return center, float(step_deg)
    ordered = np.sort(angles)
    circular_gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    largest_gap = float(np.max(circular_gaps))
    width = min(360.0, 360.0 - largest_gap + step_deg)
    return center, float(width)


def summarize_front(
    track: FrontTrack,
    config: FrontSummaryConfig | None = None,
) -> FrontSummary:
    """Compute a robust near-apex height without filling tracker gaps.

    The 90th radial percentile is deliberately less noise-sensitive than a
    single-pixel maximum while staying close to the visible leading apex.  This
    definition remains provisional until the SunCET science-product convention
    is approved.
    """

    if config is None:
        config = FrontSummaryConfig()
    frame_count, angle_count = track.radius_rsun.shape
    if angle_count != track.position_angle_deg.size:
        raise ValueError("Track angle axis and position-angle values disagree.")
    if angle_count > 1:
        step_deg = 360.0 / angle_count
    else:
        step_deg = 360.0

    observed = track.observed_mask & np.isfinite(track.radius_rsun)
    event_support = np.any(observed, axis=0)
    expected_angle_count = max(int(np.count_nonzero(event_support)), 1)

    height = np.full(frame_count, np.nan, dtype=np.float64)
    height_sigma = np.full(frame_count, np.nan, dtype=np.float64)
    position_angle = np.full(frame_count, np.nan, dtype=np.float64)
    width = np.full(frame_count, np.nan, dtype=np.float64)
    coverage = np.zeros(frame_count, dtype=np.float64)
    confidence = np.zeros(frame_count, dtype=np.float64)
    counts = np.count_nonzero(observed, axis=1).astype(np.int32)

    radius_conversion = np.divide(
        track.radius_px,
        track.radius_rsun,
        out=np.full(track.radius_px.shape, np.nan, dtype=np.float64),
        where=np.isfinite(track.radius_px) & np.isfinite(track.radius_rsun),
    )
    finite_conversion = radius_conversion[np.isfinite(radius_conversion)]
    solar_radius_px = (
        float(np.median(finite_conversion)) if finite_conversion.size else math.nan
    )

    for frame_index in range(frame_count):
        frame_mask = observed[frame_index]
        count = int(counts[frame_index])
        coverage[frame_index] = min(1.0, count / expected_angle_count)
        if count < config.minimum_angles_per_frame:
            continue

        radii = track.radius_rsun[frame_index, frame_mask]
        height[frame_index] = float(
            np.percentile(radii, config.height_percentile)
        )
        position_angle[frame_index], width[frame_index] = _circular_center_width(
            track.position_angle_deg[frame_mask], step_deg
        )

        scores = track.score[frame_index, frame_mask]
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size:
            excess = max(float(np.median(finite_scores)) / config.score_reference - 1.0, 0.0)
            confidence[frame_index] = coverage[frame_index] * (1.0 - math.exp(-excess))

        # Localization width is the measurement component.  Angular scatter is
        # included only as a standard-error term; it is not treated as if front
        # curvature itself were measurement noise.
        if math.isfinite(solar_radius_px) and solar_radius_px > 0:
            localization = (
                track.radial_sigma_px[frame_index, frame_mask] / solar_radius_px
            )
            localization = localization[np.isfinite(localization)]
        else:
            localization = np.array([], dtype=np.float64)
        mad = float(np.median(np.abs(radii - np.median(radii))))
        angular_standard_error = 1.4826 * mad / math.sqrt(count)
        localization_sigma = (
            float(np.median(localization)) if localization.size else 0.0
        )
        combined = math.hypot(localization_sigma, angular_standard_error)
        height_sigma[frame_index] = combined if combined > 0 else math.nan

    return FrontSummary(
        height_rsun=height,
        height_sigma_rsun=height_sigma,
        position_angle_deg=position_angle,
        angular_width_deg=width,
        coverage_fraction=coverage,
        confidence=np.clip(confidence, 0.0, 1.0),
        observed_angle_count=counts,
    )
