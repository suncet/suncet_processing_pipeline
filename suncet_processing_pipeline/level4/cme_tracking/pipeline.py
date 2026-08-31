"""End-to-end known-window Level 4 CME tracking orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..common.provenance import collect_run_provenance
from ..common.quality import QualityFlag, decode_quality_flags
from .config import CMETrackingConfig
from .headline_filter import (
    HeadlineHeightFilterResult,
    filter_headline_height_outliers,
)
from .input import ImageSequence
from .kinematics import KinematicsResult, fit_kinematics
from .likelihood import LikelihoodResult, score_front_likelihood
from .manifest import CadenceStatus, CorrectionState, InputSourceKind
from .movie import write_cme_tracking_movie
from .polar import PolarGrid, build_polar_grid, remap_sequence_to_polar
from .products import (
    FrontOverlayFrame,
    FrontOverlayProduct,
    FrontSamplesProduct,
    TrackProduct,
    write_event_products,
)
from .summary import FrontSummary, summarize_front
from .tracking import FrontState, FrontTrack, extract_front


@dataclass(frozen=True)
class CMETrackingRun:
    """In-memory result of one known-window algorithm run."""

    sequence: ImageSequence
    configuration: CMETrackingConfig
    polar_grid: PolarGrid
    likelihood: LikelihoodResult
    front: FrontTrack
    summary: FrontSummary
    field_of_view_limited_mask: np.ndarray
    headline_height_filter: HeadlineHeightFilterResult
    kinematics: KinematicsResult | None
    kinematics_error: str | None


def run_known_window(
    sequence: ImageSequence,
    configuration: CMETrackingConfig | None = None,
) -> CMETrackingRun:
    """Track a single expected event in one already validated image window.

    The full Cartesian sequence is materialized explicitly for this reference
    implementation.  The input boundary remains lazy, making a future chunked
    or GPU adapter possible without weakening input validation.
    """

    images = sequence.materialize(dtype=np.float32)
    return run_known_window_from_images(sequence, images, configuration)


def run_known_window_from_images(
    sequence: ImageSequence,
    images: np.ndarray,
    configuration: CMETrackingConfig | None = None,
) -> CMETrackingRun:
    """Track one event from an already resident Cartesian image sequence.

    This is the compute-only boundary used by controlled benchmarks.  It runs
    the same deterministic science path as :func:`run_known_window` while
    excluding FITS reads and manifest hash verification from the measured
    interval.  The caller retains ownership of ``images``; this function does
    not modify it.
    """

    resident = np.asarray(images)
    expected_shape = (
        sequence.frame_count,
        *sequence.geometry.image_shape_yx,
    )
    if resident.shape != expected_shape:
        raise ValueError(
            f"Resident images have shape {resident.shape}; expected "
            f"{expected_shape}."
        )
    if not np.issubdtype(resident.dtype, np.number):
        raise TypeError("Resident images must have a numeric dtype.")
    finite_by_frame = np.any(np.isfinite(resident), axis=(1, 2))
    if not np.all(finite_by_frame):
        missing = np.flatnonzero(~finite_by_frame)
        raise ValueError(
            "Every resident frame must contain at least one finite pixel; "
            f"invalid frame rows: {missing.tolist()}."
        )

    config = configuration or CMETrackingConfig()
    grid = build_polar_grid(sequence.geometry, config.polar)
    polar_images = remap_sequence_to_polar(resident, grid)
    likelihood = score_front_likelihood(polar_images, grid, config.evidence)
    del polar_images
    front = extract_front(
        likelihood,
        grid,
        config.tracking,
        frame_numbers=tuple(frame.frame_number for frame in sequence.frames),
    )
    summary = summarize_front(front, config.summary)
    field_of_view_limited = _field_of_view_limited_frames(
        front,
        grid,
        margin_px=config.field_of_view_margin_px,
        top_fraction=config.field_of_view_top_fraction,
        contact_fraction=config.field_of_view_contact_fraction,
        minimum_contact_angles=(
            config.field_of_view_minimum_contact_angles
        ),
        minimum_consecutive_frames=(
            config.field_of_view_minimum_consecutive_frames
        ),
    )

    kinematics: KinematicsResult | None = None
    kinematics_error: str | None = None
    raw_height = np.asarray(summary.height_rsun, dtype=np.float64)
    kinematic_quality = (
        (summary.coverage_fraction >= config.kinematics.minimum_front_coverage)
        & (summary.confidence >= config.kinematics.minimum_confidence)
        & ~field_of_view_limited
    )
    nominal_interval_seconds = float(np.median(np.diff(sequence.elapsed_seconds)))
    maximum_gap_seconds = (
        config.kinematics.maximum_gap_factor * nominal_interval_seconds
    )
    height_filter = filter_headline_height_outliers(
        sequence.elapsed_seconds,
        raw_height,
        support_mask=np.isfinite(raw_height) & ~field_of_view_limited,
        candidate_mask=kinematic_quality & np.isfinite(raw_height),
        enabled=config.kinematics.height_outlier_filter_enabled,
        window_samples=config.kinematics.height_outlier_filter_window_samples,
        absolute_tolerance=(
            config.kinematics.height_outlier_filter_absolute_tolerance_rsun
        ),
        minimum_neighbors=(
            config.kinematics.height_outlier_filter_minimum_neighbors
        ),
        maximum_gap_seconds=maximum_gap_seconds,
    )
    height_for_fit = height_filter.kinematic_height
    measurement = np.isfinite(height_for_fit)
    if front.event_detected and np.count_nonzero(measurement) >= 4:
        radial_floor = max(
            0.5 * grid.radial_step_px / sequence.geometry.solar_radius_px,
            np.finfo(np.float64).eps,
        )
        sigma = np.asarray(summary.height_sigma_rsun, dtype=np.float64).copy()
        sigma[measurement] = np.maximum(
            np.nan_to_num(sigma[measurement], nan=radial_floor), radial_floor
        )
        sigma[~measurement] = np.nan
        endpoint_samples = min(
            config.kinematics.endpoint_samples,
            max((int(np.count_nonzero(measurement)) - 1) // 2, 0),
        )
        try:
            kinematics = fit_kinematics(
                sequence.elapsed_seconds,
                height_for_fit,
                sigma,
                smoothing_factor=config.kinematics.smoothing_factor,
                endpoint_samples=endpoint_samples,
                uncertainty_samples=config.kinematics.uncertainty_samples,
                random_seed=config.kinematics.random_seed,
                maximum_gap_seconds=maximum_gap_seconds,
            )
            filter_metadata = dict(height_filter.method_metadata)
            filter_metadata["rejected_frame_numbers"] = [
                int(sequence.frames[index].frame_number)
                for index in np.flatnonzero(height_filter.outlier_mask)
            ]
            kinematics = replace(
                kinematics,
                method_metadata={
                    **kinematics.method_metadata,
                    "headline_height_outlier_filter": filter_metadata,
                },
            )
        except (ValueError, RuntimeError) as exc:
            kinematics_error = f"{type(exc).__name__}: {exc}"
    else:
        kinematics_error = (
            "No coherent event was detected."
            if not front.event_detected
            else (
                "Fewer than four headline-height samples passed the provisional "
                "coverage, confidence, and field-of-view criteria."
            )
        )

    return CMETrackingRun(
        sequence=sequence,
        configuration=config,
        polar_grid=grid,
        likelihood=likelihood,
        front=front,
        summary=summary,
        field_of_view_limited_mask=field_of_view_limited,
        headline_height_filter=height_filter,
        kinematics=kinematics,
        kinematics_error=kinematics_error,
    )


def _field_of_view_limited_frames(
    front: FrontTrack,
    grid: PolarGrid,
    *,
    margin_px: float,
    top_fraction: float,
    contact_fraction: float,
    minimum_contact_angles: int = 2,
    minimum_consecutive_frames: int = 2,
) -> np.ndarray:
    """Flag frames whose leading radial samples approach the rectangular FOV.

    A front can leave a rectangular image first at only some position angles.
    The test therefore considers the outer ``top_fraction`` of retained front
    radii and flags a frame when enough of those samples are within
    ``margin_px`` of the last valid polar sample. Boundary contact is confirmed
    either by enough simultaneous leading-edge angles in one frame or by the
    configured number of consecutive single-angle contacts. Such rows remain
    in the raw product but are excluded from kinematic fitting to avoid
    artificial deceleration as the measured height plateaus at a boundary.
    """

    frame_count, angle_count = front.radius_px.shape
    maximum_valid_radius = np.full(angle_count, np.nan, dtype=np.float64)
    for angle_index in range(angle_count):
        valid_radii = grid.radius_px[grid.valid_mask[angle_index]]
        if valid_radii.size:
            maximum_valid_radius[angle_index] = float(valid_radii[-1])

    if minimum_consecutive_frames < 1:
        raise ValueError("minimum_consecutive_frames must be positive.")
    if minimum_contact_angles < 1:
        raise ValueError("minimum_contact_angles must be positive.")

    contact_by_frame = np.zeros(frame_count, dtype=bool)
    spatially_confirmed_by_frame = np.zeros(frame_count, dtype=bool)
    for frame_index in range(frame_count):
        observed = front.observed_mask[frame_index]
        radii = front.radius_px[frame_index, observed]
        available = maximum_valid_radius[observed]
        finite = np.isfinite(radii) & np.isfinite(available)
        radii = radii[finite]
        available = available[finite]
        if radii.size == 0:
            continue
        cutoff = float(np.percentile(radii, 100.0 * (1.0 - top_fraction)))
        leading = radii >= cutoff
        contact = (available - radii) <= margin_px
        contact_count = int(np.count_nonzero(leading & contact))
        contact_by_frame[frame_index] = (
            contact_count / np.count_nonzero(leading) >= contact_fraction
        )
        spatially_confirmed_by_frame[frame_index] = (
            contact_by_frame[frame_index]
            and contact_count >= minimum_contact_angles
        )
    # The first implementation is restricted to an outward phase. Once the
    # leading front reaches the boundary, later lower-radius measurements do
    # not prove that it re-entered the field; they are more likely a different
    # part of the truncated front. Preserve the raw samples but stop the
    # kinematic measurement interval at first contact.
    limited = np.zeros(frame_count, dtype=bool)
    spatial_starts = np.flatnonzero(spatially_confirmed_by_frame)
    if spatial_starts.size:
        limited[int(spatial_starts[0]) :] = True
        return limited
    latest_start = frame_count - minimum_consecutive_frames
    for start in range(max(latest_start + 1, 0)):
        stop = start + minimum_consecutive_frames
        if np.all(contact_by_frame[start:stop]):
            limited[start:] = True
            break
    return limited


def _base_quality_mask(run: CMETrackingRun) -> QualityFlag:
    sequence = run.sequence
    mask = QualityFlag.NONE
    if sequence.time_axis.cadence_status == CadenceStatus.ASSUMED:
        mask |= QualityFlag.ASSUMED_CADENCE
    if not sequence.time_axis.absolute_time_valid:
        mask |= QualityFlag.ABSOLUTE_TIME_UNAVAILABLE
    if sequence.source_kind == InputSourceKind.SYNTHETIC_BYPASS:
        mask |= QualityFlag.SYNTHETIC_BYPASS
    if (
        sequence.upstream_processing.level2_psf_deconvolution
        != CorrectionState.APPLIED
    ):
        mask |= QualityFlag.LEVEL2_PSF_NOT_APPLIED
    if (
        sequence.upstream_processing.level3_geometric_correction
        != CorrectionState.APPLIED
    ):
        mask |= QualityFlag.LEVEL3_GEOMETRY_NOT_APPLIED
    if sequence.hash_verification_status != "verified_at_load_and_read":
        mask |= QualityFlag.INPUT_INTEGRITY_UNVERIFIED
    return mask


def _track_quality_masks(run: CMETrackingRun) -> np.ndarray:
    summary = run.summary
    count = run.sequence.frame_count
    masks = np.full(count, int(_base_quality_mask(run)), dtype=np.uint32)
    measured = np.isfinite(summary.height_rsun)
    masks[~measured] |= int(QualityFlag.FRONT_NOT_DETECTED)
    masks[
        summary.coverage_fraction
        < run.configuration.kinematics.minimum_front_coverage
    ] |= int(QualityFlag.LOW_FRONT_COVERAGE)
    masks[
        summary.confidence < run.configuration.kinematics.minimum_confidence
    ] |= int(QualityFlag.LOW_CONFIDENCE)
    masks[run.field_of_view_limited_mask] |= int(
        QualityFlag.PARTIAL_FIELD_OF_VIEW
    )
    masks[run.headline_height_filter.outlier_mask] |= int(
        QualityFlag.KINEMATIC_HEIGHT_OUTLIER
    )
    masks[~run.likelihood.supported_frame_mask] |= int(
        QualityFlag.TEMPORAL_EVIDENCE_UNSUPPORTED
    )

    measured_indices = np.flatnonzero(measured)
    if measured_indices.size:
        internal = np.zeros(count, dtype=bool)
        internal[measured_indices[0] : measured_indices[-1] + 1] = True
        masks[internal & ~measured] |= int(QualityFlag.TRACK_GAP)

    if run.kinematics is None:
        masks |= int(QualityFlag.KINEMATICS_NOT_FIT)
        masks |= int(QualityFlag.UNCERTAINTY_UNAVAILABLE)
    else:
        masks[run.kinematics.endpoint_mask] |= int(
            QualityFlag.DERIVATIVE_ENDPOINT
        )
        masks[run.kinematics.unsupported_gap_mask] |= int(QualityFlag.TRACK_GAP)
        missing_uncertainty = (
            run.kinematics.fit_domain_mask
            & ~np.isfinite(run.kinematics.speed_sigma)
        )
        masks[missing_uncertainty] |= int(QualityFlag.UNCERTAINTY_UNAVAILABLE)
    return masks


def _kinematic_product_arrays(
    run: CMETrackingRun,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = run.sequence.frame_count
    empty = lambda: np.full(count, np.nan, dtype=np.float64)
    if run.kinematics is None:
        return empty(), empty(), empty(), empty(), empty(), empty()
    result = run.kinematics
    solar_radius_km = run.configuration.solar_radius_km
    unsupported = result.unsupported_gap_mask

    def without_unsupported(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        output[unsupported] = np.nan
        return output

    return (
        without_unsupported(result.fitted_height),
        without_unsupported(result.fitted_height_sigma),
        without_unsupported(result.speed * solar_radius_km),
        without_unsupported(result.speed_sigma * solar_radius_km),
        without_unsupported(result.acceleration * solar_radius_km * 1000.0),
        without_unsupported(
            result.acceleration_sigma * solar_radius_km * 1000.0
        ),
    )


def build_track_product(run: CMETrackingRun, event_id: str) -> TrackProduct:
    """Build the one-row-per-frame authoritative track table payload."""

    (
        fitted_height,
        fitted_height_sigma,
        speed,
        speed_sigma,
        acceleration,
        acceleration_sigma,
    ) = _kinematic_product_arrays(run)
    raw_height_sigma = run.summary.height_sigma_rsun
    return TrackProduct(
        event_id=event_id,
        frame_number=np.asarray(
            [frame.frame_number for frame in run.sequence.frames], dtype=np.int64
        ),
        elapsed_s=run.sequence.elapsed_seconds.copy(),
        time_utc=run.sequence.observation_times_utc,
        height_raw_rsun=run.summary.height_rsun,
        height_raw_sigma_rsun=raw_height_sigma,
        height_fit_rsun=fitted_height,
        height_fit_sigma_rsun=fitted_height_sigma,
        speed_fit_km_s=speed,
        speed_sigma_km_s=speed_sigma,
        acceleration_fit_m_s2=acceleration,
        acceleration_sigma_m_s2=acceleration_sigma,
        position_angle_deg=run.summary.position_angle_deg,
        angular_width_deg=run.summary.angular_width_deg,
        front_coverage_fraction=run.summary.coverage_fraction,
        confidence=run.summary.confidence,
        quality_mask=_track_quality_masks(run),
        metadata={
            "algorithm": run.configuration.algorithm_name,
            "algorithm_provisional": True,
            "headline_height_definition": (
                f"angular percentile {run.configuration.summary.height_percentile:g}"
            ),
            "cadence_status": (
                run.sequence.time_axis.cadence_status.value
                if run.sequence.time_axis.cadence_status is not None
                else None
            ),
            "solar_radius_km": run.configuration.solar_radius_km,
            "source_kind": run.sequence.source_kind.value,
            "headline_height_outlier_filter": (
                run.headline_height_filter.method_metadata
            ),
        },
    )


def build_front_samples_product(
    run: CMETrackingRun,
    event_id: str,
) -> FrontSamplesProduct:
    """Build long-form retained event support without interpolating gaps."""

    observed = run.front.observed_mask
    support = np.any(observed, axis=0)
    angle_indices = np.flatnonzero(support)
    frame_count = run.sequence.frame_count
    if angle_indices.size == 0:
        empty_float = np.array([], dtype=np.float64)
        return FrontSamplesProduct(
            event_id=event_id,
            frame_number=np.array([], dtype=np.int64),
            elapsed_s=empty_float,
            position_angle_deg=empty_float,
            radius_rsun=empty_float,
            radius_sigma_rsun=empty_float,
            score=empty_float,
            accepted=np.array([], dtype=bool),
            quality_mask=np.array([], dtype=np.uint32),
            metadata={
                "algorithm": run.configuration.algorithm_name,
                "scenario_id": run.sequence.scenario_id,
                "source_kind": run.sequence.source_kind.value,
            },
        )

    frame_numbers = np.asarray(
        [frame.frame_number for frame in run.sequence.frames], dtype=np.int64
    )
    radius = run.front.radius_rsun[:, angle_indices]
    sigma = (
        run.front.radial_sigma_px[:, angle_indices]
        / run.sequence.geometry.solar_radius_px
    )
    score = run.front.score[:, angle_indices]
    states = run.front.state[:, angle_indices]
    sample_masks = np.full(states.shape, int(_base_quality_mask(run)), dtype=np.uint32)
    sample_masks[states == int(FrontState.MISSING)] |= int(QualityFlag.TRACK_GAP)
    sample_masks[states == int(FrontState.REJECTED)] |= int(QualityFlag.LOW_CONFIDENCE)
    sample_masks[~run.likelihood.supported_frame_mask, :] |= int(
        QualityFlag.TEMPORAL_EVIDENCE_UNSUPPORTED
    )

    return FrontSamplesProduct(
        event_id=event_id,
        frame_number=np.repeat(frame_numbers, angle_indices.size),
        elapsed_s=np.repeat(run.sequence.elapsed_seconds, angle_indices.size),
        position_angle_deg=np.tile(
            run.front.position_angle_deg[angle_indices], frame_count
        ),
        radius_rsun=radius.reshape(-1),
        radius_sigma_rsun=sigma.reshape(-1),
        score=score.reshape(-1),
        accepted=(states == int(FrontState.OBSERVED)).reshape(-1),
        quality_mask=sample_masks.reshape(-1),
        metadata={
            "algorithm": run.configuration.algorithm_name,
            "scenario_id": run.sequence.scenario_id,
            "source_kind": run.sequence.source_kind.value,
            "cadence_status": (
                run.sequence.time_axis.cadence_status.value
                if run.sequence.time_axis.cadence_status is not None
                else None
            ),
        },
    )


def _finite_statistic(values: np.ndarray, function) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(function(finite)) if finite.size else None


def _circular_mean_deg(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    vector = np.mean(np.exp(1j * np.deg2rad(finite)))
    if abs(vector) < 1e-6:
        return None
    return float(np.mod(np.rad2deg(np.angle(vector)), 360.0))


def _public_input_provenance(sequence: ImageSequence) -> dict[str, Any]:
    """Remove host-specific absolute path prefixes from publishable products."""

    values = sequence.provenance_dict()
    manifest_path = values.get("manifest_path")
    if manifest_path:
        values["manifest_path"] = Path(str(manifest_path)).name
    issues = values.get("issues", [])
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict) and issue.get("path"):
                issue["path"] = Path(str(issue["path"])).name
    frames = values.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict) and frame.get("path"):
                frame["path"] = Path(str(frame["path"])).name
    verification = sequence.hash_verification_status
    if verification == "verified_at_load_and_read":
        values["path_policy"] = (
            "Host-specific prefixes removed; the reviewed manifest, manifest "
            "SHA-256, and verified per-frame SHA-256 values identify the inputs."
        )
    elif verification == "skipped_by_request":
        values["path_policy"] = (
            "Host-specific prefixes removed; the reviewed manifest declares "
            "per-frame SHA-256 values, but verification was explicitly skipped."
        )
    else:
        values["path_policy"] = (
            "Host-specific prefixes removed; direct-directory development input "
            "was not identified by a reviewed manifest or content hashes."
        )
    return values


def build_event_summary(
    run: CMETrackingRun,
    event_id: str,
    *,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Build JSON summary and full reproducibility provenance."""

    configuration = run.configuration.to_dict()
    provenance = collect_run_provenance(
        repository=repository,
        configuration=configuration,
    )
    measured = np.isfinite(run.summary.height_rsun)
    uncensored = measured & ~run.field_of_view_limited_mask
    measured_indices = np.flatnonzero(measured)
    kinematics = run.kinematics
    speed = (
        np.where(
            kinematics.valid_mask,
            kinematics.speed * run.configuration.solar_radius_km,
            np.nan,
        )
        if kinematics is not None
        else np.array([], dtype=np.float64)
    )
    acceleration = (
        np.where(
            kinematics.valid_mask,
            kinematics.acceleration
            * run.configuration.solar_radius_km
            * 1000.0,
            np.nan,
        )
        if kinematics is not None
        else np.array([], dtype=np.float64)
    )
    fit_indices = (
        np.flatnonzero(kinematics.valid_mask)
        if kinematics is not None
        else np.array([], dtype=np.int64)
    )
    event_mask = _base_quality_mask(run)
    if not run.front.event_detected:
        event_mask |= QualityFlag.FRONT_NOT_DETECTED
    if np.any(run.field_of_view_limited_mask):
        event_mask |= QualityFlag.PARTIAL_FIELD_OF_VIEW
    if kinematics is None:
        event_mask |= QualityFlag.KINEMATICS_NOT_FIT
    if np.any(run.headline_height_filter.outlier_mask):
        event_mask |= QualityFlag.KINEMATIC_HEIGHT_OUTLIER

    height_filter_metadata = dict(run.headline_height_filter.method_metadata)
    height_filter_metadata["rejected_frame_numbers"] = [
        int(run.sequence.frames[index].frame_number)
        for index in np.flatnonzero(run.headline_height_filter.outlier_mask)
    ]
    scientific_caveats = [
        "This is a provisional known-window research product, not continuous event discovery.",
        (
            "Headline height is a robust angular percentile pending approval "
            "of the mission convention."
        ),
        "Projected plane-of-sky kinematics are not deprojected three-dimensional CME motion.",
        (
            "Field-of-view-limited heights are retained as censored "
            "observations but excluded from maximum height and kinematic fits."
        ),
        (
            "Front association, central position angle, angular width, and "
            "headline-height convention remain scientifically provisional."
        ),
    ]
    if run.configuration.kinematics.height_outlier_filter_enabled:
        scientific_caveats.append(
            "The kinematic fit excludes flagged temporal headline-height "
            "outliers without replacing or modifying the raw measurements."
        )
    unsupported_evidence_count = int(
        np.count_nonzero(~run.likelihood.supported_frame_mask)
    )
    if unsupported_evidence_count:
        scientific_caveats.append(
            "Centered temporal filtering leaves global endpoint frames "
            "unsupported rather than padding or shifting their timestamps."
        )

    return {
        "event_id": event_id,
        "scenario_id": run.sequence.scenario_id,
        "event_detected": run.front.event_detected,
        "known_window": True,
        "continuous_event_discovery_performed": False,
        "event_start_elapsed_s": (
            float(run.sequence.elapsed_seconds[measured_indices[0]])
            if measured_indices.size
            else None
        ),
        "event_end_elapsed_s": (
            float(run.sequence.elapsed_seconds[measured_indices[-1]])
            if measured_indices.size
            else None
        ),
        "kinematics_start_elapsed_s": (
            float(run.sequence.elapsed_seconds[fit_indices[0]])
            if fit_indices.size
            else None
        ),
        "kinematics_end_elapsed_s": (
            float(run.sequence.elapsed_seconds[fit_indices[-1]])
            if fit_indices.size
            else None
        ),
        "central_position_angle_deg": _circular_mean_deg(
            run.summary.position_angle_deg
        ),
        "median_angular_width_deg": _finite_statistic(
            run.summary.angular_width_deg, np.median
        ),
        "minimum_projected_height_rsun": _finite_statistic(
            np.where(uncensored, run.summary.height_rsun, np.nan), np.min
        ),
        "maximum_projected_height_rsun": _finite_statistic(
            np.where(uncensored, run.summary.height_rsun, np.nan), np.max
        ),
        "maximum_observed_height_including_fov_limited_rsun": _finite_statistic(
            run.summary.height_rsun, np.max
        ),
        "median_projected_speed_km_s": _finite_statistic(speed, np.median),
        "maximum_projected_speed_km_s": _finite_statistic(speed, np.max),
        "median_projected_acceleration_m_s2": _finite_statistic(
            acceleration, np.median
        ),
        "quality_mask": int(event_mask),
        "quality_flags": list(decode_quality_flags(event_mask)),
        "tracker_quality_flags": list(run.front.quality_flags),
        "field_of_view_limited_frame_count": int(
            np.count_nonzero(run.field_of_view_limited_mask)
        ),
        "temporal_evidence_unsupported_frame_count": unsupported_evidence_count,
        "kinematics_error": run.kinematics_error,
        "headline_height_outlier_filter": height_filter_metadata,
        "kinematics_method": (
            kinematics.method_metadata if kinematics is not None else None
        ),
        "input": _public_input_provenance(run.sequence),
        "run_provenance": provenance,
        "scientific_caveats": scientific_caveats,
    }


def build_front_overlay_product(
    run: CMETrackingRun,
    *,
    maximum_frames: int = 6,
) -> FrontOverlayProduct:
    """Load a few source frames and overlay the exact retained measurements."""

    count = min(maximum_frames, run.sequence.frame_count)
    indices = np.unique(
        np.linspace(0, run.sequence.frame_count - 1, count).round().astype(int)
    )
    frames: list[FrontOverlayFrame] = []
    for index in indices:
        observed = run.front.observed_mask[index]
        frames.append(
            FrontOverlayFrame(
                image=run.sequence.read_frame(int(index)).data,
                frame_number=run.sequence.frames[index].frame_number,
                elapsed_s=float(run.sequence.elapsed_seconds[index]),
                radius_px=run.front.radius_px[index, observed],
                position_angle_deg=run.front.position_angle_deg[observed],
                headline_height_rsun=float(run.summary.height_rsun[index]),
            )
        )
    return FrontOverlayProduct(
        frames=tuple(frames),
        center_yx=run.sequence.geometry.center_yx,
        north_vector_yx=run.sequence.geometry.north_vector_yx,
        east_vector_yx=run.sequence.geometry.east_vector_yx,
    )


def write_known_window_products(
    run: CMETrackingRun,
    output_root: str | Path,
    event_id: str,
    *,
    repository: str | Path | None = None,
    include_diagnostic_movie: bool = False,
    movie_fps: float = 10.0,
    overwrite: bool = False,
) -> Path:
    """Write the provisional product directory for a completed run."""

    track = build_track_product(run, event_id)
    front_samples = build_front_samples_product(run, event_id)
    summary = build_event_summary(run, event_id, repository=repository)
    overlay = build_front_overlay_product(run)

    diagnostic_movie_writer = None
    diagnostic_movie_metadata = None
    if include_diagnostic_movie:
        if not np.isfinite(movie_fps) or movie_fps <= 0:
            raise ValueError("movie_fps must be finite and positive.")

        def diagnostic_movie_writer(path: Path) -> None:
            write_cme_tracking_movie(run, track, path, fps=float(movie_fps))

        diagnostic_movie_metadata = {
            "frame_count": run.sequence.frame_count,
            "frames": "all input frames in sequence order",
            "fps": float(movie_fps),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "timing_policy": (
                "One input image per movie frame; playback speed does not alter "
                "the scientific elapsed-time axis."
            ),
        }
    return write_event_products(
        output_root,
        track,
        front_samples,
        summary,
        front_overlay=overlay,
        diagnostic_movie_writer=diagnostic_movie_writer,
        diagnostic_movie_metadata=diagnostic_movie_metadata,
        overwrite=overwrite,
    )
