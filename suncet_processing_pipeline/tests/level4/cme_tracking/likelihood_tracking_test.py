"""Analytic tests for deterministic CME evidence and temporal tracking."""

from dataclasses import replace

import numpy as np
import pytest

from suncet_processing_pipeline.level4.cme_tracking.input import SequenceGeometry
from suncet_processing_pipeline.level4.cme_tracking.likelihood import (
    EvidenceConfig,
    LikelihoodError,
    _centered_temporal_nanmedian,
    score_front_likelihood,
)
from suncet_processing_pipeline.level4.cme_tracking.polar import (
    PolarConfig,
    PolarGrid,
    build_polar_grid,
    remap_sequence_to_polar,
)
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    FrontState,
    TrackingConfig,
    TrackingError,
    _frame_candidates,
    extract_front,
)


def _analytic_expanding_arc(
    *,
    frame_count=14,
    missing_frames=(),
    seed=947,
):
    """Return images with an exact smooth outer edge and its polar truth."""

    shape = (128, 128)
    center_x = center_y = 63.5
    solar_radius_px = 18.0
    yy, xx = np.indices(shape, dtype=np.float64)
    dx = xx - center_x
    dy = yy - center_y
    radius_rsun = np.hypot(dx, dy) / solar_radius_px
    position_angle_deg = (
        np.rad2deg(np.arctan2(-dx, -dy)) % 360.0
    )

    central_pa_deg = 60.0
    half_width_deg = 34.0
    angular_delta_deg = (
        (position_angle_deg - central_pa_deg + 180.0) % 360.0 - 180.0
    )
    angular_coordinate = angular_delta_deg / half_width_deg
    angular_taper = np.where(
        np.abs(angular_coordinate) <= 1.0,
        0.5 * (1.0 + np.cos(np.pi * angular_coordinate)),
        0.0,
    )

    background = 7.0 * np.exp(-np.maximum(radius_rsun - 0.8, 0.0) / 0.8) + 0.5
    inner_gate = 0.5 * (1.0 + np.tanh((radius_rsun - 1.08) / 0.025))
    rng = np.random.default_rng(seed)
    images = []
    truth_by_frame = []
    for frame_index in range(frame_count):
        apex_rsun = 1.30 + 0.045 * frame_index
        front_rsun = apex_rsun - 0.07 * angular_coordinate**2
        outer_edge = 0.5 * (1.0 - np.tanh((radius_rsun - front_rsun) / 0.025))
        feature = 8.0 * angular_taper * inner_gate * outer_edge
        if frame_index in missing_frames:
            feature = np.zeros_like(feature)
        image = background + feature + rng.normal(0.0, 0.025, shape)
        images.append(image.astype(np.float32))
        truth_by_frame.append(apex_rsun)

    geometry = SequenceGeometry(
        image_shape_yx=shape,
        center_x_px=center_x,
        center_y_px=center_y,
        solar_radius_px=solar_radius_px,
        pixel_scales_arcsec_xy=None,
        pixel_scale_arcsec=None,
        pixel_scale_anisotropy_fraction=None,
        north_vector_yx=(-1.0, 0.0),
        east_vector_yx=(0.0, -1.0),
        orientation_source="explicit_override",
    )
    return (
        np.stack(images),
        geometry,
        np.asarray(truth_by_frame),
        central_pa_deg,
    )


def _run_tracker(images, geometry, *, temporal_median_window_frames=1):
    grid = build_polar_grid(
        geometry,
        PolarConfig(
            position_angle_step_deg=4.0,
            radial_step_px=1.0,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
    )
    polar = remap_sequence_to_polar(images, grid)
    likelihood = score_front_likelihood(
        polar,
        grid,
        EvidenceConfig(
            temporal_median_window_frames=temporal_median_window_frames,
            smooth_position_angle_deg=3.0,
            smooth_radius_px=0.8,
            minimum_leading_edge_z=0.5,
        ),
    )
    front = extract_front(
        likelihood,
        grid,
        TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=4.0,
            inward_localization_tolerance_px=1.5,
            maximum_gap_frames=2,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            angular_outlier_tolerance_px=5.0,
            minimum_event_frames=6,
            minimum_observed_angles_per_frame=3,
        ),
    )
    return grid, likelihood, front


def _fully_valid_polar_grid(*, angle_count=36, radius_count=64):
    radius_px = np.arange(1.0, radius_count + 1.0)
    return PolarGrid(
        position_angle_deg=np.arange(angle_count, dtype=float)
        * (360.0 / angle_count),
        radius_px=radius_px,
        radius_rsun=radius_px / 10.0,
        sample_y_px=np.zeros((angle_count, radius_count)),
        sample_x_px=np.zeros((angle_count, radius_count)),
        valid_mask=np.ones((angle_count, radius_count), dtype=bool),
        image_shape_yx=(128, 128),
        interpolation_order=1,
    )


@pytest.mark.parametrize("window", [0, 2, 4, 1.5, True, np.bool_(False)])
def test_temporal_median_window_must_be_a_positive_odd_integer(window):
    with pytest.raises(LikelihoodError, match="positive odd integer"):
        EvidenceConfig(temporal_median_window_frames=window)


def test_centered_temporal_median_removes_impulse_and_preserves_two_frames():
    values = np.ones((7, 3), dtype=np.float64)
    values[3, 0] = 101.0
    values[3:5, 1] = 9.0
    values[:, 2] = np.nan
    original = values.copy()

    filtered = _centered_temporal_nanmedian(values, 3)

    np.testing.assert_array_equal(values, original)
    assert np.all(np.isnan(filtered[[0, -1]]))
    np.testing.assert_array_equal(filtered[1:-1, 0], np.ones(5))
    np.testing.assert_array_equal(filtered[3:5, 1], np.full(2, 9.0))
    assert np.all(np.isnan(filtered[:, 2]))


def test_centered_temporal_median_requires_a_finite_majority():
    values = np.array(
        [
            [[1.0, np.nan]],
            [[3.0, 7.0]],
            [[np.nan, np.nan]],
        ]
    )

    filtered = _centered_temporal_nanmedian(values, 3)

    # Two finite samples are a strict majority of a three-frame window.
    assert filtered[1, 0, 0] == 2.0
    assert np.isnan(filtered[1, 0, 1])


def test_window_one_is_bitwise_identical_to_the_default_likelihood():
    rng = np.random.default_rng(20260827)
    grid = _fully_valid_polar_grid(angle_count=12, radius_count=20)
    values = rng.normal(size=(7, *grid.shape))
    default = score_front_likelihood(
        values,
        grid,
        EvidenceConfig(retain_diagnostics=True),
    )
    explicit = score_front_likelihood(
        values,
        grid,
        EvidenceConfig(
            temporal_median_window_frames=1,
            retain_diagnostics=True,
        ),
    )

    for name in (
        "score",
        "temporal_background",
        "supported_frame_mask",
        "base_excess_z",
        "running_difference_z",
        "leading_edge_z",
    ):
        np.testing.assert_array_equal(
            getattr(explicit, name),
            getattr(default, name),
        )


def test_temporal_median_suppresses_isolated_particle_likelihood():
    frame_count = 20
    grid = _fully_valid_polar_grid()
    values = np.ones((frame_count, *grid.shape), dtype=np.float64)
    for frame_index in range(1, frame_count - 1):
        angle_index = (7 * frame_index) % grid.shape[0]
        radius_index = 8 + (11 * frame_index) % (grid.shape[1] - 16)
        values[frame_index, angle_index, radius_index] = 101.0
    common = dict(
        smooth_position_angle_deg=0.0,
        smooth_radius_px=0.0,
        retain_diagnostics=True,
    )

    raw = score_front_likelihood(values, grid, EvidenceConfig(**common))
    filtered = score_front_likelihood(
        values,
        grid,
        EvidenceConfig(temporal_median_window_frames=3, **common),
    )
    threshold = 2.5
    raw_count = np.count_nonzero(raw.score >= threshold)
    filtered_count = np.count_nonzero(filtered.score >= threshold)

    assert raw_count > 0
    assert filtered_count <= 0.05 * raw_count
    np.testing.assert_array_equal(
        filtered.supported_frame_mask,
        np.array([False, *([True] * (frame_count - 2)), False]),
    )
    for channel in (
        filtered.score,
        filtered.base_excess_z,
        filtered.running_difference_z,
        filtered.leading_edge_z,
    ):
        assert np.all(np.isnan(channel[[0, -1]]))
    assert np.all(np.isfinite(filtered.score[1:-1]))
    assert np.nanmax(filtered.score) == 0.0


def test_temporal_median_preserves_the_analytic_broad_expanding_front():
    images, geometry, truth_rsun, central_pa_deg = _analytic_expanding_arc(
        frame_count=18
    )
    raw_grid, _, raw_front = _run_tracker(images, geometry)
    filtered_grid, filtered_likelihood, filtered_front = _run_tracker(
        images,
        geometry,
        temporal_median_window_frames=3,
    )
    np.testing.assert_array_equal(filtered_grid.radius_px, raw_grid.radius_px)
    central_angle_index = int(
        np.argmin(np.abs(raw_grid.position_angle_deg - central_pa_deg))
    )
    raw_radius = raw_front.radius_px[:, central_angle_index]
    filtered_radius = filtered_front.radius_px[:, central_angle_index]
    common = np.isfinite(raw_radius) & np.isfinite(filtered_radius)

    assert filtered_front.event_detected
    assert np.count_nonzero(np.isfinite(filtered_radius)) >= 14
    assert np.count_nonzero(common) >= 14
    difference_px = np.abs(filtered_radius[common] - raw_radius[common])
    assert np.median(difference_px) <= 1.0
    assert np.max(difference_px) <= 2.0
    filtered_truth_error = np.abs(
        filtered_radius / geometry.solar_radius_px - truth_rsun
    )
    assert np.nanmedian(filtered_truth_error) < 0.12
    assert np.all(np.isnan(filtered_likelihood.score[[0, -1]]))


def test_fast_thin_front_stress_case_documents_median_loss():
    """A same-pixel median cannot preserve a one-frame, fast thin shell."""

    values = np.zeros((7, 1, 40), dtype=np.float64)
    for frame_index in range(values.shape[0]):
        values[frame_index, 0, 3 + 5 * frame_index] = 1.0

    filtered = _centered_temporal_nanmedian(values, 3)

    assert np.nanmax(filtered) == 0.0


def test_expanding_arc_is_tracked_outward_at_the_correct_radius():
    images, geometry, truth_rsun, central_pa_deg = _analytic_expanding_arc()
    grid, likelihood, front = _run_tracker(images, geometry)

    assert np.nanmax(likelihood.score) > 1.5
    assert front.event_detected
    central_angle_index = int(
        np.argmin(np.abs(grid.position_angle_deg - central_pa_deg))
    )
    measured = front.radius_rsun[:, central_angle_index]
    observed = np.isfinite(measured)

    assert np.count_nonzero(observed) >= 10
    assert np.nanmedian(np.diff(measured[observed])) > 0.0
    assert np.nanmedian(np.abs(measured[observed] - truth_rsun[observed])) < 0.12


def test_static_sequence_has_no_front_event():
    images, geometry, _, _ = _analytic_expanding_arc()
    static_images = np.repeat(images[:1], images.shape[0], axis=0)
    _, likelihood, front = _run_tracker(static_images, geometry)

    assert np.nanmax(likelihood.score) == 0.0
    assert not front.event_detected
    assert "NO_EVENT" in front.quality_flags
    assert np.all(np.isnan(front.radius_rsun))


def test_short_missing_frame_remains_nan_and_path_reconnects():
    missing_frame = 7
    images, geometry, _, central_pa_deg = _analytic_expanding_arc(
        missing_frames=(missing_frame,)
    )
    grid, _, front = _run_tracker(images, geometry)
    central_angle_index = int(
        np.argmin(np.abs(grid.position_angle_deg - central_pa_deg))
    )

    assert front.event_detected
    assert np.isnan(front.radius_px[missing_frame, central_angle_index])
    assert (
        front.state[missing_frame, central_angle_index]
        == int(FrontState.MISSING)
    )
    assert np.isfinite(front.radius_px[missing_frame - 1, central_angle_index])
    assert np.isfinite(front.radius_px[missing_frame + 1, central_angle_index])


def _nominal_gap_tracking_case():
    """Return a sparse score cube whose middle jump needs two frame intervals."""

    angle_count = 4
    radius_px = np.arange(1.0, 9.0)
    grid = PolarGrid(
        position_angle_deg=np.arange(angle_count, dtype=float) * 90.0,
        radius_px=radius_px,
        radius_rsun=radius_px / 10.0,
        sample_y_px=np.zeros((angle_count, radius_px.size)),
        sample_x_px=np.zeros((angle_count, radius_px.size)),
        valid_mask=np.ones((angle_count, radius_px.size), dtype=bool),
        image_shape_yx=(16, 16),
        interpolation_order=1,
    )
    score = np.zeros((4, angle_count, radius_px.size), dtype=float)
    for frame_index, radius_index in enumerate((1, 2, 4, 5)):
        score[frame_index, :, radius_index] = 6.0
    config = TrackingConfig(
        score_threshold=1.0,
        maximum_candidates_per_frame=1,
        minimum_peak_separation_bins=1,
        maximum_outward_step_px_per_frame=1.1,
        inward_localization_tolerance_px=0.0,
        maximum_gap_frames=1,
        minimum_track_points=4,
        minimum_outward_displacement_px=3.5,
        minimum_angular_support_deg=180.0,
        minimum_event_frames=4,
        minimum_observed_angles_per_frame=3,
    )
    return score, grid, config


def test_candidate_fast_reject_skips_peak_search_below_threshold(monkeypatch):
    _, grid, config = _nominal_gap_tracking_case()

    def unexpected_peak_search(*_args, **_kwargs):
        raise AssertionError("find_peaks must not run below the score threshold")

    monkeypatch.setattr(
        "suncet_processing_pipeline.level4.cme_tracking.tracking.find_peaks",
        unexpected_peak_search,
    )

    assert _frame_candidates(
        np.array([np.nan, 0.25, 0.99, -1.0]),
        grid,
        config,
    ) == ()


def test_candidate_at_threshold_and_endpoint_survives_fast_reject():
    _, grid, config = _nominal_gap_tracking_case()
    score = np.zeros(grid.radius_px.size, dtype=float)
    score[-1] = config.score_threshold

    candidates = _frame_candidates(score, grid, config)

    assert len(candidates) == 1
    assert candidates[0].radius_index == score.size - 1
    assert candidates[0].score == config.score_threshold


def test_nominal_frame_gap_scales_motion_budget_and_missing_frame_limit():
    score, grid, config = _nominal_gap_tracking_case()

    contiguous = extract_front(score, grid, config)
    one_missing_frame = extract_front(
        score,
        grid,
        config,
        frame_numbers=(0, 1, 3, 4),
    )
    two_missing_frames = extract_front(
        score,
        grid,
        config,
        frame_numbers=(0, 1, 4, 5),
    )

    # Treating the rows as contiguous gives only one frame interval for the
    # two-pixel middle jump, while the reviewed manifest coordinate gives it
    # two. A gap larger than maximum_gap_frames still cannot reconnect.
    assert not contiguous.event_detected
    assert one_missing_frame.event_detected
    assert np.all(one_missing_frame.observed_mask)
    assert not two_missing_frames.event_detected


def test_nominal_frame_gap_applies_penalty_per_missing_interval():
    score, grid, config = _nominal_gap_tracking_case()
    penalty_dominated = replace(
        config,
        maximum_outward_step_px_per_frame=3.0,
        gap_penalty=20.0,
    )

    contiguous = extract_front(score, grid, penalty_dominated)
    one_missing_frame = extract_front(
        score,
        grid,
        penalty_dominated,
        frame_numbers=(0, 1, 3, 4),
    )

    assert contiguous.event_detected
    assert not one_missing_frame.event_detected


def test_explicit_contiguous_frame_numbers_match_legacy_default():
    score, grid, config = _nominal_gap_tracking_case()
    contiguous_config = replace(
        config,
        maximum_outward_step_px_per_frame=3.0,
    )

    implicit = extract_front(score, grid, contiguous_config)
    explicit = extract_front(
        score,
        grid,
        contiguous_config,
        frame_numbers=np.arange(score.shape[0]),
    )

    np.testing.assert_allclose(
        explicit.radius_px,
        implicit.radius_px,
        equal_nan=True,
    )
    np.testing.assert_array_equal(explicit.state, implicit.state)
    assert explicit.event_detected == implicit.event_detected
    assert explicit.quality_flags == implicit.quality_flags


def test_position_angle_window_excludes_paths_outside_reviewed_sector():
    score, grid, config = _nominal_gap_tracking_case()
    windowed = replace(
        config,
        maximum_outward_step_px_per_frame=3.0,
        minimum_angular_support_deg=90.0,
        minimum_observed_angles_per_frame=2,
        position_angle_window_deg=(80.0, 190.0),
    )

    front = extract_front(score, grid, windowed)

    assert front.event_detected
    np.testing.assert_array_equal(
        np.any(front.observed_mask, axis=0),
        np.array([False, True, True, False]),
    )
    assert "POSITION_ANGLE_WINDOW_APPLIED" in front.quality_flags


def test_position_angle_window_wraps_through_zero_degrees():
    score, grid, config = _nominal_gap_tracking_case()
    windowed = replace(
        config,
        maximum_outward_step_px_per_frame=3.0,
        minimum_angular_support_deg=180.0,
        position_angle_window_deg=(260.0, 100.0),
    )

    front = extract_front(score, grid, windowed)

    assert front.event_detected
    np.testing.assert_array_equal(
        np.any(front.observed_mask, axis=0),
        np.array([True, True, False, True]),
    )


def _coherence_test_grid(angle_count=24, radius_count=50):
    radius_px = np.arange(1.0, radius_count + 1.0)
    return PolarGrid(
        position_angle_deg=np.arange(angle_count, dtype=float) * 360.0 / angle_count,
        radius_px=radius_px,
        radius_rsun=radius_px / 10.0,
        sample_y_px=np.zeros((angle_count, radius_count)),
        sample_x_px=np.zeros((angle_count, radius_count)),
        valid_mask=np.ones((angle_count, radius_count), dtype=bool),
        image_shape_yx=(128, 128),
        interpolation_order=1,
    )


def test_automatic_coherent_fragments_track_arc_and_ignore_brighter_dots():
    grid = _coherence_test_grid()
    frame_count = 10
    score = np.zeros((frame_count, *grid.shape), dtype=float)
    arc_angles = np.arange(8, 14)
    isolated_particle_angles = (0, 3, 6, 16, 19, 22)
    for frame_index in range(frame_count):
        front_radius_index = 9 + frame_index
        score[frame_index, arc_angles, front_radius_index] = 5.0
        particle_angle = isolated_particle_angles[
            frame_index % len(isolated_particle_angles)
        ]
        score[frame_index, particle_angle, 34 + frame_index % 5] = 10.0

    config = TrackingConfig(
        score_threshold=1.0,
        maximum_candidates_per_frame=2,
        maximum_outward_step_px_per_frame=2.0,
        inward_localization_tolerance_px=1.0,
        maximum_gap_frames=1,
        minimum_track_points=6,
        minimum_outward_displacement_px=5.0,
        minimum_angular_support_deg=45.0,
        angular_outlier_tolerance_px=2.0,
        minimum_event_frames=6,
        minimum_observed_angles_per_frame=3,
    )

    front = extract_front(score, grid, config)

    assert front.event_detected
    assert "POSITION_ANGLE_WINDOW_APPLIED" not in front.quality_flags
    np.testing.assert_array_equal(
        np.flatnonzero(np.any(front.observed_mask, axis=0)),
        arc_angles,
    )
    expected_radius = np.arange(10.0, 20.0)
    np.testing.assert_allclose(
        front.radius_px[:, arc_angles],
        np.repeat(expected_radius[:, None], arc_angles.size, axis=1),
    )


def test_spatially_incoherent_particle_paths_fail_automatic_association():
    grid = _coherence_test_grid(angle_count=12, radius_count=50)
    frame_count = 8
    score = np.zeros((frame_count, *grid.shape), dtype=float)
    for frame_index in range(frame_count):
        for angle_index in range(grid.position_angle_deg.size):
            # Every PA contains a convincing outward temporal path, but
            # adjacent dots alternate between widely separated radii and do
            # not form a simultaneous physical front.
            radius_index = 4 + 20 * (angle_index % 2) + frame_index
            score[frame_index, angle_index, radius_index] = 8.0

    common = TrackingConfig(
        score_threshold=1.0,
        maximum_candidates_per_frame=1,
        maximum_outward_step_px_per_frame=2.0,
        inward_localization_tolerance_px=0.0,
        maximum_gap_frames=1,
        minimum_track_points=5,
        minimum_outward_displacement_px=4.0,
        minimum_angular_support_deg=60.0,
        angular_consistency_half_width_deg=6.0,
        angular_outlier_tolerance_px=2.0,
        minimum_angular_neighbors=4,
        minimum_event_frames=5,
        minimum_observed_angles_per_frame=3,
    )

    automatic = extract_front(score, grid, common)
    historical = extract_front(
        score,
        grid,
        replace(common, association_method="independent_angle_paths"),
    )

    assert not automatic.event_detected
    assert not np.any(automatic.observed_mask)
    assert "INSUFFICIENT_SPATIOTEMPORAL_COHERENCE" in automatic.quality_flags
    # This control demonstrates the exact false-association mode the new
    # ordering removes: independent PA paths can merge despite radial
    # incoherence in every individual frame.
    assert historical.event_detected


def test_coherent_path_cannot_accumulate_sustained_inward_drift():
    grid = _coherence_test_grid(radius_count=70)
    frame_count = 16
    score = np.zeros((frame_count, *grid.shape), dtype=float)
    arc_angles = np.arange(8, 14)

    # A valid outward front rises from 10 to 31 px. A brighter unrelated
    # structure then retreats one pixel per frame. Every local inward step is
    # within tolerance, but the sequence must not be allowed to walk far below
    # the maximum radius already reached by the event.
    for frame_index in range(8):
        radius_px = 10 + 3 * frame_index
        score[frame_index, arc_angles, radius_px - 1] = 5.0
    for frame_index in range(8, frame_count):
        radius_px = 30 - (frame_index - 8)
        score[frame_index, arc_angles, radius_px - 1] = 9.0

    config = TrackingConfig(
        score_threshold=1.0,
        maximum_candidates_per_frame=1,
        maximum_outward_step_px_per_frame=4.0,
        inward_localization_tolerance_px=2.0,
        maximum_gap_frames=1,
        minimum_track_points=6,
        minimum_outward_displacement_px=10.0,
        minimum_angular_support_deg=45.0,
        angular_outlier_tolerance_px=2.0,
        minimum_event_frames=6,
        minimum_observed_angles_per_frame=3,
    )

    front = extract_front(score, grid, config)

    assert front.event_detected
    observed_frames = np.flatnonzero(np.any(front.observed_mask, axis=1))
    np.testing.assert_array_equal(observed_frames, np.arange(10))
    median_radius = np.nanmedian(front.radius_px[observed_frames], axis=1)
    running_maximum = np.maximum.accumulate(median_radius)
    assert np.all(
        median_radius
        >= running_maximum - config.inward_localization_tolerance_px
    )


@pytest.mark.parametrize(
    ("frame_numbers", "message"),
    (
        ((0, 1, 2), "one integer per likelihood frame"),
        ((0, 1.5, 2, 3), "must contain integers"),
        ((0, True, 2, 3), "must contain integers"),
        ((0, 2, 2, 3), "strictly increasing"),
    ),
)
def test_tracking_rejects_invalid_nominal_frame_numbers(frame_numbers, message):
    score, grid, config = _nominal_gap_tracking_case()

    with pytest.raises(TrackingError, match=message):
        extract_front(
            score,
            grid,
            config,
            frame_numbers=frame_numbers,
        )
