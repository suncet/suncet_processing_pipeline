"""Acceptance tests for full-circle, coherence-first CME association.

These fixtures operate on an already constructed likelihood cube so they
exercise association independently of image remapping and evidence scaling.
No fixture supplies a position-angle window: the event sector must emerge
from simultaneous angular continuity and its outward temporal evolution.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.pipeline import (
    _field_of_view_limited_frames,
)
from suncet_processing_pipeline.level4.cme_tracking.polar import PolarGrid
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    FrontState,
    FrontTrack,
    TrackingConfig,
    extract_front,
)


_ANGLE_COUNT = 36
_ANGLE_STEP_DEG = 360.0 / _ANGLE_COUNT
_RADIUS_COUNT = 64


def _polar_grid() -> PolarGrid:
    radius_px = np.arange(1, _RADIUS_COUNT + 1, dtype=np.float64)
    shape = (_ANGLE_COUNT, _RADIUS_COUNT)
    return PolarGrid(
        position_angle_deg=np.arange(_ANGLE_COUNT, dtype=np.float64)
        * _ANGLE_STEP_DEG,
        radius_px=radius_px,
        radius_rsun=radius_px / 20.0,
        sample_y_px=np.zeros(shape, dtype=np.float64),
        sample_x_px=np.zeros(shape, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        image_shape_yx=(160, 160),
        interpolation_order=1,
    )


def _tracking_config() -> TrackingConfig:
    return TrackingConfig(
        score_threshold=1.0,
        maximum_candidates_per_frame=2,
        minimum_peak_separation_bins=1,
        maximum_outward_step_px_per_frame=2.5,
        inward_localization_tolerance_px=0.5,
        maximum_gap_frames=2,
        minimum_track_points=5,
        minimum_outward_displacement_px=4.0,
        maximum_angular_gap_bins=1,
        minimum_angular_support_deg=40.0,
        angular_consistency_half_width_deg=20.0,
        angular_outlier_tolerance_px=3.0,
        minimum_angular_neighbors=3,
        minimum_event_frames=6,
        minimum_observed_angles_per_frame=4,
        position_angle_window_deg=None,
    )


def _coherent_arc_score(
    *,
    center_angle_bin: int,
    frame_count: int = 16,
    missing_frames: tuple[int, ...] = (),
    include_isolated_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth eight-angle front plus optional isolated impulses."""

    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    offsets = np.arange(-4, 4, dtype=np.int64)
    arc_angles = (center_angle_bin + offsets) % _ANGLE_COUNT
    for frame_index in range(frame_count):
        if frame_index in missing_frames:
            continue
        for offset, angle_index in zip(offsets, arc_angles, strict=True):
            # A stable, gently curved front moving one radial pixel per frame.
            curvature_px = int(round(0.12 * float(offset) ** 2))
            radius_index = 10 + frame_index - curvature_px
            score[frame_index, angle_index, radius_index] = 8.0

        if include_isolated_noise:
            # High-scoring impulses deliberately appear outside the arc, but
            # neither recur soon enough at one PA nor form a simultaneous arc.
            noise_angle = (center_angle_bin + 13 + 5 * frame_index) % _ANGLE_COUNT
            noise_radius = 42 + (7 * frame_index) % 15
            score[frame_index, noise_angle, noise_radius] = 10.0
    return score, arc_angles


def _front_with_single_boundary_impulse(grid: PolarGrid) -> FrontTrack:
    """Return a coherent inner front with one unrelated outer-FOV sample."""

    frame_count = 12
    radius_px = np.full((frame_count, _ANGLE_COUNT), np.nan)
    sigma_px = np.full_like(radius_px, np.nan)
    score = np.full_like(radius_px, np.nan)
    state = np.full(
        radius_px.shape,
        int(FrontState.MISSING),
        dtype=np.uint8,
    )
    arc_angles = np.arange(10, 18)
    for frame_index in range(frame_count):
        radii = 30.0 + 0.4 * frame_index + np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        )
        radius_px[frame_index, arc_angles] = radii
        sigma_px[frame_index, arc_angles] = 0.5
        score[frame_index, arc_angles] = 6.0
        state[frame_index, arc_angles] = int(FrontState.OBSERVED)

    # A single high-radius sample is exactly the failure mode that must not
    # start a permanent FOV-limited latch.
    radius_px[4, 25] = grid.radius_px[-1]
    sigma_px[4, 25] = 0.5
    score[4, 25] = 10.0
    state[4, 25] = int(FrontState.OBSERVED)
    return FrontTrack(
        radius_px=radius_px,
        radius_rsun=radius_px / 20.0,
        radial_sigma_px=sigma_px,
        score=score,
        state=state,
        position_angle_deg=grid.position_angle_deg.copy(),
        event_detected=True,
        quality_flags=(),
    )


def test_staggered_isolated_particle_paths_do_not_create_an_event() -> None:
    """Independent temporal paths must not masquerade as one global front."""

    grid = _polar_grid()
    frame_count = 24
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260827)
    for angle_index in range(_ANGLE_COUNT):
        # Even and odd angles occupy disjoint time blocks. Every PA therefore
        # contains a valid-looking outward path, but no adjacent PAs form a
        # simultaneous front fragment.
        first_frame = 1 if angle_index % 2 == 0 else 13
        first_radius = 8 + int(rng.integers(0, 8))
        for local_index in range(6):
            score[
                first_frame + local_index,
                angle_index,
                first_radius + local_index,
            ] = 6.0 + float(rng.uniform(0.0, 2.0))

    front = extract_front(score, grid, _tracking_config())

    assert not front.event_detected
    assert "NO_EVENT" in front.quality_flags
    assert not np.any(front.observed_mask)


def test_random_single_frame_particle_hits_do_not_create_an_event() -> None:
    """Bright but temporally impulsive points are not a front."""

    grid = _polar_grid()
    frame_count = 24
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    rng = np.random.default_rng(1942027)
    for frame_index in range(frame_count):
        # Each frame contains three widely separated, independently located
        # impulses. Their scores exceed the front threshold by a wide margin.
        phase = int(rng.integers(0, 4))
        for angle_index in (phase, phase + 12, phase + 24):
            radius_index = int(rng.integers(5, _RADIUS_COUNT - 5))
            score[frame_index, angle_index, radius_index] = float(
                rng.uniform(8.0, 12.0)
            )

    front = extract_front(score, grid, _tracking_config())

    assert not front.event_detected
    assert "NO_EVENT" in front.quality_flags
    assert not np.any(front.observed_mask)


def test_endpoint_excursion_does_not_turn_a_stationary_arc_into_an_event() -> None:
    grid = _polar_grid()
    frame_count = 24
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    arc_angles = np.arange(3, 13)
    terminal_offsets = (1, 2, 3, 5)
    for frame_index in range(frame_count):
        radius_index = 7
        if frame_index >= frame_count - len(terminal_offsets):
            radius_index += terminal_offsets[
                frame_index - (frame_count - len(terminal_offsets))
            ]
        score[frame_index, arc_angles, radius_index] = 9.0

    front = extract_front(score, grid, _tracking_config())

    assert not front.event_detected
    assert "NO_EVENT" in front.quality_flags


def test_slow_quantized_coherent_front_remains_detectable() -> None:
    grid = _polar_grid()
    frame_count = 80
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    arc_angles = np.arange(14, 22)
    for frame_index in range(frame_count):
        radius_index = 9 + frame_index // 4
        score[frame_index, arc_angles, radius_index] = 6.0

    front = extract_front(score, grid, _tracking_config())

    assert front.event_detected
    observed_frames = np.flatnonzero(
        np.any(front.observed_mask[:, arc_angles], axis=1)
    )
    assert observed_frames.size == frame_count
    median_radius = np.median(
        front.radius_px[observed_frames][:, arc_angles],
        axis=1,
    )
    assert median_radius[-1] - median_radius[0] == 19.0


def test_full_circle_search_recovers_a_coherent_outward_arc() -> None:
    grid = _polar_grid()
    score, arc_angles = _coherent_arc_score(center_angle_bin=14)

    front = extract_front(score, grid, _tracking_config())

    assert front.event_detected
    assert "POSITION_ANGLE_WINDOW_APPLIED" not in front.quality_flags
    event_angles = np.flatnonzero(np.any(front.observed_mask, axis=0))
    assert np.intersect1d(event_angles, arc_angles).size >= 6
    assert np.setdiff1d(event_angles, arc_angles).size <= 1

    center_angle = 14
    measured = front.radius_px[:, center_angle]
    observed = np.isfinite(measured)
    assert np.count_nonzero(observed) >= 14
    assert measured[observed][-1] - measured[observed][0] >= 12.0


def test_propagating_front_wins_over_brighter_persistent_inner_arc() -> None:
    """Persistence alone must not make a slowly drifting structure the CME."""

    grid = _polar_grid()
    frame_count = 24
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    inner_angles = np.arange(3, 13)
    cme_angles = np.arange(20, 28)

    # This high-evidence inner arc lasts for the complete interval. A short
    # terminal excursion makes its single-endpoint displacement exceed the
    # generic floor, reproducing the real particle-snow case's tempting but
    # effectively stationary non-CME component.
    terminal_offsets = (1, 2, 3, 5)
    for frame_index in range(frame_count):
        inner_radius_index = 7
        if frame_index >= frame_count - len(terminal_offsets):
            inner_radius_index += terminal_offsets[
                frame_index - (frame_count - len(terminal_offsets))
            ]
        score[frame_index, inner_angles, inner_radius_index] = 9.0

    # The actual front is somewhat fainter and shorter, but it is a broad,
    # temporally overlapping ridge that propagates decisively outward.
    for frame_index in range(4, 18):
        cme_radius_index = 19 + 2 * (frame_index - 4)
        score[frame_index, cme_angles, cme_radius_index] = 5.5

    front = extract_front(score, grid, _tracking_config())

    assert front.event_detected
    event_angles = np.flatnonzero(np.any(front.observed_mask, axis=0))
    np.testing.assert_array_equal(event_angles, cme_angles)
    observed = np.any(front.observed_mask[:, cme_angles], axis=1)
    median_radius = np.median(front.radius_px[observed][:, cme_angles], axis=1)
    assert np.count_nonzero(observed) == 14
    assert median_radius[-1] - median_radius[0] == 26.0


def test_coherent_discovery_can_refine_an_automatic_sector() -> None:
    grid = _polar_grid()
    frame_count = 24
    score = np.zeros(
        (frame_count, _ANGLE_COUNT, _RADIUS_COUNT),
        dtype=np.float64,
    )
    inner_angles = np.arange(3, 13)
    cme_angles = np.arange(20, 28)
    terminal_offsets = (1, 2, 3, 5)
    for frame_index in range(frame_count):
        inner_radius_index = 7
        if frame_index >= frame_count - len(terminal_offsets):
            inner_radius_index += terminal_offsets[
                frame_index - (frame_count - len(terminal_offsets))
            ]
        score[frame_index, inner_angles, inner_radius_index] = 9.0
    for frame_index in range(4, 18):
        cme_radius_index = 19 + 2 * (frame_index - 4)
        score[frame_index, cme_angles, cme_radius_index] = 5.5

    config = replace(
        _tracking_config(),
        association_method="coherent_sector_refined_paths",
        automatic_sector_padding_deg=10.0,
    )
    front = extract_front(score, grid, config)

    assert front.event_detected
    assert "AUTOMATIC_POSITION_ANGLE_SECTOR" in front.quality_flags
    event_angles = np.flatnonzero(np.any(front.observed_mask, axis=0))
    np.testing.assert_array_equal(event_angles, cme_angles)
    assert np.intersect1d(event_angles, inner_angles).size == 0


def test_full_circle_detection_is_rotation_equivariant_through_pa_zero() -> None:
    grid = _polar_grid()
    score, arc_angles = _coherent_arc_score(
        center_angle_bin=0,
        include_isolated_noise=False,
    )
    assert 0 in arc_angles and _ANGLE_COUNT - 1 in arc_angles

    rotation_bins = 7
    baseline = extract_front(score, grid, _tracking_config())
    rotated = extract_front(
        np.roll(score, rotation_bins, axis=1),
        grid,
        _tracking_config(),
    )

    assert baseline.event_detected
    assert rotated.event_detected
    np.testing.assert_array_equal(
        rotated.observed_mask,
        np.roll(baseline.observed_mask, rotation_bins, axis=1),
    )
    np.testing.assert_allclose(
        rotated.radius_px,
        np.roll(baseline.radius_px, rotation_bins, axis=1),
        equal_nan=True,
    )


def test_one_missing_front_frame_remains_a_gap_and_reconnects() -> None:
    grid = _polar_grid()
    missing_frame = 7
    score, arc_angles = _coherent_arc_score(
        center_angle_bin=14,
        missing_frames=(missing_frame,),
        include_isolated_noise=False,
    )

    front = extract_front(score, grid, _tracking_config())

    assert front.event_detected
    assert not np.any(front.observed_mask[missing_frame, arc_angles])
    assert np.count_nonzero(front.observed_mask[missing_frame - 1, arc_angles]) >= 6
    assert np.count_nonzero(front.observed_mask[missing_frame + 1, arc_angles]) >= 6


def test_isolated_boundary_hit_does_not_start_fov_latch() -> None:
    grid = _polar_grid()
    front = _front_with_single_boundary_impulse(grid)

    limited = _field_of_view_limited_frames(
        front,
        grid,
        margin_px=2.0,
        top_fraction=0.2,
        contact_fraction=0.25,
    )

    assert not np.any(limited)


def test_sustained_coherent_boundary_contact_still_starts_fov_latch() -> None:
    grid = _polar_grid()
    front = _front_with_single_boundary_impulse(grid)
    boundary_angles = np.arange(13, 17)
    for frame_index in (8, 9):
        front.radius_px[frame_index, boundary_angles] = grid.radius_px[-1]
        front.radius_rsun[frame_index, boundary_angles] = (
            grid.radius_px[-1] / 20.0
        )

    limited = _field_of_view_limited_frames(
        front,
        grid,
        margin_px=2.0,
        top_fraction=0.5,
        contact_fraction=0.5,
    )

    assert not np.any(limited[:8])
    assert np.all(limited[8:])


def test_two_simultaneous_boundary_samples_can_start_fov_latch() -> None:
    grid = _polar_grid()
    front = _front_with_single_boundary_impulse(grid)
    boundary_angles = np.arange(14, 16)
    front.radius_px[8, boundary_angles] = grid.radius_px[-1]
    front.radius_rsun[8, boundary_angles] = grid.radius_px[-1] / 20.0

    limited = _field_of_view_limited_frames(
        front,
        grid,
        margin_px=2.0,
        top_fraction=0.5,
        contact_fraction=0.5,
        minimum_contact_angles=2,
        minimum_consecutive_frames=2,
    )

    assert not np.any(limited[:8])
    assert np.all(limited[8:])
