from astropy.io import fits
from astropy.table import Table
import json
import numpy as np
from PIL import Image
import pytest

from suncet_processing_pipeline.level4.cme_tracking.config import (
    CMETrackingConfig,
    KinematicsConfig,
)
from suncet_processing_pipeline.level4.cme_tracking.input import (
    load_sequence_from_manifest,
    load_synthetic_sequence,
)
from suncet_processing_pipeline.level4.cme_tracking.likelihood import EvidenceConfig
from suncet_processing_pipeline.level4.cme_tracking.manifest import (
    InputSourceKind,
    ManifestReview,
    manifest_from_paths,
    write_manifest,
)
from suncet_processing_pipeline.level4.cme_tracking.movie import (
    write_cme_tracking_movie,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (
    build_event_summary,
    build_track_product,
    run_known_window,
    run_known_window_from_images,
    write_known_window_products,
)
from suncet_processing_pipeline.level4.cme_tracking.polar import PolarConfig
from suncet_processing_pipeline.level4.cme_tracking.summary import FrontSummary
from suncet_processing_pipeline.level4.cme_tracking.tracking import (
    TrackingConfig,
    extract_front,
)
from suncet_processing_pipeline.level4.common.quality import QualityFlag


def _write_expanding_fits_sequence(directory, frame_count=12):
    shape = (96, 96)
    center = 47.5
    solar_radius_px = 15.0
    yy, xx = np.indices(shape, dtype=float)
    dx = xx - center
    dy = yy - center
    radius = np.hypot(dx, dy) / solar_radius_px
    # This WCS has north at increasing row and solar east at decreasing column.
    pa = np.rad2deg(np.arctan2(-dx, dy)) % 360.0
    delta = (pa - 270.0 + 180.0) % 360.0 - 180.0
    taper = np.where(
        np.abs(delta) <= 35.0,
        0.5 * (1.0 + np.cos(np.pi * delta / 35.0)),
        0.0,
    )
    background = 4.0 * np.exp(-np.maximum(radius - 0.8, 0) / 0.7)
    paths = []
    for index in range(frame_count):
        front = 1.30 + 0.05 * index - 0.04 * (delta / 35.0) ** 2
        bright = 7.0 * taper * 0.5 * (1.0 - np.tanh((radius - front) / 0.025))
        image = (background + bright).astype(np.float32)
        header = fits.Header(
            {
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CDELT1": 10.0,
                "CDELT2": 10.0,
                "CRPIX1": center + 1.0,
                "CRPIX2": center + 1.0,
                "CRVAL1": 0.0,
                "CRVAL2": 0.0,
                "RSUN": solar_radius_px * 10.0,
                "DATE-OBS": "2026-01-01T00:00:00.000",
                "LEVEL": "0.5",
            }
        )
        path = directory / f"fixture_{index:03d}.fits"
        fits.writeto(path, image, header, overwrite=True)
        paths.append(path)
    return paths


def test_known_window_pipeline_writes_inspectable_products(tmp_path) -> None:
    paths = _write_expanding_fits_sequence(tmp_path)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="analytic-expanding-front",
        assumed_cadence_seconds=30.0,
    )
    config = CMETrackingConfig(
        polar=PolarConfig(
            position_angle_step_deg=4.0,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
        evidence=EvidenceConfig(minimum_leading_edge_z=0.5),
        tracking=TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=4.0,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            minimum_event_frames=6,
        ),
        kinematics=KinematicsConfig(
            endpoint_samples=1,
            uncertainty_samples=8,
            random_seed=4,
        ),
        field_of_view_margin_px=3.0,
    )

    run = run_known_window(sequence, config)

    assert run.front.event_detected
    assert run.kinematics is not None
    finite = run.summary.height_rsun[np.isfinite(run.summary.height_rsun)]
    assert finite[-1] > finite[0]
    directory = write_known_window_products(
        run,
        tmp_path / "products",
        "analytic-event-001",
        repository=tmp_path,
    )
    assert (directory / "front_overlay.png").is_file()
    assert (directory / "COMPLETE.json").is_file()
    table = Table.read(directory / "track.ecsv", format="ascii.ecsv")
    assert len(table) == len(paths)
    assert table.meta["source_kind"] == "synthetic_bypass"
    assert "height_raw_sigma_rsun" in table.colnames
    assert "height_fit_sigma_rsun" in table.colnames
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["input"]["frames"][0]["path"] == paths[0].name
    assert str(tmp_path) not in json.dumps(summary["input"])
    assert summary["input"]["hash_verification_status"] == "not_applicable"
    assert "direct-directory development input" in summary["input"]["path_policy"]
    assert "INPUT_INTEGRITY_UNVERIFIED" in summary["quality_flags"]

    movie_path = write_cme_tracking_movie(
        run,
        build_track_product(run, "analytic-event-001"),
        tmp_path / "all-frame-diagnostic.gif",
        fps=4.0,
        dpi=40,
        writer_name="pillow",
    )
    with Image.open(movie_path) as movie:
        assert movie.n_frames == len(paths)


def test_resident_compute_boundary_matches_fits_backed_pipeline(tmp_path) -> None:
    paths = _write_expanding_fits_sequence(tmp_path, frame_count=10)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="resident-equivalence",
        assumed_cadence_seconds=30.0,
    )
    configuration = CMETrackingConfig(
        polar=PolarConfig(
            position_angle_step_deg=4.0,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
        evidence=EvidenceConfig(minimum_leading_edge_z=0.5),
        tracking=TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=4.0,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            minimum_event_frames=6,
        ),
        kinematics=KinematicsConfig(
            endpoint_samples=1,
            uncertainty_samples=8,
            random_seed=4,
        ),
        field_of_view_margin_px=3.0,
    )

    fits_backed = run_known_window(sequence, configuration)
    images = sequence.materialize(dtype=np.float32)
    images_before = images.copy()
    resident = run_known_window_from_images(sequence, images, configuration)

    np.testing.assert_array_equal(images, images_before)
    np.testing.assert_array_equal(
        resident.likelihood.score,
        fits_backed.likelihood.score,
    )
    np.testing.assert_allclose(
        resident.front.radius_rsun,
        fits_backed.front.radius_rsun,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        resident.summary.height_rsun,
        fits_backed.summary.height_rsun,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        resident.field_of_view_limited_mask,
        fits_backed.field_of_view_limited_mask,
    )
    assert resident.kinematics is not None
    assert fits_backed.kinematics is not None
    np.testing.assert_allclose(
        resident.kinematics.fitted_height,
        fits_backed.kinematics.fitted_height,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        resident.kinematics.speed,
        fits_backed.kinematics.speed,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        resident.kinematics.acceleration,
        fits_backed.kinematics.acceleration,
        equal_nan=True,
    )


def test_temporal_median_endpoints_are_explicitly_flagged(tmp_path) -> None:
    paths = _write_expanding_fits_sequence(tmp_path, frame_count=10)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="temporal-median-endpoints",
        assumed_cadence_seconds=30.0,
    )
    run = run_known_window(
        sequence,
        CMETrackingConfig(
            polar=PolarConfig(
                position_angle_step_deg=4.0,
                minimum_radius_rsun=1.05,
                maximum_radius_rsun=3.0,
            ),
            evidence=EvidenceConfig(
                temporal_median_window_frames=3,
                minimum_leading_edge_z=0.5,
            ),
        ),
    )

    track = build_track_product(run, "temporal-median-endpoints-001")
    unsupported = int(QualityFlag.TEMPORAL_EVIDENCE_UNSUPPORTED)

    assert track.quality_mask[0] & unsupported
    assert track.quality_mask[-1] & unsupported
    assert not np.any(track.quality_mask[1:-1] & unsupported)
    assert build_event_summary(run, "temporal-median-endpoints-001")[
        "temporal_evidence_unsupported_frame_count"
    ] == 2


def test_resident_compute_boundary_rejects_wrong_shape(tmp_path) -> None:
    paths = _write_expanding_fits_sequence(tmp_path, frame_count=4)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="resident-shape-check",
        assumed_cadence_seconds=30.0,
    )

    with pytest.raises(ValueError, match="Resident images have shape"):
        run_known_window_from_images(
            sequence,
            np.zeros((3, *sequence.geometry.image_shape_yx), dtype=np.float32),
        )


def test_known_window_pipeline_uses_reviewed_manifest_frame_gaps(tmp_path) -> None:
    all_paths = _write_expanding_fits_sequence(tmp_path)
    selected_indices = (0, 1, 3, 4, 6, 7, 9, 10)
    selected_paths = tuple(all_paths[index] for index in selected_indices)
    manifest = manifest_from_paths(
        selected_paths,
        scenario_id="analytic-expanding-front-with-gaps",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=ManifestReview(
            status="reviewed",
            reviewed_by="SunCET test suite",
            reviewed_at_utc="2026-08-26T12:00:00Z",
        ),
        relative_to=tmp_path,
        cadence_seconds=30.0,
        frame_numbers=selected_indices,
        source_indices=selected_indices,
    )
    manifest_path = write_manifest(manifest, tmp_path / "gapped-manifest.json")
    sequence, _ = load_sequence_from_manifest(manifest_path)
    config = CMETrackingConfig(
        polar=PolarConfig(
            position_angle_step_deg=4.0,
            radial_step_px=0.5,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
        evidence=EvidenceConfig(minimum_leading_edge_z=0.5),
        tracking=TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=1.5,
            maximum_gap_frames=1,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            minimum_event_frames=6,
        ),
        kinematics=KinematicsConfig(
            endpoint_samples=1,
            uncertainty_samples=8,
            random_seed=4,
        ),
        field_of_view_margin_px=3.0,
    )

    run = run_known_window(sequence, config)

    np.testing.assert_array_equal(
        [frame.frame_number for frame in sequence.frames],
        selected_indices,
    )
    np.testing.assert_array_equal(
        sequence.elapsed_seconds,
        np.asarray(selected_indices) * 30.0,
    )
    assert run.front.event_detected
    assert run.kinematics is not None
    row_only_front = extract_front(
        run.likelihood,
        run.polar_grid,
        config.tracking,
    )
    assert not row_only_front.event_detected


def test_pipeline_flags_filtered_height_without_changing_raw_track(
    tmp_path, monkeypatch
) -> None:
    paths = _write_expanding_fits_sequence(tmp_path)
    sequence = load_synthetic_sequence(
        paths,
        scenario_id="analytic-height-outlier",
        assumed_cadence_seconds=30.0,
    )
    raw_height = 1.3 + 0.03 * np.arange(len(paths), dtype=float)
    raw_height[6] -= 0.8

    def fake_summary(front, config):
        count = raw_height.size
        return FrontSummary(
            height_rsun=raw_height.copy(),
            height_sigma_rsun=np.full(count, 0.01),
            position_angle_deg=np.full(count, 270.0),
            angular_width_deg=np.full(count, 40.0),
            coverage_fraction=np.ones(count),
            confidence=np.ones(count),
            observed_angle_count=np.full(count, 10),
        )

    monkeypatch.setattr(
        "suncet_processing_pipeline.level4.cme_tracking.pipeline.summarize_front",
        fake_summary,
    )
    config = CMETrackingConfig(
        polar=PolarConfig(
            position_angle_step_deg=4.0,
            minimum_radius_rsun=1.05,
            maximum_radius_rsun=3.0,
        ),
        evidence=EvidenceConfig(minimum_leading_edge_z=0.5),
        tracking=TrackingConfig(
            score_threshold=1.5,
            maximum_outward_step_px_per_frame=4.0,
            minimum_track_points=5,
            minimum_outward_displacement_px=3.0,
            minimum_angular_support_deg=16.0,
            minimum_event_frames=6,
        ),
        kinematics=KinematicsConfig(
            endpoint_samples=1,
            uncertainty_samples=8,
            random_seed=4,
            height_outlier_filter_enabled=True,
            height_outlier_filter_window_samples=7,
            height_outlier_filter_absolute_tolerance_rsun=0.2,
            height_outlier_filter_minimum_neighbors=4,
        ),
        field_of_view_margin_px=0.0,
    )

    run = run_known_window(sequence, config)
    track = build_track_product(run, "filtered-event")
    summary = build_event_summary(run, "filtered-event", repository=tmp_path)

    np.testing.assert_array_equal(
        np.flatnonzero(run.headline_height_filter.outlier_mask), [6]
    )
    assert run.summary.height_rsun[6] == raw_height[6]
    assert np.isnan(run.kinematics.raw_height[6])
    assert track.height_raw_rsun[6] == raw_height[6]
    assert track.quality_mask[6] & int(QualityFlag.KINEMATIC_HEIGHT_OUTLIER)
    assert "KINEMATIC_HEIGHT_OUTLIER" in summary["quality_flags"]
    assert summary["headline_height_outlier_filter"]["rejected_frame_numbers"] == [6]
    assert (
        run.kinematics.method_metadata["headline_height_outlier_filter"]
        ["replacement_policy"]
        .startswith("none;")
    )
