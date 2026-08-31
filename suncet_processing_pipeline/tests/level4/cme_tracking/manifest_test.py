"""Tests for reviewed CME input manifests."""

from pathlib import Path

import pytest

from suncet_processing_pipeline.level4.cme_tracking.manifest import (
    CadenceStatus,
    CorrectionState,
    InputSourceKind,
    ManifestFrame,
    ManifestPathBase,
    ManifestReview,
    ManifestTimeAxis,
    ManifestValidationError,
    SequenceManifest,
    TimeAxisKind,
    UpstreamProcessing,
    manifest_from_paths,
    read_manifest,
    write_manifest,
)


def _review(status: str = "reviewed") -> ManifestReview:
    return ManifestReview(
        status=status,
        reviewed_by="SunCET science team",
        reviewed_at_utc="2026-08-26T12:00:00Z",
    )


def _frame(path: str, number: int, source_index: int) -> ManifestFrame:
    return ManifestFrame(
        path=path,
        frame_number=number,
        source_index=source_index,
        sha256=f"{number + 1:064x}",
    )


def _synthetic_manifest(frames: tuple[ManifestFrame, ...]) -> SequenceManifest:
    return SequenceManifest(
        scenario_id="bright-fast-test",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        time_axis=ManifestTimeAxis(
            kind=TimeAxisKind.FIXED_CADENCE,
            cadence_seconds=60.0,
            cadence_status=CadenceStatus.ASSUMED,
            absolute_time_valid=False,
        ),
        upstream_processing=UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.NOT_APPLIED,
            level3_geometric_correction=CorrectionState.NOT_APPLIED,
        ),
        frames=frames,
    )


def test_manifest_round_trip_retains_explicit_gap_and_review(tmp_path: Path):
    frames = (
        _frame("images/image_000.fits", 0, 0),
        _frame("images/image_003.fits", 1, 3),
        _frame("images/image_009.fits", 3, 9),
    )
    manifest = _synthetic_manifest(frames)
    path = write_manifest(manifest, tmp_path / "scenario.json")

    loaded = read_manifest(path)

    assert [frame.frame_number for frame in loaded.frames] == [0, 1, 3]
    assert [frame.source_index for frame in loaded.frames] == [0, 3, 9]
    assert loaded.review.status == "reviewed"
    assert loaded.cadence_seconds == 60.0
    assert loaded.source_kind == InputSourceKind.SYNTHETIC_BYPASS
    assert loaded.to_dict() == manifest.to_dict()


def test_historical_bright_fast_manifest_has_verified_one_hour_timing():
    repository = Path(__file__).resolve().parents[4]
    manifest = read_manifest(
        repository
        / "benchmarks/cme_tracking/manifests/bright_fast_no_jitter_historical.json"
    )

    assert len(manifest.frames) == 120
    assert manifest.time_axis.cadence_seconds == 30.0
    assert manifest.time_axis.cadence_status is CadenceStatus.VERIFIED
    assert manifest.time_axis.absolute_time_valid is False
    assert [frame.frame_number for frame in manifest.frames] == list(range(120))
    assert [frame.source_index for frame in manifest.frames] == list(
        range(0, 360, 3)
    )

    metadata = manifest.metadata
    assert metadata["source_cadence_seconds"] == 10.0
    assert metadata["source_run_duration_seconds"] == 3600.0
    assert metadata["source_terminal_index"] == 360
    assert metadata["selected_span_seconds"] == 3570.0

    cadence = manifest.time_axis.cadence_seconds
    assert cadence is not None
    assert len(manifest.frames) * cadence == metadata["source_run_duration_seconds"]
    assert (
        manifest.frames[-1].frame_number - manifest.frames[0].frame_number
    ) * cadence == metadata["selected_span_seconds"]


def test_config_default_manifest_uses_trusted_fits_time_and_full_sequence():
    repository = Path(__file__).resolve().parents[4]
    manifest = read_manifest(
        repository
        / "benchmarks/cme_tracking/manifests/"
        "config_default_no_particle_filter_20230114.json"
    )

    assert manifest.source_kind is InputSourceKind.SYNTHETIC_BYPASS
    assert manifest.time_axis.kind is TimeAxisKind.FITS_HEADERS
    assert manifest.time_axis.absolute_time_valid is True
    assert manifest.time_axis.cadence_seconds is None
    assert (
        manifest.upstream_processing.level2_psf_deconvolution
        is CorrectionState.NOT_APPLIED
    )
    assert (
        manifest.upstream_processing.level3_geometric_correction
        is CorrectionState.NOT_APPLIED
    )
    assert len(manifest.frames) == 241
    assert [frame.frame_number for frame in manifest.frames] == list(range(241))
    assert [frame.source_index for frame in manifest.frames] == list(range(241))
    assert manifest.metadata["expected_cadence_seconds"] == 15.0
    assert manifest.metadata["selected_sample_span_seconds"] == 3600.0
    assert manifest.metadata["fits_checksum_review"] == {
        "verified_frame_count": 241,
        "checksum_and_datasum_valid": True,
    }


def test_manifest_from_paths_hashes_files_and_preserves_supplied_frame_numbers(
    tmp_path: Path,
):
    paths = []
    for suffix in (0, 3, 9):
        path = tmp_path / f"image_{suffix:03d}.fits"
        path.write_bytes(f"frame {suffix}".encode())
        paths.append(path)

    manifest = manifest_from_paths(
        paths,
        scenario_id="gap-test",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=15.0,
        frame_numbers=(0, 1, 3),
        source_indices=(0, 3, 9),
    )

    assert [frame.frame_number for frame in manifest.frames] == [0, 1, 3]
    assert all(len(frame.sha256) == 64 for frame in manifest.frames)
    assert manifest.resolve_files(tmp_path) == tuple(paths)


def test_manifest_from_synthetic_paths_without_cadence_uses_fits_time(
    tmp_path: Path,
):
    paths = []
    for suffix in (0, 1):
        path = tmp_path / f"image_{suffix:03d}.fits"
        path.write_bytes(f"frame {suffix}".encode())
        paths.append(path)

    manifest = manifest_from_paths(
        paths,
        scenario_id="header-time-test",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
    )

    assert manifest.time_axis.kind is TimeAxisKind.FITS_HEADERS
    assert manifest.time_axis.absolute_time_valid is True
    assert manifest.time_axis.cadence_seconds is None
    assert (
        manifest.upstream_processing.level2_psf_deconvolution
        is CorrectionState.NOT_APPLIED
    )
    assert (
        manifest.upstream_processing.level3_geometric_correction
        is CorrectionState.NOT_APPLIED
    )


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.fits", "/absolute.fits", r"C:\outside.fits"),
)
def test_manifest_rejects_unsafe_paths(unsafe_path: str):
    with pytest.raises(ManifestValidationError, match="relative|portable|contain"):
        _frame(unsafe_path, 0, 0)


def test_manifest_rejects_unreviewed_status():
    with pytest.raises(ManifestValidationError, match="status='reviewed'"):
        _review("draft")


def test_manifest_rejects_nonincreasing_frame_numbers():
    with pytest.raises(ManifestValidationError, match="strictly increasing"):
        _synthetic_manifest(
            (
                _frame("image_000.fits", 0, 0),
                _frame("image_003.fits", 0, 3),
            )
        )


def test_legacy_files_only_manifest_is_rejected():
    with pytest.raises(ManifestValidationError, match="files-only"):
        SequenceManifest.from_dict(
            {
                "schema_version": 1,
                "scenario_id": "unsafe-old-manifest",
                "input_stage": "synthetic_bypass",
                "files": ["a.fits", "b.fits"],
            }
        )


def test_production_level3_requires_real_timestamps_and_upstream_corrections():
    frames = (
        _frame("image_000.fits", 0, 0),
        _frame("image_001.fits", 1, 1),
    )
    with pytest.raises(ManifestValidationError, match="FITS timestamps"):
        SequenceManifest(
            scenario_id="bad-production",
            source_kind=InputSourceKind.PRODUCTION_LEVEL3,
            review=_review(),
            time_axis=ManifestTimeAxis(
                kind=TimeAxisKind.FIXED_CADENCE,
                cadence_seconds=60,
                cadence_status=CadenceStatus.ASSUMED,
            ),
            upstream_processing=UpstreamProcessing(
                level2_psf_deconvolution=CorrectionState.APPLIED,
                level3_geometric_correction=CorrectionState.APPLIED,
            ),
            frames=frames,
        )


def test_suncet_data_path_base_resolves_under_declared_root(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = _synthetic_manifest(
        (
            _frame("synthetic/a.fits", 0, 0),
            _frame("synthetic/b.fits", 1, 3),
        )
    )
    manifest = SequenceManifest(
        scenario_id=manifest.scenario_id,
        source_kind=manifest.source_kind,
        review=manifest.review,
        time_axis=manifest.time_axis,
        upstream_processing=manifest.upstream_processing,
        frames=manifest.frames,
        path_base=ManifestPathBase.SUNCET_DATA,
    )

    resolved = manifest.resolve_files(tmp_path, data_root=data_root)

    assert resolved == (
        data_root / "synthetic" / "a.fits",
        data_root / "synthetic" / "b.fits",
    )
