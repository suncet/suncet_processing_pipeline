"""Tests for the lazy Level 4 FITS-sequence input boundary."""

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from suncet_processing_pipeline.level4.cme_tracking.input import (
    SequenceInputError,
    load_level3_sequence,
    load_sequence_from_manifest,
    load_synthetic_sequence,
)
from suncet_processing_pipeline.level4.cme_tracking.manifest import (
    CorrectionState,
    InputSourceKind,
    ManifestReview,
    TimeAxisKind,
    manifest_from_paths,
    write_manifest,
)


def _write_fits(
    path: Path,
    value: float,
    *,
    date_obs: str = "2023-02-14T17:00:00.000",
    level: str = "0.5",
    shape: tuple[int, int] = (12, 16),
    invalid_datasum: bool = False,
) -> np.ndarray:
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + value
    header = fits.Header()
    header["DATE-OBS"] = date_obs
    header["LEVEL"] = level
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = shape[1] / 2 + 0.5
    header["CRPIX2"] = shape[0] / 2 + 0.5
    header["CDELT1"] = 9.6 / 3600.0
    header["CDELT2"] = 9.6 / 3600.0
    header["PC1_1"] = 0.916
    header["PC1_2"] = 0.0
    header["PC2_1"] = 0.0
    header["PC2_2"] = 0.9146666666666666
    header["RSUN"] = 43.9
    if invalid_datasum:
        header["DATASUM"] = "not-an-integer"
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return data


def _make_sequence(
    tmp_path: Path,
    *,
    dates: tuple[str, ...] | None = None,
    level: str = "0.5",
    suffixes: tuple[int, ...] = (0, 3, 6),
) -> tuple[list[Path], list[np.ndarray]]:
    paths: list[Path] = []
    arrays: list[np.ndarray] = []
    for index, suffix in enumerate(suffixes):
        path = tmp_path / f"image_{suffix:03d}.fits"
        paths.append(path)
        arrays.append(
            _write_fits(
                path,
                float(index * 1000),
                date_obs=(
                    dates[index]
                    if dates is not None
                    else "2023-02-14T17:00:00.000"
                ),
                level=level,
                invalid_datasum=index == 0,
            )
        )
    return paths, arrays


def _review() -> ManifestReview:
    return ManifestReview(
        status="reviewed",
        reviewed_by="SunCET science team",
        reviewed_at_utc="2026-08-26T12:00:00Z",
    )


def test_direct_synthetic_loader_preserves_gap_raw_inputs_and_wcs(tmp_path: Path):
    paths, arrays = _make_sequence(tmp_path, suffixes=(0, 3, 9))

    sequence = load_synthetic_sequence(
        paths,
        scenario_id="bright-fast-test",
        assumed_cadence_seconds=60.0,
        frame_numbers=(0, 1, 3),
    )

    np.testing.assert_array_equal(sequence.elapsed_seconds, [0.0, 60.0, 180.0])
    assert sequence.source_kind == InputSourceKind.SYNTHETIC_BYPASS
    assert sequence.observation_times_utc == (None, None, None)
    assert sequence.header_date_obs == (
        "2023-02-14T17:00:00.000",
        "2023-02-14T17:00:00.000",
        "2023-02-14T17:00:00.000",
    )
    assert not hasattr(sequence, "data")
    assert sequence.paths == tuple(paths)
    assert sequence.frames[-1].frame_number == 3
    assert sequence.frames[-1].source_index == 9
    assert sequence.hash_verification_status == "not_applicable"

    loaded = sequence.read_frame(1)
    np.testing.assert_array_equal(loaded.data, arrays[1])
    assert loaded.data.dtype.str == sequence.frames[1].dtype == ">f8"
    assert loaded.data.dtype.kind == arrays[1].dtype.kind
    assert loaded.header["DATE-OBS"] == "2023-02-14T17:00:00.000"
    assert loaded.wcs.has_celestial

    materialized = sequence.materialize(dtype=np.float32)
    assert materialized.shape == (3, 12, 16)
    assert materialized.dtype == np.float32
    np.testing.assert_allclose(materialized[2], arrays[2])


def test_geometry_uses_full_wcs_matrix_and_cardinal_vectors(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)

    sequence = load_synthetic_sequence(
        paths,
        scenario_id="wcs-test",
        assumed_cadence_seconds=15.0,
    )
    geometry = sequence.geometry

    np.testing.assert_allclose(
        geometry.pixel_scales_arcsec_xy,
        (9.6 * 0.916, 9.6 * 0.9146666666666666),
        rtol=1e-10,
    )
    assert geometry.pixel_scale_arcsec == pytest.approx(
        np.mean(geometry.pixel_scales_arcsec_xy)
    )
    assert geometry.solar_radius_px == pytest.approx(
        43.9 / geometry.pixel_scale_arcsec
    )
    np.testing.assert_allclose(geometry.north_vector_yx, (1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(geometry.east_vector_yx, (0.0, -1.0), atol=1e-12)
    assert geometry.orientation_source == "fits_wcs"


def test_direct_synthetic_loader_uses_strict_header_time_without_cadence(
    tmp_path: Path,
):
    dates = (
        "2027-02-15T00:00:00.000",
        "2027-02-15T00:00:15.000",
        "2027-02-15T00:00:30.000",
    )
    paths, _arrays = _make_sequence(tmp_path, dates=dates)

    sequence = load_synthetic_sequence(paths, scenario_id="header-timed-synthetic")

    np.testing.assert_allclose(sequence.elapsed_seconds, [0.0, 15.0, 30.0])
    assert sequence.time_axis.kind is TimeAxisKind.FITS_HEADERS
    assert sequence.time_axis.absolute_time_valid is True
    assert sequence.time_source == "fits_headers"
    assert sequence.observation_times_utc == dates
    assert (
        sequence.upstream_processing.level2_psf_deconvolution
        is CorrectionState.NOT_APPLIED
    )
    assert (
        sequence.upstream_processing.level3_geometric_correction
        is CorrectionState.NOT_APPLIED
    )


def test_direct_synthetic_header_time_must_be_strictly_increasing(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)

    with pytest.raises(SequenceInputError, match="strictly increasing"):
        load_synthetic_sequence(paths, scenario_id="missing-cadence")


def test_unsigned_scaled_fits_preserves_physical_uint16_values(tmp_path: Path):
    physical_arrays = (
        np.array([[0, 1, 32767, 32768], [40000, 50000, 65534, 65535]], dtype=np.uint16),
        np.array([[2, 3, 32000, 33000], [41000, 51000, 65000, 65533]], dtype=np.uint16),
    )
    dates = (
        "2027-02-15T00:00:00.000",
        "2027-02-15T00:00:15.000",
    )
    paths: list[Path] = []
    for index, (array, date_obs) in enumerate(
        zip(physical_arrays, dates, strict=True)
    ):
        path = tmp_path / f"unsigned_{index:03d}.fits"
        _write_fits(path, 0, date_obs=date_obs, shape=array.shape)
        header = fits.getheader(path)
        fits.PrimaryHDU(data=array, header=header).writeto(path, overwrite=True)
        paths.append(path)

    header = fits.getheader(paths[0])
    assert header["BSCALE"] == 1
    assert header["BZERO"] == 32768

    sequence = load_synthetic_sequence(paths, scenario_id="unsigned-scaled")
    loaded = sequence.read_frame(0)
    materialized = sequence.materialize(dtype=None)

    assert np.dtype(sequence.frames[0].dtype) == np.dtype(np.uint16)
    assert loaded.data.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(loaded.data, physical_arrays[0])
    assert materialized.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(materialized, np.stack(physical_arrays))


def test_manifest_mode_preserves_missing_frame_interval_and_verifies_hashes(
    tmp_path: Path,
):
    paths, _arrays = _make_sequence(tmp_path, suffixes=(0, 3, 9))
    manifest = manifest_from_paths(
        paths,
        scenario_id="reviewed-gap",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=15.0,
        frame_numbers=(0, 1, 3),
        source_indices=(0, 3, 9),
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")

    sequence, loaded_manifest = load_sequence_from_manifest(manifest_path)

    np.testing.assert_array_equal(sequence.elapsed_seconds, [0.0, 15.0, 45.0])
    assert sequence.time_source == "reviewed_manifest_fixed_cadence"
    assert sequence.manifest_path == manifest_path
    assert loaded_manifest.review.status == "reviewed"
    assert all(frame.expected_sha256 for frame in sequence.frames)
    assert sequence.hash_verification_status == "verified_at_load_and_read"
    assert len(sequence.manifest_sha256 or "") == 64


def test_synthetic_manifest_can_use_strict_fits_header_time(tmp_path: Path):
    dates = (
        "2027-02-15T00:00:00.000",
        "2027-02-15T00:00:15.000",
        "2027-02-15T00:00:30.000",
    )
    paths, _arrays = _make_sequence(tmp_path, dates=dates)
    manifest = manifest_from_paths(
        paths,
        scenario_id="reviewed-header-time",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")

    sequence, loaded_manifest = load_sequence_from_manifest(manifest_path)

    np.testing.assert_allclose(sequence.elapsed_seconds, [0.0, 15.0, 30.0])
    assert loaded_manifest.time_axis.kind is TimeAxisKind.FITS_HEADERS
    assert loaded_manifest.time_axis.absolute_time_valid is True
    assert sequence.time_source == "reviewed_manifest_fits_headers"
    assert sequence.observation_times_utc == dates
    assert (
        sequence.upstream_processing.level2_psf_deconvolution
        is CorrectionState.NOT_APPLIED
    )
    assert (
        sequence.upstream_processing.level3_geometric_correction
        is CorrectionState.NOT_APPLIED
    )


def test_manifest_hash_mismatch_fails_before_fits_loading(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)
    manifest = manifest_from_paths(
        paths,
        scenario_id="hash-test",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=60.0,
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")
    paths[1].write_bytes(paths[1].read_bytes() + b"changed")

    with pytest.raises(SequenceInputError, match="SHA-256 mismatch"):
        load_sequence_from_manifest(manifest_path)


def test_manifest_hash_is_rechecked_when_a_frame_is_read(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)
    manifest = manifest_from_paths(
        paths,
        scenario_id="hash-recheck",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=60.0,
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")
    sequence, _loaded_manifest = load_sequence_from_manifest(manifest_path)
    paths[1].write_bytes(paths[1].read_bytes() + b"changed after validation")

    with pytest.raises(SequenceInputError, match="while reading manifest input"):
        sequence.read_frame(1)


def test_skipped_manifest_hash_verification_is_explicit(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)
    manifest = manifest_from_paths(
        paths,
        scenario_id="hash-skipped",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=60.0,
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")

    sequence, _loaded_manifest = load_sequence_from_manifest(
        manifest_path,
        verify_hashes=False,
    )

    assert sequence.hash_verification_status == "skipped_by_request"
    assert not any(frame.verify_sha256_on_read for frame in sequence.frames)
    assert sequence.provenance_dict()["hash_verification_status"] == (
        "skipped_by_request"
    )


def test_direct_suffix_gap_is_warning_not_an_implicit_clock(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path, suffixes=(0, 3, 9))

    sequence = load_synthetic_sequence(
        paths,
        scenario_id="direct-gap-warning",
        assumed_cadence_seconds=10.0,
    )

    # Direct mode uses the ordered-list ordinal unless explicit frame numbers
    # are supplied; it never turns the suffix difference into elapsed time.
    np.testing.assert_array_equal(sequence.elapsed_seconds, [0.0, 10.0, 20.0])
    assert "NONUNIFORM_SOURCE_INDEX_STEP" in {
        issue.code for issue in sequence.issues
    }


def test_invalid_historical_datasum_is_recorded_but_does_not_block(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)

    sequence = load_synthetic_sequence(
        paths,
        scenario_id="bad-datasum",
        assumed_cadence_seconds=10.0,
    )

    assert "INVALID_FITS_DATASUM" in {issue.code for issue in sequence.issues}


def test_production_level3_rejects_synthetic_level_and_frozen_time(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)

    with pytest.raises(SequenceInputError, match="LEVEL=3"):
        load_level3_sequence(paths, scenario_id="not-level3")


def test_production_level3_uses_strictly_increasing_header_times(tmp_path: Path):
    dates = (
        "2027-02-15T00:00:00.000",
        "2027-02-15T00:00:10.000",
        "2027-02-15T00:00:20.000",
    )
    paths, _arrays = _make_sequence(tmp_path, dates=dates, level="3")

    sequence = load_level3_sequence(paths, scenario_id="production-test")

    np.testing.assert_allclose(sequence.elapsed_seconds, [0.0, 10.0, 20.0])
    assert sequence.source_kind == InputSourceKind.PRODUCTION_LEVEL3
    assert sequence.observation_times_utc == dates


def test_production_level3_never_allows_inconsistent_geometry(tmp_path: Path):
    dates = (
        "2027-02-15T00:00:00.000",
        "2027-02-15T00:00:10.000",
        "2027-02-15T00:00:20.000",
    )
    paths, _arrays = _make_sequence(tmp_path, dates=dates, level="3")

    with pytest.raises(SequenceInputError, match="may not allow inconsistent"):
        load_level3_sequence(
            paths,
            scenario_id="unsafe-production-override",
            allow_inconsistent_geometry=True,
        )


def test_sequence_rejects_mirrored_east_west_wcs(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)
    with fits.open(paths[1], mode="update") as hdul:
        hdul[0].header["CDELT1"] *= -1.0

    with pytest.raises(SequenceInputError, match="east by 180"):
        load_synthetic_sequence(
            paths,
            scenario_id="mirrored-frame",
            assumed_cadence_seconds=10.0,
        )


def test_explicit_cardinal_vectors_must_be_orthogonal(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)

    with pytest.raises(SequenceInputError, match="must be orthogonal"):
        load_synthetic_sequence(
            paths,
            scenario_id="skewed-cardinals",
            assumed_cadence_seconds=10.0,
            north_vector_yx=(1.0, 0.0),
            east_vector_yx=(1.0, 1.0),
        )


def test_manifest_cannot_be_mixed_with_direct_cadence_arguments(tmp_path: Path):
    paths, _arrays = _make_sequence(tmp_path)
    manifest = manifest_from_paths(
        paths,
        scenario_id="mode-test",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=_review(),
        relative_to=tmp_path,
        cadence_seconds=60.0,
    )
    manifest_path = write_manifest(manifest, tmp_path / "scenario.json")

    with pytest.raises(SequenceInputError, match="may not be combined"):
        load_synthetic_sequence(
            paths,
            scenario_id="bad-mixed-mode",
            assumed_cadence_seconds=30.0,
            manifest_path=manifest_path,
        )


def test_mismatched_shapes_are_rejected(tmp_path: Path):
    first = tmp_path / "image_000.fits"
    second = tmp_path / "image_003.fits"
    _write_fits(first, 0, shape=(12, 16))
    _write_fits(second, 1, shape=(13, 16))

    with pytest.raises(SequenceInputError, match="differs"):
        load_synthetic_sequence(
            (first, second),
            scenario_id="shape-test",
            assumed_cadence_seconds=10.0,
        )
