import csv

import pytest

import setup_minimum_required_folders_files as setup_data


def _write_metadata(path, variables, *, columns):
    rows = [[""] * columns for _ in range(2 + len(variables))]
    for row, variable in zip(rows[2:], sorted(variables)):
        row[1] = variable
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_initialize_data_tree_is_idempotent(monkeypatch, tmp_path):
    data_root = tmp_path / "suncet-data"
    monkeypatch.setenv("suncet_data", str(data_root))

    assert setup_data.initialize_data_tree() == data_root
    assert setup_data.initialize_data_tree() == data_root
    for relative_directory in setup_data.DATA_DIRECTORIES:
        assert (data_root / relative_directory).is_dir()
    assert "transfer_logs/aws_ingest" not in setup_data.DATA_DIRECTORIES
    assert not (data_root / "transfer_logs/aws_ingest").exists()


def test_require_reviewed_metadata_validates_without_replacing_csvs(tmp_path):
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()

    fits_path = metadata_directory / setup_data.FITS_METADATA_FILENAME
    netcdf_zarr_path = metadata_directory / setup_data.NETCDF_ZARR_METADATA_FILENAME
    _write_metadata(fits_path, setup_data.FITS_REQUIRED_VARIABLES, columns=10)
    _write_metadata(
        netcdf_zarr_path,
        setup_data.NETCDF_ZARR_REQUIRED_VARIABLES,
        columns=13,
    )
    original_fits = fits_path.read_bytes()
    original_netcdf_zarr = netcdf_zarr_path.read_bytes()

    actual_fits, actual_netcdf_zarr = setup_data.require_reviewed_metadata(
        metadata_directory
    )

    assert actual_fits == fits_path
    assert actual_netcdf_zarr == netcdf_zarr_path
    assert fits_path.read_bytes() == original_fits
    assert netcdf_zarr_path.read_bytes() == original_netcdf_zarr
    assert (
        metadata_directory / setup_data.METADATA_VERSION_FILENAME
    ).read_text() == setup_data.METADATA_VERSION + "\n"
    assert not list(metadata_directory.glob("*.tmp"))


def test_require_reviewed_metadata_fails_with_rclone_guidance(tmp_path):
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()

    with pytest.raises(FileNotFoundError, match="pull-metadata"):
        setup_data.require_reviewed_metadata(metadata_directory)

    assert not (metadata_directory / setup_data.METADATA_VERSION_FILENAME).exists()


def test_run_requires_reviewed_metadata_by_default(monkeypatch, tmp_path):
    data_root = tmp_path / "suncet-data"
    monkeypatch.setenv("suncet_data", str(data_root))

    with pytest.raises(FileNotFoundError, match="live Google Sheet is not downloaded"):
        setup_data.run([])

    assert (data_root / "metadata").is_dir()


def test_run_can_explicitly_defer_metadata_validation(monkeypatch, tmp_path):
    data_root = tmp_path / "suncet-data"
    monkeypatch.setenv("suncet_data", str(data_root))

    assert setup_data.run(["--allow-missing-metadata"]) == data_root
    assert (data_root / "metadata").is_dir()
