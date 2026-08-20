import csv

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


def test_download_live_metadata_validates_before_replacing(monkeypatch, tmp_path):
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()

    def fake_download(url, out_path, **_kwargs):
        if url == setup_data.SUNCET_METADATA_FITS_LIVE_URL:
            _write_metadata(
                out_path,
                setup_data.FITS_REQUIRED_VARIABLES,
                columns=10,
            )
        else:
            _write_metadata(
                out_path,
                setup_data.NETCDF_ZARR_REQUIRED_VARIABLES,
                columns=13,
            )

    monkeypatch.setattr(setup_data, "download_file", fake_download)
    fits_path, netcdf_zarr_path = setup_data.download_live_metadata(
        metadata_directory
    )

    assert fits_path.name == setup_data.FITS_METADATA_FILENAME
    assert netcdf_zarr_path.name == setup_data.NETCDF_ZARR_METADATA_FILENAME
    assert fits_path.is_file()
    assert netcdf_zarr_path.is_file()
    assert (
        metadata_directory / setup_data.METADATA_VERSION_FILENAME
    ).read_text() == setup_data.METADATA_VERSION + "\n"
    assert not list(metadata_directory.glob("*.download"))
