from configparser import ConfigParser
import json
from pathlib import Path

import pytest

from ..metadata_snapshots import (
    FITS_SOURCE_FILENAME,
    FITS_SNAPSHOT_FILENAME,
    METADATA_VERSION,
    NETCDF_ZARR_SOURCE_FILENAME,
    snapshot_metadata_for_run,
    verify_run_metadata_snapshot,
)


def test_active_metadata_filenames_share_the_code_pinned_version():
    config = ConfigParser()
    config.read(
        Path(__file__).parents[1] / "config_files" / "config_default.ini",
        encoding="utf-8",
    )

    assert config["structure"]["base_metadata_filename"] == FITS_SOURCE_FILENAME
    assert FITS_SOURCE_FILENAME == (
        f"suncet_metadata_definition_v{METADATA_VERSION}-FITS.csv"
    )
    assert NETCDF_ZARR_SOURCE_FILENAME == (
        f"suncet_metadata_definition_v{METADATA_VERSION}-NetCDF-Zarr.csv"
    )


def test_run_metadata_snapshot_is_versioned_and_checksum_guarded(tmp_path):
    data_root = tmp_path / "data"
    metadata = data_root / "metadata"
    run = data_root / "processing_runs" / "test"
    metadata.mkdir(parents=True)
    run.mkdir(parents=True)
    (metadata / FITS_SOURCE_FILENAME).write_text("fits definition\n")
    (metadata / NETCDF_ZARR_SOURCE_FILENAME).write_text("array definition\n")

    manifest = snapshot_metadata_for_run(run, data_root=data_root)

    assert manifest["metadata_version"] == "1.0.2dev"
    assert verify_run_metadata_snapshot(run) == json.loads(
        (run / "metadata_snapshot.json").read_text()
    )
    (run / FITS_SNAPSHOT_FILENAME).write_text("changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_run_metadata_snapshot(run)
