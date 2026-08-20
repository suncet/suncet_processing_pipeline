import json

import pytest

from ..metadata_snapshots import (
    FITS_SOURCE_FILENAME,
    FITS_SNAPSHOT_FILENAME,
    NETCDF_ZARR_SOURCE_FILENAME,
    snapshot_metadata_for_run,
    verify_run_metadata_snapshot,
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
