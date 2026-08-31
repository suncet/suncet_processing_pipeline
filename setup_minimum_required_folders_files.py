"""Initialize the portable ``suncet_data`` tree and validate its metadata.

This script is intentionally host-independent: the absolute data location comes
only from the required ``suncet_data`` environment variable. It may therefore
be run unchanged on a developer laptop or the SOC Jetson. Reviewed, versioned
metadata CSVs are distributed separately through Dropbox/rclone; setup never
downloads the mutable live Google Sheet or writes metadata CSV content.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from suncet_processing_pipeline.data_paths import get_data_root
from suncet_processing_pipeline.metadata_snapshots import (
    FITS_SOURCE_FILENAME,
    METADATA_VERSION,
    NETCDF_ZARR_SOURCE_FILENAME,
    VERSION_FILENAME,
)


FITS_METADATA_FILENAME = FITS_SOURCE_FILENAME
NETCDF_ZARR_METADATA_FILENAME = NETCDF_ZARR_SOURCE_FILENAME
METADATA_VERSION_FILENAME = VERSION_FILENAME

DATA_DIRECTORIES = (
    "calibration",
    "level0_5",
    "level1",
    "level2",
    "level3",
    "level4",
    "metadata",
    "processing_runs",
    "synthetic/level1",
    "synthetic/level2",
    "telemetry/incoming/uhf",
    "telemetry/incoming/xband",
    "test_data",
    "trends",
    "transfer_logs/lasp_publication",
    "transfer_staging/lasp_publication",
)

FITS_REQUIRED_VARIABLES = {
    "exposure_time",
    "integration_time_inner",
    "integration_time_outer",
    "number_stacked_integrations_inner",
    "number_stacked_integrations_outer",
    "stack_normalization_factor_inner",
    "stack_normalization_factor_outer",
    "dark_file",
    "flat_file",
    "vignette_file",
}

NETCDF_ZARR_REQUIRED_VARIABLES = {
    "dark_file",
    "flat_file",
    "vignette_file",
    "psf_file",
    "bad_pixel_file",
    "stray_light_file",
}


def initialize_data_tree() -> Path:
    """Create the minimum directory structure below ``suncet_data``."""
    data_root = get_data_root(must_exist=False)
    data_root.mkdir(parents=True, exist_ok=True)
    for relative_directory in DATA_DIRECTORIES:
        (data_root / relative_directory).mkdir(parents=True, exist_ok=True)
    return data_root


def _validate_variable_column(
    metadata_path: Path,
    *,
    variable_column: int,
    required_variables: set[str],
) -> None:
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    variables = {
        row[variable_column].strip()
        for row in rows
        if len(row) > variable_column and row[variable_column].strip()
    }
    missing = sorted(required_variables - variables)
    if missing:
        raise ValueError(
            f"Metadata export {metadata_path} is missing required variables: "
            + ", ".join(missing)
        )
    if "vignet_file" in variables:
        raise ValueError(
            f"Metadata export {metadata_path} contains obsolete 'vignet_file'; "
            "use 'vignette_file'."
        )


def require_reviewed_metadata(metadata_directory: Path) -> tuple[Path, Path]:
    """Validate the code-pinned, reviewed metadata CSVs already on this host.

    The live Google Sheet is the development and review surface, not a runtime
    distribution endpoint. A reviewer exports each approved change under a new
    versioned filename and Dropbox/rclone delivers that immutable CSV here.
    """
    fits_path = metadata_directory / FITS_METADATA_FILENAME
    netcdf_zarr_path = metadata_directory / NETCDF_ZARR_METADATA_FILENAME
    missing = [path for path in (fits_path, netcdf_zarr_path) if not path.is_file()]
    if missing:
        expected = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Reviewed metadata CSVs are required but missing from "
            f"{metadata_directory}: {expected}. Run the reviewed Dropbox/rclone "
            "pull-metadata task, then rerun setup. The live Google Sheet is not "
            "downloaded by this setup script."
        )

    _validate_variable_column(
        fits_path,
        variable_column=1,
        required_variables=FITS_REQUIRED_VARIABLES,
    )
    _validate_variable_column(
        netcdf_zarr_path,
        variable_column=1,
        required_variables=NETCDF_ZARR_REQUIRED_VARIABLES,
    )

    version_path = metadata_directory / METADATA_VERSION_FILENAME
    temporary_version_path = version_path.with_suffix(version_path.suffix + ".tmp")
    temporary_version_path.write_text(METADATA_VERSION + "\n", encoding="utf-8")
    temporary_version_path.replace(version_path)

    return fits_path, netcdf_zarr_path


def add_metadata_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help=(
            "Initialize directories before reviewed metadata has been delivered. "
            "After the rclone pull, rerun setup without this option."
        ),
    )
    parser.add_argument(
        "--skip-metadata-download",
        dest="allow_missing_metadata",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def report_calibration_status(calibration_directory: Path) -> None:
    """Report whether calibration FITS assets are present without guessing names."""
    calibration_files = sorted(calibration_directory.glob("*.fits*"))
    if calibration_files:
        print(f"Calibration assets found: {len(calibration_files)}")
    else:
        print(
            "WARNING: no calibration FITS assets are present yet in "
            f"{calibration_directory}. The directory is ready, but Level 1/2 "
            "production processing will require mission-approved files."
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize $suncet_data and validate the reviewed metadata CSVs "
            "already delivered to it."
        )
    )
    add_metadata_policy_arguments(parser)
    return parser


def run(argv: list[str] | None = None) -> Path:
    args = get_parser().parse_args(argv)
    data_root = initialize_data_tree()
    print(f"Initialized SunCET data tree: {data_root}")

    if args.allow_missing_metadata:
        print(
            "WARNING: reviewed metadata validation was explicitly deferred. "
            "Run the Dropbox/rclone pull-metadata task, then rerun setup without "
            "--allow-missing-metadata before processing data."
        )
    else:
        fits_path, netcdf_zarr_path = require_reviewed_metadata(
            data_root / "metadata"
        )
        print(f"Validated reviewed FITS metadata: {fits_path}")
        print(f"Validated reviewed NetCDF/Zarr metadata: {netcdf_zarr_path}")

    report_calibration_status(data_root / "calibration")
    return data_root


if __name__ == "__main__":
    run()
