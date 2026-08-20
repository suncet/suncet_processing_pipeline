"""Initialize the portable ``suncet_data`` tree and its metadata definitions.

This script is intentionally host-independent: the absolute data location comes
only from the required ``suncet_data`` environment variable. It may therefore
be run unchanged on a developer laptop or the SOC Jetson.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import urllib.request

from suncet_processing_pipeline.data_paths import get_data_root
from suncet_processing_pipeline.metadata_snapshots import (
    FITS_SOURCE_FILENAME,
    METADATA_VERSION,
    NETCDF_ZARR_SOURCE_FILENAME,
    VERSION_FILENAME,
)


GOOGLE_SHEET_ID = "18W6bGEchqG0jehqo0b5oj78b8fG_wEmbKwkjHDh5a6w"
GOOGLE_SHEET_FITS_GID = "0"
GOOGLE_SHEET_NETCDF_ZARR_GID = "2088748176"

SUNCET_METADATA_FITS_LIVE_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export"
    f"?format=csv&gid={GOOGLE_SHEET_FITS_GID}"
)
SUNCET_METADATA_NETCDF_ZARR_LIVE_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export"
    f"?format=csv&gid={GOOGLE_SHEET_NETCDF_ZARR_GID}"
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
    "telemetry",
    "test_data",
    "trends",
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


def download_file(url: str, out_path: Path, *, timeout: float = 60.0) -> None:
    """Download one file with normal TLS verification and atomic replacement."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            temporary_path.write_bytes(response.read())
        temporary_path.replace(out_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_variable_column(
    metadata_path: Path,
    *,
    variable_column: int,
    required_variables: set[str],
) -> None:
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"Downloaded metadata file is empty: {metadata_path}")

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


def download_live_metadata(metadata_directory: Path) -> tuple[Path, Path]:
    """Download and validate both authoritative development metadata tabs."""
    fits_path = metadata_directory / FITS_METADATA_FILENAME
    netcdf_zarr_path = metadata_directory / NETCDF_ZARR_METADATA_FILENAME
    fits_staging_path = fits_path.with_suffix(fits_path.suffix + ".download")
    netcdf_zarr_staging_path = netcdf_zarr_path.with_suffix(
        netcdf_zarr_path.suffix + ".download"
    )

    try:
        download_file(SUNCET_METADATA_FITS_LIVE_URL, fits_staging_path)
        download_file(
            SUNCET_METADATA_NETCDF_ZARR_LIVE_URL, netcdf_zarr_staging_path
        )

        _validate_variable_column(
            fits_staging_path,
            variable_column=1,
            required_variables=FITS_REQUIRED_VARIABLES,
        )
        _validate_variable_column(
            netcdf_zarr_staging_path,
            variable_column=1,
            required_variables=NETCDF_ZARR_REQUIRED_VARIABLES,
        )
        fits_staging_path.replace(fits_path)
        netcdf_zarr_staging_path.replace(netcdf_zarr_path)
    finally:
        fits_staging_path.unlink(missing_ok=True)
        netcdf_zarr_staging_path.unlink(missing_ok=True)

    (metadata_directory / METADATA_VERSION_FILENAME).write_text(
        METADATA_VERSION + "\n", encoding="utf-8"
    )
    return fits_path, netcdf_zarr_path


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
        description="Initialize the directory structure below $suncet_data."
    )
    parser.add_argument(
        "--skip-metadata-download",
        action="store_true",
        help="Create directories without refreshing metadata from the live Sheet.",
    )
    return parser


def run(argv: list[str] | None = None) -> Path:
    args = get_parser().parse_args(argv)
    data_root = initialize_data_tree()
    print(f"Initialized SunCET data tree: {data_root}")

    if not args.skip_metadata_download:
        fits_path, netcdf_zarr_path = download_live_metadata(data_root / "metadata")
        print(f"Downloaded FITS metadata: {fits_path}")
        print(f"Downloaded NetCDF/Zarr metadata: {netcdf_zarr_path}")

    report_calibration_status(data_root / "calibration")
    return data_root


if __name__ == "__main__":
    run()
