"""Create the reviewed manifest for the timestamp-correct simulator sequence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

from astropy.io import fits
from astropy.time import Time
import numpy as np

from suncet_processing_pipeline.level4.cme_tracking.input import discover_fits_files
from suncet_processing_pipeline.level4.cme_tracking.manifest import (
    InputSourceKind,
    ManifestPathBase,
    ManifestReview,
    manifest_from_paths,
    write_manifest,
)


_PATTERN = "config_default_no_particle_filter_OBS_*.fits"
_FILENAME_PATTERN = re.compile(
    r"^config_default_no_particle_filter_OBS_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})_"
    r"(?P<index>\d{3})\.fits$"
)
_EXPECTED_COUNT = 241
_EXPECTED_START = "2023-01-14T17:00:00.000"
_EXPECTED_END = "2023-01-14T18:00:00.000"
_EXPECTED_LAST_DATE_END = "2023-01-14T18:00:15.000"
_EXPECTED_CADENCE_SECONDS = 15.0
_EXPECTED_HEADER_VALUES: dict[str, object] = {
    "TITLE": "SunCET Level 1 Image",
    "TIMESYS": "UTC",
    "BZERO": 32768,
    "BSCALE": 1,
    "CRPIX1": 499.5,
    "CRPIX2": 375.5,
    "CDELT1": 9.6,
    "CDELT2": 9.6,
    "CUNIT1": "arcsec",
    "CUNIT2": "arcsec",
    "RSUN_OBS": 959.63,
    "RSUN_REF": 696000000,
    "IMAGEH": 1000,
    "IMAGEW": 752,
    "NAXIS1": 1000,
    "NAXIS2": 750,
}
_DATE_CARDS_WITH_UTC_COMMENTS = ("DATE-OBS", "DATE-END", "DATE")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the reviewed, hash-locked manifest after checking the "
            "timestamp-correct config-default/no-particle-filter simulator run."
        )
    )
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument(
        "--reviewed-at-utc",
        required=True,
        help="ISO 8601 timestamp with a UTC offset or trailing Z.",
    )
    return parser.parse_args()


def _validate_reviewed_header(
    header: fits.Header,
    path: Path,
    *,
    date_obs: str,
    date_end: str,
) -> None:
    """Reject drift in every header fact recorded in the reviewed manifest."""
    for key, expected in _EXPECTED_HEADER_VALUES.items():
        actual = header.get(key)
        if isinstance(expected, str) and actual is not None:
            actual = str(actual).strip()
        if actual != expected:
            raise RuntimeError(
                f"Expected {key}={expected!r} in {path.name}; found {actual!r}."
            )

    for key in _DATE_CARDS_WITH_UTC_COMMENTS:
        comment = str(header.comments[key]) if key in header else ""
        if "UTC" not in comment.upper():
            raise RuntimeError(
                f"Expected the documented UTC comment on {key} in {path.name}."
            )

    try:
        exposure_seconds = float(
            (
                Time(date_end, format="isot", scale="utc")
                - Time(date_obs, format="isot", scale="utc")
            ).to_value("s")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not parse DATE-OBS/DATE-END in {path.name}."
        ) from error
    if not np.isclose(
        exposure_seconds,
        _EXPECTED_CADENCE_SECONDS,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Expected DATE-END = DATE-OBS + 15 s in {path.name}; "
            f"found {exposure_seconds:.9g} s."
        )


def _validated_sequence(
    directory: Path,
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    paths = discover_fits_files(directory, _PATTERN)
    if len(paths) != _EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {_EXPECTED_COUNT} inputs matching {_PATTERN!r}; "
            f"found {len(paths)}."
        )

    timestamps: list[str] = []
    date_ends: list[str] = []
    source_indices: list[int] = []
    for path in paths:
        match = _FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected simulator filename: {path.name}")
        filename_timestamp = match.group("timestamp")
        source_index = int(match.group("index"))
        with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
            primary = hdul[0]
            if primary.verify_checksum() != 1 or primary.verify_datasum() != 1:
                raise RuntimeError(f"FITS checksum validation failed: {path}")
            header = primary.header
            if primary.data is None or primary.data.shape != (750, 1000):
                raise RuntimeError(f"Unexpected image shape: {path}")
            if primary.data.dtype != np.dtype("uint16"):
                raise RuntimeError(
                    f"Expected physical uint16 image values in {path}; "
                    f"found {primary.data.dtype}."
                )
            date_obs = str(header.get("DATE-OBS", "")).strip()
            date_beg = str(header.get("DATE-BEG", "")).strip()
            date_end = str(header.get("DATE-END", "")).strip()
            if date_obs != filename_timestamp or date_beg != filename_timestamp:
                raise RuntimeError(
                    f"Filename/DATE-OBS/DATE-BEG disagreement in {path.name}."
                )
            if str(header.get("TIMESYS", "")).strip().upper() != "UTC":
                raise RuntimeError(f"TIMESYS is not UTC in {path.name}.")
            if str(header.get("LEVEL", "")).strip() != "0.5":
                raise RuntimeError(f"Expected LEVEL=0.5 in {path.name}.")
            _validate_reviewed_header(
                header,
                path,
                date_obs=date_obs,
                date_end=date_end,
            )
        timestamps.append(date_obs)
        date_ends.append(date_end)
        source_indices.append(source_index)

    if source_indices != list(range(_EXPECTED_COUNT)):
        raise RuntimeError("Expected contiguous source indices 000 through 240.")
    times = Time(timestamps, format="isot", scale="utc")
    elapsed = np.asarray(times.unix - times.unix[0], dtype=np.float64)
    if timestamps[0] != _EXPECTED_START or timestamps[-1] != _EXPECTED_END:
        raise RuntimeError("Simulator start/end DATE-OBS values changed.")
    if not np.allclose(
        np.diff(elapsed),
        _EXPECTED_CADENCE_SECONDS,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("DATE-OBS cadence is not uniformly 15 seconds.")
    if date_ends[-1] != _EXPECTED_LAST_DATE_END:
        raise RuntimeError("Final DATE-END changed from the reviewed value.")
    return paths, tuple(source_indices)


def main() -> None:
    arguments = _arguments()
    data_root = Path(os.environ["suncet_data"]).expanduser().resolve()
    directory = data_root / "synthetic/level0/fits"
    paths, source_indices = _validated_sequence(directory)

    manifest = manifest_from_paths(
        paths,
        scenario_id="config-default-no-particle-filter-20230114",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=ManifestReview(
            status="reviewed",
            reviewed_by=arguments.reviewed_by,
            reviewed_at_utc=arguments.reviewed_at_utc,
        ),
        relative_to=data_root,
        path_base=ManifestPathBase.SUNCET_DATA,
        frame_numbers=range(len(paths)),
        source_indices=source_indices,
        notes=(
            "Reviewed timestamp-correct config-default/no-particle-filter "
            "simulator sequence. FITS DATE-OBS supplies authoritative UTC and "
            "agrees exactly with DATE-BEG and each filename: 241 sample starts "
            "at 15 s cadence from 2023-01-14T17:00:00 through 18:00:00 UTC "
            "(a 3600 s sample span); the final DATE-END is 18:00:15. These are "
            "synthetic LEVEL=0.5 images that explicitly bypass Level 2 PSF "
            "deconvolution and Level 3 geometric correction. The LEVEL=0.5 / "
            "TITLE='SunCET Level 1 Image' conflict, DATE-card comments mentioning "
            "TAI despite TIMESYS=UTC, IMAGEH/IMAGEW disagreement with NAXIS, and "
            "possible one-pixel CRPIX1 midpoint offset are retained as simulator "
            "documentation debt; no geometry override is applied."
        ),
        metadata={
            "truth_linkage_status": "unavailable",
            "declared_directory_stage": "synthetic/level0",
            "fits_declared_level": "0.5",
            "fits_title": "SunCET Level 1 Image",
            "input_pattern": _PATTERN,
            "simulator_configuration": "config_default",
            "particle_filter_enabled": False,
            "expected_frame_count": len(paths),
            "source_index_step": 1,
            "source_terminal_index": source_indices[-1],
            "expected_start_time_utc": _EXPECTED_START,
            "expected_end_time_utc": _EXPECTED_END,
            "expected_last_date_end_utc": _EXPECTED_LAST_DATE_END,
            "expected_cadence_seconds": _EXPECTED_CADENCE_SECONDS,
            "selected_sample_span_seconds": 3600.0,
            "image_shape_yx": [750, 1000],
            "timestamp_basis": (
                "FITS DATE-OBS, cross-checked against DATE-BEG and filename; "
                "TIMESYS=UTC"
            ),
            "fits_checksum_review": {
                "verified_frame_count": len(paths),
                "checksum_and_datasum_valid": True,
            },
            "header_caveats": [
                "LEVEL=0.5 conflicts with TITLE='SunCET Level 1 Image'",
                "DATE card comments mention TAI despite TIMESYS=UTC",
                "CRPIX1 places the declared center one pixel left of the array midpoint",
                "IMAGEH/IMAGEW disagree with NAXIS2/NAXIS1",
            ],
            "tracking_configuration_caveat": (
                "Reference-v0 association limits are expressed per frame, so "
                "their physical time windows differ from the earlier 30 s run."
            ),
        },
    )
    output = Path(__file__).with_name("manifests") / (
        "config_default_no_particle_filter_20230114.json"
    )
    write_manifest(manifest, output)
    print(f"Wrote {len(manifest.frames)} reviewed frames to {output}")


if __name__ == "__main__":
    main()
