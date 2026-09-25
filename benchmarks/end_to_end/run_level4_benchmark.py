#!/usr/bin/env python3
"""Run benchmark Level 4 on synthetic Level 3 pass-through products.

This entry point is deliberately separate from the production Level 4 CLI.
The input images have passed through real Level 1 and Level 2 processing, but
the benchmark-only Level 3 adapter does not apply a geometric correction.  We
therefore retain all three scientifically important facts in the Level 4
provenance and quality flags:

* the source images are synthetic;
* Level 2 PSF deconvolution was applied; and
* Level 3 geometric correction was not applied.

FITS ``DATE-OBS`` values provide the time coordinate.  Diagnostic PNG plots,
overlays, and movie encoding are intentionally unavailable here so they cannot
accidentally enter an energy measurement.  The authoritative ECSV science
tables and JSON provenance products are still written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

from astropy.io import fits

# Permit direct execution from a source checkout as well as module execution
# from an installed package environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from suncet_processing_pipeline.level4.cme_tracking.config import (  # noqa: E402
    read_configuration,
)
from suncet_processing_pipeline.level4.cme_tracking.input import (  # noqa: E402
    ImageSequence,
    discover_fits_files,
    load_fits_sequence,
)
from suncet_processing_pipeline.level4.cme_tracking.manifest import (  # noqa: E402
    CorrectionState,
    InputSourceKind,
    UpstreamProcessing,
)
from suncet_processing_pipeline.level4.cme_tracking.pipeline import (  # noqa: E402
    run_known_window,
    write_known_window_products,
)


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LEVEL3_VALUES = {"3", "3.0", "L3", "LEVEL3"}


def _normalized_level(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def _require_boolean(header: fits.Header, key: str, expected: bool, path: Path) -> None:
    value = header.get(key)
    if value is not expected:
        raise ValueError(
            f"Benchmark Level 3 input {path} must declare {key}={expected}; "
            f"got {value!r}"
        )


def validate_benchmark_level3_header(path: str | Path) -> None:
    """Reject a Level 3 file whose pass-through status is ambiguous.

    Only headers are read here.  Pixel checksums are generated and validated
    by the preceding Level 3 adapter; rescanning every array before Level 4
    would add duplicate image I/O to the measured Level 4 phase.
    """

    resolved = Path(path).expanduser().resolve()
    header = fits.getheader(resolved, ext=0)
    if _normalized_level(header.get("LEVEL")) not in _LEVEL3_VALUES:
        raise ValueError(
            f"Benchmark Level 4 requires LEVEL=3 inputs; {resolved} declares "
            f"{header.get('LEVEL')!r}"
        )
    _require_boolean(header, "DECONV", True, resolved)
    _require_boolean(header, "BMRKONLY", True, resolved)
    _require_boolean(header, "L3PASS", True, resolved)
    _require_boolean(header, "L3GEOM", False, resolved)
    _require_boolean(header, "L3DARK", False, resolved)
    if str(header.get("L3CORR", "")).strip().upper() != "NONE":
        raise ValueError(
            f"Benchmark Level 3 input {resolved} must declare L3CORR='NONE'; "
            f"got {header.get('L3CORR')!r}"
        )
    if str(header.get("TIMESYS", "")).strip().upper() != "UTC":
        raise ValueError(
            f"Benchmark Level 3 input {resolved} must declare TIMESYS='UTC'"
        )
    if not str(header.get("DATE-OBS", "")).strip():
        raise ValueError(
            f"Benchmark Level 3 input {resolved} is missing DATE-OBS"
        )
    for key in ("CHECKSUM", "DATASUM"):
        if key not in header:
            raise ValueError(
                f"Benchmark Level 3 input {resolved} is missing {key}"
            )


def load_benchmark_sequence(
    input_directory: str | Path,
    *,
    pattern: str,
    scenario_id: str,
) -> ImageSequence:
    """Load pass-through files with their honest upstream-processing state."""

    paths = discover_fits_files(input_directory, pattern)
    for path in paths:
        validate_benchmark_level3_header(path)
    return load_fits_sequence(
        paths,
        scenario_id=scenario_id,
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        # No cadence override: the corrected simulator timestamps are used.
        cadence_seconds=None,
        upstream_processing=UpstreamProcessing(
            level2_psf_deconvolution=CorrectionState.APPLIED,
            level3_geometric_correction=CorrectionState.NOT_APPLIED,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument(
        "--event-id",
        help="Deterministic output directory name; defaults to <scenario-id>-event-001",
    )
    parser.add_argument("--pattern", default="*.fits", help="Nonrecursive FITS glob")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing event product directory",
    )
    return parser


def _validate_identifier(
    parser: argparse.ArgumentParser,
    value: str,
    name: str,
) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        parser.error(
            f"{name} must be one portable component containing letters, numbers, "
            "'.', '_', or '-'"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _validate_identifier(parser, arguments.scenario_id, "--scenario-id")
    event_id = arguments.event_id or f"{arguments.scenario_id}-event-001"
    _validate_identifier(parser, event_id, "--event-id")

    sequence = load_benchmark_sequence(
        arguments.input_directory,
        pattern=arguments.pattern,
        scenario_id=arguments.scenario_id,
    )
    configuration = read_configuration(arguments.config)
    run = run_known_window(sequence, configuration)
    event_directory = write_known_window_products(
        run,
        arguments.output_root,
        event_id,
        repository=Path(__file__).resolve().parents[2],
        include_diagnostic_plots=False,
        include_diagnostic_movie=False,
        overwrite=arguments.overwrite,
    )
    status = "detected" if run.front.event_detected else "not detected"
    print(f"CME known-window benchmark result: {status}")
    print(f"Products: {event_directory}")
    if run.kinematics_error:
        print(f"Kinematics: {run.kinematics_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
