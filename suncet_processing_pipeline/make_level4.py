"""Create SunCET Level 4 science products.

The first implemented product is a known-window CME front and projected
kinematics track. Continuous event discovery remains a later layer; this
dispatcher never labels known-window tracking as autonomous discovery.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Sequence

from .data_paths import data_path
from .level4.cme_tracking.config import CMETrackingConfig, read_configuration
from .level4.cme_tracking.input import (
    discover_fits_files,
    load_level3_sequence,
    load_sequence_from_manifest,
    load_synthetic_sequence,
)
from .level4.cme_tracking.pipeline import (
    run_known_window,
    write_known_window_products,
)


_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="product", required=True)
    cme = subcommands.add_parser(
        "cme-track",
        help="Track one expected CME in an explicitly supplied time window.",
    )
    source = cme.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="Reviewed JSON scenario manifest (synthetic or production Level 3).",
    )
    source.add_argument(
        "--synthetic-directory",
        type=Path,
        help=(
            "Direct synthetic development directory; a reviewed manifest is "
            "preferred."
        ),
    )
    source.add_argument(
        "--level3-directory",
        type=Path,
        help="Production Level 3 FITS directory with valid monotonic timestamps.",
    )
    cme.add_argument("--pattern", default="*.fits", help="Nonrecursive FITS glob.")
    cme.add_argument(
        "--scenario-id",
        help="Required with direct-directory input; manifests provide their own ID.",
    )
    cme.add_argument(
        "--assumed-cadence-seconds",
        type=float,
        help=(
            "Override invalid historical synthetic timestamps with a fixed "
            "cadence; omit to use strict FITS DATE-OBS timing."
        ),
    )
    cme.add_argument(
        "--data-root",
        type=Path,
        help="Override suncet_data only when resolving a manifest's portable paths.",
    )
    cme.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help=(
            "Skip manifest SHA-256 verification; products are explicitly flagged "
            "as input-integrity unverified."
        ),
    )
    cme.add_argument(
        "--allow-inconsistent-synthetic-geometry",
        action="store_true",
        help=(
            "Use first-frame geometry while recording later synthetic-header "
            "differences; forbidden for production Level 3."
        ),
    )
    cme.add_argument(
        "--config",
        type=Path,
        help="Complete JSON tracker configuration; defaults are provisional.",
    )
    cme.add_argument(
        "--event-id",
        help="Output identifier; defaults to <scenario-id>-event-001.",
    )
    cme.add_argument(
        "--output-root",
        type=Path,
        help="Parent directory; default is $suncet_data/level4/cme_tracking.",
    )
    cme.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in an existing event directory.",
    )
    cme.add_argument(
        "--diagnostic-movie",
        action="store_true",
        help=(
            "Encode every image with the retained front and synchronized "
            "kinematic plots; requires ffmpeg and adds processing time."
        ),
    )
    cme.add_argument(
        "--movie-fps",
        type=_positive_float,
        default=10.0,
        help=(
            "Diagnostic movie playback rate (default: 10); this does not change "
            "the scientific cadence."
        ),
    )
    return parser


def _load_sequence(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    if arguments.manifest is not None:
        if (
            arguments.scenario_id is not None
            or arguments.assumed_cadence_seconds is not None
        ):
            parser.error(
                "--scenario-id and --assumed-cadence-seconds may not accompany "
                "--manifest"
            )
        sequence, _manifest = load_sequence_from_manifest(
            arguments.manifest,
            data_root=arguments.data_root,
            verify_hashes=not arguments.skip_hash_verification,
            allow_inconsistent_geometry=(
                arguments.allow_inconsistent_synthetic_geometry
            ),
        )
        if (
            arguments.allow_inconsistent_synthetic_geometry
            and sequence.source_kind.value == "production_level3"
        ):
            parser.error(
                "--allow-inconsistent-synthetic-geometry is forbidden for "
                "production Level 3"
            )
        return sequence

    if not arguments.scenario_id:
        parser.error("--scenario-id is required with a direct input directory")
    directory = arguments.synthetic_directory or arguments.level3_directory
    paths = discover_fits_files(directory, arguments.pattern)
    if arguments.synthetic_directory is not None:
        return load_synthetic_sequence(
            paths,
            scenario_id=arguments.scenario_id,
            assumed_cadence_seconds=arguments.assumed_cadence_seconds,
            allow_inconsistent_geometry=(
                arguments.allow_inconsistent_synthetic_geometry
            ),
        )
    if arguments.assumed_cadence_seconds is not None:
        parser.error("Production --level3-directory forbids a cadence override")
    if arguments.allow_inconsistent_synthetic_geometry:
        parser.error(
            "--allow-inconsistent-synthetic-geometry is forbidden for production "
            "Level 3"
        )
    return load_level3_sequence(paths, scenario_id=arguments.scenario_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected Level 4 product and return a process exit code."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.product != "cme-track":  # pragma: no cover - argparse owns this
        parser.error(f"Unsupported Level 4 product: {arguments.product}")

    sequence = _load_sequence(arguments, parser)
    configuration = (
        read_configuration(arguments.config)
        if arguments.config is not None
        else CMETrackingConfig()
    )
    event_id = arguments.event_id or f"{sequence.scenario_id}-event-001"
    if not _EVENT_ID_PATTERN.fullmatch(event_id):
        parser.error(
            "--event-id must be one portable component containing letters, "
            "numbers, '.', '_', or '-'"
        )
    output_root = arguments.output_root or data_path("level4", "cme_tracking")

    run = run_known_window(sequence, configuration)
    event_directory = write_known_window_products(
        run,
        output_root,
        event_id,
        repository=Path(__file__).resolve().parents[1],
        include_diagnostic_movie=arguments.diagnostic_movie,
        movie_fps=arguments.movie_fps,
        overwrite=arguments.overwrite,
    )
    status = "detected" if run.front.event_detected else "not detected"
    print(f"CME known-window result: {status}")
    print(f"Products: {event_directory}")
    if run.kinematics_error:
        print(f"Kinematics: {run.kinematics_error}")
    # A processed window with no detected event is a valid scientific result,
    # not an execution failure that an operational scheduler should retry.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
