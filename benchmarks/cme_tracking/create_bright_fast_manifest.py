"""Regenerate the reviewed historical bright-fast/no-jitter manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

from suncet_processing_pipeline.level4.cme_tracking.input import discover_fits_files
from suncet_processing_pipeline.level4.cme_tracking.manifest import (
    CadenceStatus,
    InputSourceKind,
    ManifestPathBase,
    ManifestReview,
    manifest_from_paths,
    write_manifest,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the historical manifest after a human has reviewed "
            "the selected files, ordering, cadence declaration, and notes."
        )
    )
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument(
        "--reviewed-at-utc",
        required=True,
        help="ISO 8601 timestamp with UTC offset or trailing Z.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    data_root = Path(os.environ["suncet_data"]).expanduser().resolve()
    directory = (
        data_root
        / "synthetic/level0/fits/cubesat/bright_fast/no jitter"
    )
    paths = discover_fits_files(directory)
    source_indices = [
        int(re.search(r"_(\d+)\.fits$", path.name).group(1))  # type: ignore[union-attr]
        for path in paths
    ]
    manifest = manifest_from_paths(
        paths,
        scenario_id="bright-fast-no-jitter-historical",
        source_kind=InputSourceKind.SYNTHETIC_BYPASS,
        review=ManifestReview(
            status="reviewed",
            reviewed_by=arguments.reviewed_by,
            reviewed_at_utc=arguments.reviewed_at_utc,
        ),
        relative_to=data_root,
        path_base=ManifestPathBase.SUNCET_DATA,
        cadence_seconds=30.0,
        cadence_status=CadenceStatus.VERIFIED,
        frame_numbers=range(len(paths)),
        source_indices=source_indices,
        notes=(
            "Curated ordered bright-fast/no-jitter historical simulator sequence. "
            "Relative timing is verified at 30 s from the confirmed one-hour "
            "source run, 10 s native model cadence, and every-third retained "
            "source indices. Samples run from 0 through 3570 s; omitted source "
            "index 360 is the 3600 s endpoint. DATE-OBS remains frozen, so "
            "absolute UTC is unavailable. Seventeen later headers have "
            "inconsistent RSUN-derived geometry; development runs must explicitly "
            "authorize use of the first-frame geometry."
        ),
        metadata={
            "truth_linkage_status": "unverified",
            "declared_directory_stage": "synthetic/level0",
            "fits_declared_level": "0.5",
            "source_index_step": 3,
            "source_cadence_seconds": 10.0,
            "source_run_duration_seconds": 3600.0,
            "source_terminal_index": 360,
            "selected_span_seconds": 3570.0,
            "timing_basis": (
                "The simulator provider confirmed a one-hour run. Underlying "
                "MHD metadata and simulator configuration establish a 10-second "
                "source cadence. The 120 retained frames occupy every third "
                "source index from 0 through 357, so their verified cadence is "
                "30 seconds and their sample times span 0 through 3570 seconds; "
                "the omitted source index 360 is at 3600 seconds."
            ),
            "expected_frame_count": len(paths),
            "instrument_effects": {"jitter": False},
        },
    )
    output = Path(__file__).with_name("manifests") / (
        "bright_fast_no_jitter_historical.json"
    )
    write_manifest(manifest, output)
    print(f"Wrote {len(manifest.frames)} reviewed frames to {output}")


if __name__ == "__main__":
    main()
