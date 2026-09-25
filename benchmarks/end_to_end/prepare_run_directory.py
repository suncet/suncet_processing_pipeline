#!/usr/bin/env python3
"""Prepare zero-copy Level 0.5 inputs for one end-to-end benchmark trial.

The staged X-band source and benchmark run directory must share a filesystem.
Two hard links are created before power sampling: one for the no-image Level
0.5 baseline and one for the full Level 0.5 phase.  The hybrid synthetic-image
handoff is only a path named in the benchmark plan, so it needs no staging
operation inside the measured interval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


_LEVEL0_5_DIRECTORIES = ("level0_5_skip_images", "level0_5_full")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def prepare_run_directory(run_directory: Path, xband_input: Path) -> Path:
    run_directory = Path(run_directory).expanduser().resolve()
    xband_input = Path(xband_input).expanduser().resolve()
    if not xband_input.is_file():
        raise FileNotFoundError(f"Staged X-band input not found: {xband_input}")
    if run_directory.exists():
        raise FileExistsError(
            f"Benchmark run directory already exists: {run_directory}"
        )

    run_directory.mkdir(parents=True, exist_ok=False)
    linked_paths: list[Path] = []
    try:
        for name in _LEVEL0_5_DIRECTORIES:
            directory = run_directory / name
            directory.mkdir()
            linked = directory / xband_input.name
            os.link(xband_input, linked)
            linked_paths.append(linked)

        source_stat = xband_input.stat()
        manifest_path = run_directory / "staging_manifest.json"
        manifest = {
            "schema": "suncet.end_to_end_staging",
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "method": "hard_link",
            "measurement_scope": "excluded_preflight",
            "source": {
                "path": str(xband_input),
                "size_bytes": source_stat.st_size,
                "device": source_stat.st_dev,
                "inode": source_stat.st_ino,
            },
            "links": [],
            "synthetic_handoff": {
                "method": "pre_staged_path_reference",
                "measurement_cost": "excluded/no copy",
            },
        }
        for linked in linked_paths:
            linked_stat = linked.stat()
            if (
                linked_stat.st_dev != source_stat.st_dev
                or linked_stat.st_ino != source_stat.st_ino
            ):
                raise RuntimeError(f"Input was not hard-linked as required: {linked}")
            manifest["links"].append(
                {
                    "path": str(linked),
                    "device": linked_stat.st_dev,
                    "inode": linked_stat.st_ino,
                }
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path
    except BaseException:
        # Only remove the paths created by this failed preflight.  The staged
        # source itself is never modified.
        for linked in reversed(linked_paths):
            linked.unlink(missing_ok=True)
        for name in reversed(_LEVEL0_5_DIRECTORIES):
            directory = run_directory / name
            if directory.is_dir():
                directory.rmdir()
        if run_directory.is_dir():
            run_directory.rmdir()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xband-input", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = prepare_run_directory(arguments.run_dir, arguments.xband_input)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
