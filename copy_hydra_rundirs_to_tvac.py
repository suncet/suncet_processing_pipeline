"""Throwaway helper to collect Hydra rundir files into the TVAC test folder.

The defaults copy files from ``suncet_test_laptop_hydra_rundirs`` for rundirs at
or after ``2026_149*`` into the ``2026-05-29_thermal_vacuum_tvac`` folder.

Files that already exist in the destination are skipped. ``runTimeNotes`` files
are skipped. Timestamped ``EventLog_*`` files are merged into one ``eventlog``
file in the destination and then removed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    "/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/"
    "suncet_test_laptop_hydra_rundirs"
)
DEFAULT_TARGET_ROOT = Path(
    "/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/"
    "2026-05-29_thermal_vacuum_tvac"
)
DEFAULT_START = "2026_149"
MERGED_EVENTLOG_NAME = "eventlog"
MANIFEST_NAME = ".hydra_to_tvac_copy_manifest.json"
RUNDIR_RE = re.compile(r"^(?P<year>\d{4})_(?P<doy>\d{3})(?:_|$)")


def parse_start(value: str) -> tuple[int, int]:
    match = RUNDIR_RE.match(value)
    if not match:
        raise ValueError(f"Start value must look like yyyy_doy, got: {value!r}")
    return int(match.group("year")), int(match.group("doy"))


def rundir_key(path: Path) -> tuple[int, int, str] | None:
    match = RUNDIR_RE.match(path.name)
    if not match:
        return None
    return int(match.group("year")), int(match.group("doy")), path.name


def selected_rundirs(source_root: Path, start: str) -> list[Path]:
    start_year, start_doy = parse_start(start)
    rundirs: list[tuple[tuple[int, int, str], Path]] = []
    for path in source_root.iterdir():
        if not path.is_dir():
            continue
        key = rundir_key(path)
        if key is None:
            continue
        year, doy, _name = key
        if (year, doy) >= (start_year, start_doy):
            rundirs.append((key, path))
    return [path for _key, path in sorted(rundirs)]


def is_runtime_notes(path: Path) -> bool:
    return "runtimenotes" in path.name.lower()


def is_eventlog(path: Path) -> bool:
    return path.name.lower().startswith("eventlog")


def load_manifest(target_root: Path) -> dict:
    manifest_path = target_root / MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "copied_files": [],
            "eventlog_names_merged": [],
        }
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.setdefault("copied_files", [])
    manifest.setdefault("eventlog_names_merged", [])
    return manifest


def save_manifest(target_root: Path, manifest: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    manifest_path = target_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_file_bytes(output_handle, input_path: Path) -> bool:
    """Append a file and return whether the resulting output ends with a newline."""
    ends_with_newline = True
    with input_path.open("rb") as input_handle:
        while True:
            chunk = input_handle.read(1024 * 1024)
            if not chunk:
                break
            output_handle.write(chunk)
            ends_with_newline = chunk.endswith(b"\n")
    return ends_with_newline


def merge_eventlogs(target_root: Path, manifest: dict, *, dry_run: bool) -> tuple[int, int]:
    output_path = target_root / MERGED_EVENTLOG_NAME
    temp_path = target_root / f".{MERGED_EVENTLOG_NAME}.tmp"
    eventlog_paths = sorted(
        path
        for path in target_root.iterdir()
        if path.is_file()
        and is_eventlog(path)
        and path.name != MERGED_EVENTLOG_NAME
        and path.name != temp_path.name
    )
    if not eventlog_paths:
        return 0, 0

    total_bytes = 0
    if dry_run:
        for path in eventlog_paths:
            total_bytes += path.stat().st_size
        return len(eventlog_paths), total_bytes

    output_ends_with_newline = True
    with temp_path.open("wb") as output_handle:
        if output_path.is_file():
            output_ends_with_newline = append_file_bytes(output_handle, output_path)
        for path in eventlog_paths:
            if not output_ends_with_newline:
                output_handle.write(b"\n")
            output_ends_with_newline = append_file_bytes(output_handle, path)
            total_bytes += path.stat().st_size

    temp_path.replace(output_path)
    merged_names = set(manifest["eventlog_names_merged"])
    for path in eventlog_paths:
        merged_names.add(path.name)
        path.unlink()
    manifest["eventlog_names_merged"] = sorted(merged_names)
    return len(eventlog_paths), total_bytes


def copy_rundir_files(
    source_root: Path,
    target_root: Path,
    *,
    start: str,
    dry_run: bool,
) -> dict[str, int]:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not target_root.is_dir():
        raise FileNotFoundError(f"Target root not found: {target_root}")

    manifest = load_manifest(target_root)
    merged_eventlog_names = set(manifest["eventlog_names_merged"])
    copied_files = set(manifest["copied_files"])
    stats = {
        "rundirs": 0,
        "files_seen": 0,
        "runtime_notes_skipped": 0,
        "existing_skipped": 0,
        "merged_eventlog_skipped": 0,
        "copied": 0,
        "copied_bytes": 0,
        "eventlogs_merged": 0,
        "eventlog_bytes_merged": 0,
    }

    for rundir in selected_rundirs(source_root, start):
        stats["rundirs"] += 1
        for source_path in sorted(rundir.iterdir()):
            if not source_path.is_file():
                continue
            stats["files_seen"] += 1
            if is_runtime_notes(source_path):
                stats["runtime_notes_skipped"] += 1
                continue
            destination_path = target_root / source_path.name
            if destination_path.exists():
                stats["existing_skipped"] += 1
                continue
            if is_eventlog(source_path) and source_path.name in merged_eventlog_names:
                stats["merged_eventlog_skipped"] += 1
                continue

            source_rel = str(source_path.relative_to(source_root))
            stats["copied"] += 1
            stats["copied_bytes"] += source_path.stat().st_size
            if not dry_run:
                shutil.copy2(source_path, destination_path)
                copied_files.add(source_rel)

    manifest["copied_files"] = sorted(copied_files)
    eventlogs_merged, eventlog_bytes_merged = merge_eventlogs(
        target_root,
        manifest,
        dry_run=dry_run,
    )
    stats["eventlogs_merged"] = eventlogs_merged
    stats["eventlog_bytes_merged"] = eventlog_bytes_merged
    save_manifest(target_root, manifest, dry_run=dry_run)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy Hydra rundir files into the TVAC folder and merge EventLogs."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Earliest rundir day to include, in yyyy_doy form. Default: 2026_149.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying, merging, or deleting files.",
    )
    args = parser.parse_args()

    stats = copy_rundir_files(
        args.source_root,
        args.target_root,
        start=args.start,
        dry_run=args.dry_run,
    )
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: source root: {args.source_root}")
    print(f"{mode}: target root: {args.target_root}")
    print(f"{mode}: start rundir day: {args.start}")
    for key, value in stats.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
