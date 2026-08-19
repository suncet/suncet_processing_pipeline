"""Level 0.5 for each immediate test_data subfolder whose name starts with 2026.

Each folder is processed independently in a subprocess; if one fails (e.g. bad JPEG-LS),
it is skipped and the rest continue. If any folder's ``process()`` runs longer than
``PER_FOLDER_TIMEOUT_SEC`` wall time, the whole script **exits** immediately (no merge).

HDF5/FITS/PNGs go under ``<that folder>/level0_5/`` (same as ``make_level0_5`` when
``data_to_process_path`` points at that folder).

After all folders (when nothing times out), telemetry and DSPS HDF5 files from **successful**
runs are merged into ``<test_data>/_level0_5_merged_2026_subfolders/level0_5/`` (same
layout as ``Level0_5.save_packet_data_to_hdf5``).
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import sys

from suncet_processing_pipeline.level05_merge_utils import merge_per_folder_h5s
from suncet_processing_pipeline.level05_subprocess_runner import level05_folder_worker
from suncet_processing_pipeline.make_level0_5 import discover_level0_5_input_files
from suncet_processing_pipeline.config_parser import Config


PER_FOLDER_TIMEOUT_SEC = 30


def _run_folder_deadline(folder: str, name: str, file_paths: list, config_path: str) -> str | None:
    """
    Spawn ``Level0_5.process()`` with a wall-clock limit.

    Returns:
        ``None`` on success.
        ``"err", trace`` style is handled via message string: returns error message on worker error.
        On timeout this function does not return; it calls ``sys.exit(2)``.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    cfg = os.path.abspath(config_path)
    proc = ctx.Process(
        target=level05_folder_worker,
        args=(q, cfg, folder, file_paths),
    )
    proc.start()
    proc.join(PER_FOLDER_TIMEOUT_SEC)
    if proc.is_alive():
        proc.terminate()
        proc.join(8)
        if proc.is_alive():
            proc.kill()
            proc.join(3)
        print(
            f"[TIMEOUT] {name} exceeded {PER_FOLDER_TIMEOUT_SEC}s — exiting run.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.exitcode != 0:
        return f"subprocess exited with code {proc.exitcode}"
    try:
        status, payload = q.get(timeout=5)
    except queue.Empty:
        return "worker finished but sent no queue message"
    if status == "ok":
        return None
    if status == "err":
        assert isinstance(payload, str)
        return payload
    return f"unexpected status {status!r}"


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(repo, "suncet_processing_pipeline", "config_files", "config_default.ini")
    config = Config(config_path)

    sd = os.environ.get("suncet_data")
    if not sd:
        sys.exit("suncet_data environment variable is not set")

    test_data = os.path.join(sd, "test_data")
    if not os.path.isdir(test_data):
        sys.exit(f"test_data directory does not exist: {test_data}")

    subdirs = sorted(
        x
        for x in os.listdir(test_data)
        if x.startswith("2026") and os.path.isdir(os.path.join(test_data, x))
    )

    ok, failed, empty = 0, 0, 0
    succeeded_names = []
    for name in subdirs:
        folder = os.path.join(test_data, name)
        file_paths = discover_level0_5_input_files(folder, ignore_realtime=config.ignore_realtime)
        if not file_paths:
            print(f"[skip empty] {name}")
            empty += 1
            continue
        print(f"[run] {name} ({len(file_paths)} files) -> {os.path.join(folder, 'level0_5')} [limit {PER_FOLDER_TIMEOUT_SEC}s]")
        err = _run_folder_deadline(folder, name, file_paths, config_path)
        if err is None:
            ok += 1
            succeeded_names.append(name)
            print(f"[ok] {name}")
        else:
            print(f"[FAILED] {name}")
            print(err)
            failed += 1

    print(
        f"Done. {ok} succeeded, {failed} failed (skipped), {empty} had no telemetry, "
        f"{len(subdirs)} folders total."
    )

    merge_root = os.path.join(test_data, "_level0_5_merged_2026_subfolders")
    if succeeded_names:
        print(f"Merging HDF5 from {len(succeeded_names)} successful folder(s) -> {merge_root}")
        merge_per_folder_h5s(test_data, succeeded_names, config, merge_root)
    else:
        print("No successful folders; skipping HDF5 merge.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
