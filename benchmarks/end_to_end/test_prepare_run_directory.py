from pathlib import Path
import json
import os
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_run_directory as prepare  # noqa: E402


def test_prepares_two_hard_links_and_records_zero_copy_handoff(tmp_path):
    source = tmp_path / "staged" / "xband.dat"
    source.parent.mkdir()
    source.write_bytes(b"flight-like input")
    run_directory = tmp_path / "runs" / "trial-01"

    manifest_path = prepare.prepare_run_directory(run_directory, source)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method"] == "hard_link"
    assert manifest["measurement_scope"] == "excluded_preflight"
    assert manifest["synthetic_handoff"]["measurement_cost"] == "excluded/no copy"
    assert len(manifest["links"]) == 2
    source_stat = source.stat()
    for name in prepare._LEVEL0_5_DIRECTORIES:
        linked = run_directory / name / source.name
        linked_stat = linked.stat()
        assert linked_stat.st_dev == source_stat.st_dev
        assert linked_stat.st_ino == source_stat.st_ino
    assert source_stat.st_nlink == 3


def test_refuses_existing_run_directory_without_modifying_it(tmp_path):
    source = tmp_path / "xband.dat"
    source.write_bytes(b"source")
    run_directory = tmp_path / "trial"
    run_directory.mkdir()
    sentinel = run_directory / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        prepare.prepare_run_directory(run_directory, source)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_missing_source_does_not_create_run_directory(tmp_path):
    run_directory = tmp_path / "trial"

    with pytest.raises(FileNotFoundError, match="not found"):
        prepare.prepare_run_directory(run_directory, tmp_path / "missing.dat")

    assert not run_directory.exists()
