"""Tests for incremental processing across Level 0 delivery routes."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from ..incremental_level0 import (
    STANDARD_ROUTE_MODES,
    _bootstrap_entries_from_manifests,
    _plan_incremental_inputs,
    discover_level0_routes,
    operational_arguments,
    operational_main,
)


def _entry(path):
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_default_plan_processes_only_new_inputs(tmp_path):
    old = tmp_path / "xbandDemod_old.dat"
    new = tmp_path / "xbandDemod_new.dat"
    old.write_bytes(b"old")
    new.write_bytes(b"new")

    entries, pending, changed, removed = _plan_incremental_inputs(
        {old.name: _entry(old)},
        {old.name: old, new.name: new},
        force_rebuild=False,
    )

    assert set(entries) == {old.name}
    assert pending == [new.name]
    assert changed == []
    assert removed == []


def test_operational_defaults_use_level0_root(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))

    arguments = operational_arguments([])

    assert arguments[:2] == [
        "--folder",
        str(tmp_path / "level0"),
    ]


def test_operational_help_does_not_require_data_root(monkeypatch, capsys):
    monkeypatch.delenv("suncet_data", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        operational_main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text
    assert "--rebuild" in help_text


def test_operational_first_positional_folder_and_options_override_defaults(tmp_path):
    arguments = operational_arguments(
        [str(tmp_path), "--prefix=custom", "--input-mode", "xband"]
    )

    assert arguments == [
        "--folder",
        str(tmp_path),
        "--prefix=custom",
        "--input-mode",
        "xband",
    ]


def test_standard_level0_routes_map_xband_and_uhf(tmp_path):
    level0 = tmp_path / "level0"
    for name in STANDARD_ROUTE_MODES:
        (level0 / name).mkdir(parents=True)
    (level0 / "leafspace_ground_stations" / "xbandDemod_capture").write_bytes(b"x")
    for name in ("fairbanks_ground_station", "lasp_ground_station", "satnogs"):
        (level0 / name / "ccsds_2026_001_00_00_00").write_bytes(b"u")
    args = SimpleNamespace(input_mode="auto", prefix=None)

    routes = discover_level0_routes(level0, args)

    assert [(route.route_id, route.input_mode) for route in routes] == [
        ("fairbanks_ground_station", "uhf"),
        ("lasp_ground_station", "uhf"),
        ("leafspace_ground_stations", "xband"),
        ("satnogs", "uhf"),
    ]


def test_override_folder_discovers_all_supported_sources(tmp_path):
    (tmp_path / "xbandDemod_capture").write_bytes(b"x")
    (tmp_path / "ccsds_hardline").write_bytes(b"h")
    uhf = tmp_path / "hydra_uhf_rundirs" / "pass"
    uhf.mkdir(parents=True)
    (uhf / "ccsds_2026_001_00_00_00").write_bytes(b"u")
    args = SimpleNamespace(input_mode="auto", prefix=None)

    routes = discover_level0_routes(tmp_path, args)

    assert [(route.route_id, route.input_mode) for route in routes] == [
        ("xband", "xband"),
        ("hardline", "ccsds"),
        ("uhf", "uhf"),
    ]


def test_default_plan_reprocesses_only_replaced_input(tmp_path):
    unchanged = tmp_path / "xbandDemod_unchanged.dat"
    replaced = tmp_path / "xbandDemod_replaced.dat"
    unchanged.write_bytes(b"unchanged")
    replaced.write_bytes(b"before")
    entries = {unchanged.name: _entry(unchanged), replaced.name: _entry(replaced)}
    replaced.write_bytes(b"after-with-a-different-size")

    _entries, pending, changed, removed = _plan_incremental_inputs(
        entries,
        {unchanged.name: unchanged, replaced.name: replaced},
        force_rebuild=False,
    )

    assert pending == [replaced.name]
    assert changed == [replaced.name]
    assert removed == []


def test_rebuild_plan_explicitly_processes_every_current_input(tmp_path):
    first = tmp_path / "xbandDemod_first.dat"
    second = tmp_path / "xbandDemod_second.dat"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    entries, pending, changed, removed = _plan_incremental_inputs(
        {first.name: _entry(first)},
        {first.name: first, second.name: second},
        force_rebuild=True,
    )

    assert entries == {}
    assert pending == sorted([first.name, second.name])
    assert changed == []
    assert removed == []


def test_removed_input_is_forgotten_without_replaying_remaining_files(tmp_path):
    present = tmp_path / "xbandDemod_present.dat"
    present.write_bytes(b"present")

    entries, pending, changed, removed = _plan_incremental_inputs(
        {
            present.name: _entry(present),
            "xbandDemod_removed.dat": {
                "size_bytes": 1,
                "mtime_ns": 1,
                "sha256": "gone",
            },
        },
        {present.name: present},
        force_rebuild=False,
    )

    assert set(entries) == {present.name}
    assert pending == []
    assert changed == []
    assert removed == ["xbandDemod_removed.dat"]


def test_manifest_recovery_combines_independent_successful_runs(tmp_path):
    manifest_dir = tmp_path / "processing_manifests"
    manifest_dir.mkdir()
    config = SimpleNamespace(
        version_pipeline="2.0.1",
        version_bus="2.0.5",
        version_csie="1.1.8",
        version_dsps="6.09",
    )
    paths = {}
    for index in (1, 2):
        path = tmp_path / f"xbandDemod_{index}.dat"
        path.write_bytes(f"capture-{index}".encode())
        paths[path.name] = path
        manifest = {
            "status": "succeeded",
            "invocation": {
                "arguments": {
                    "duckdb_source_key": f"original-source-{index}",
                    "input_mode": "xband",
                    "skip_duckdb": False,
                }
            },
            "resolved_configuration": {
                "version_pipeline": config.version_pipeline,
                "version_bus": config.version_bus,
                "version_csie": config.version_csie,
                "version_dsps": config.version_dsps,
            },
            "inputs": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        }
        (manifest_dir / f"processing_run_{index}.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    recovered = _bootstrap_entries_from_manifests(tmp_path, paths, config)

    assert set(recovered) == set(paths)
    assert recovered["xbandDemod_1.dat"]["source_key"] == "original-source-1"
    assert recovered["xbandDemod_2.dat"]["source_key"] == "original-source-2"
