"""Tests for the portable SunCET data-root contract."""

from pathlib import Path
import re

import pytest

from .. import data_paths


def test_data_root_is_required(monkeypatch):
    monkeypatch.delenv(data_paths.SUNCET_DATA_ENV, raising=False)

    with pytest.raises(data_paths.SuncetDataPathError, match="is not set"):
        data_paths.get_data_root()


def test_data_root_must_be_absolute(monkeypatch):
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, "relative/data")

    with pytest.raises(data_paths.SuncetDataPathError, match="absolute path"):
        data_paths.get_data_root()


def test_ctdb_root_is_separate(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    ctdb_root = tmp_path / "private-ctdb"
    data_root.mkdir()
    ctdb_root.mkdir()
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(data_root))
    monkeypatch.setenv(data_paths.SUNCET_CTDB_ENV, str(ctdb_root))

    assert data_paths.get_data_root() == data_root
    assert data_paths.resolve_ctdb_root("${suncet_ctdb}") == ctdb_root
    assert not data_paths.get_ctdb_root().is_relative_to(data_root)


def test_ctdb_root_cannot_be_inside_public_data_tree(tmp_path, monkeypatch):
    ctdb_root = tmp_path / "private-ctdb"
    ctdb_root.mkdir()
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(tmp_path))
    monkeypatch.setenv(data_paths.SUNCET_CTDB_ENV, str(ctdb_root))

    with pytest.raises(data_paths.SuncetDataPathError, match="must not overlap"):
        data_paths.resolve_ctdb_root("${suncet_ctdb}")


def test_managed_paths_follow_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(tmp_path))

    assert data_paths.data_path("test_data", "capture.bin") == (
        tmp_path / "test_data" / "capture.bin"
    )
    assert data_paths.processing_run_path("pass-001") == (
        tmp_path / "processing_runs" / "pass-001"
    )


@pytest.mark.parametrize("part", ["../outside", "/tmp/outside"])
def test_managed_paths_cannot_escape_data_root(tmp_path, monkeypatch, part):
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(tmp_path))

    with pytest.raises(data_paths.SuncetDataPathError):
        data_paths.data_path(part)


def test_explicit_absolute_input_is_allowed_when_root_is_declared(tmp_path, monkeypatch):
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(tmp_path))
    external = tmp_path.parent / "external.bin"

    assert data_paths.resolve_data_path(external) == external.resolve()


def test_run_name_must_be_one_component(tmp_path, monkeypatch):
    monkeypatch.setenv(data_paths.SUNCET_DATA_ENV, str(tmp_path))

    with pytest.raises(data_paths.SuncetDataPathError, match="one non-empty"):
        data_paths.processing_run_path("nested/run")


def test_python_and_config_sources_do_not_embed_workstation_paths():
    repository = Path(__file__).resolve().parents[2]
    forbidden_fragments = (
        "/" + "Users/",
        "/" + "home/",
        "~" + "/",
        "Library/" + "CloudStorage/",
    )
    direct_data_lookup = re.compile(
        r"(?:os\.getenv|os\.environ\.get)\(\s*['\"]suncet_data['\"]"
    )
    offenders = []
    for pattern in ("*.py", "*.ini", "*.ipynb"):
        for path in repository.rglob(pattern):
            text = path.read_text(encoding="utf-8")
            if any(fragment in text for fragment in forbidden_fragments) or (
                direct_data_lookup.search(text)
            ):
                offenders.append(str(path.relative_to(repository)))

    assert offenders == []
