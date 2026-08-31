import json

import pytest

from suncet_processing_pipeline.ctdb_snapshot import (
    CTDBSnapshotError,
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)


def _ctdb_tree(tmp_path):
    root = tmp_path / "ctdb"
    (root / "bus" / "packet_definitions").mkdir(parents=True)
    (root / "csie" / "decoders").mkdir(parents=True)
    (root / "bus" / "packet_definitions" / "ct_tlm.csv").write_text(
        "Packet,DataName\n1,mode\n", encoding="utf-8"
    )
    (root / "csie" / "decoders" / "gen_pkts.py").write_text(
        "PACKET = 538\n", encoding="utf-8"
    )
    return root


def test_manifest_is_relative_deterministic_and_exact(tmp_path):
    root = _ctdb_tree(tmp_path)
    (root / ".DS_Store").write_bytes(b"host metadata")
    cache = root / "csie" / "decoders" / "__pycache__"
    cache.mkdir()
    (cache / "gen_pkts.cpython-314.pyc").write_bytes(b"host cache")
    (root / "bus" / "packet_definitions" / "local.pyc").write_bytes(b"cache")
    first = build_manifest(root)
    second = build_manifest(root)

    assert first["tree_sha256"] == second["tree_sha256"]
    assert first["file_count"] == 2
    assert first["total_bytes"] > 0
    assert [entry["path"] for entry in first["files"]] == [
        "bus/packet_definitions/ct_tlm.csv",
        "csie/decoders/gen_pkts.py",
    ]
    assert str(root) not in json.dumps(first)


def test_manifest_inside_tree_is_excluded_and_verifies(tmp_path):
    root = _ctdb_tree(tmp_path)
    manifest_path = root / ".private_snapshot.json"
    written = write_manifest(root, manifest_path)

    assert load_manifest(manifest_path)["tree_sha256"] == written["tree_sha256"]
    result = verify_manifest(root, manifest_path)
    assert result.ok
    assert result.file_count == 2
    assert manifest_path.stat().st_mode & 0o077 == 0


def test_verify_reports_missing_unexpected_and_mismatched(tmp_path):
    source = _ctdb_tree(tmp_path / "source")
    manifest = tmp_path / "ctdb_snapshot.json"
    write_manifest(source, manifest)

    destination = _ctdb_tree(tmp_path / "destination")
    (destination / "bus" / "packet_definitions" / "ct_tlm.csv").write_text(
        "changed\n", encoding="utf-8"
    )
    (destination / "csie" / "decoders" / "gen_pkts.py").unlink()
    (destination / "unexpected.txt").write_text("extra", encoding="utf-8")

    result = verify_manifest(destination, manifest)
    assert not result.ok
    assert result.missing == ("csie/decoders/gen_pkts.py",)
    assert result.unexpected == ("unexpected.txt",)
    assert result.mismatched == ("bus/packet_definitions/ct_tlm.csv",)


def test_snapshot_refuses_symlinks_and_manifest_replacement(tmp_path):
    root = _ctdb_tree(tmp_path)
    (root / "outside-link").symlink_to(tmp_path / "outside")
    with pytest.raises(CTDBSnapshotError, match="symbolic links"):
        build_manifest(root)

    (root / "outside-link").unlink()
    manifest = tmp_path / "manifest.json"
    write_manifest(root, manifest)
    with pytest.raises(CTDBSnapshotError, match="Refusing to replace"):
        write_manifest(root, manifest)


def test_snapshot_fails_closed_on_existing_writer_lock(tmp_path):
    root = _ctdb_tree(tmp_path)
    manifest = tmp_path / "manifest.json"
    lock = manifest.with_name(f".{manifest.name}.lock")
    lock.write_text("held\n", encoding="utf-8")

    with pytest.raises(CTDBSnapshotError, match="holds the lock"):
        write_manifest(root, manifest)
    assert not manifest.exists()
    assert lock.read_text(encoding="utf-8") == "held\n"


def test_snapshot_refuses_public_data_tree(tmp_path, monkeypatch):
    root = _ctdb_tree(tmp_path)
    public_root = tmp_path / "public"
    public_root.mkdir()
    monkeypatch.setenv("suncet_data", str(public_root))

    with pytest.raises(CTDBSnapshotError, match="must not be written"):
        write_manifest(root, public_root / "ctdb_snapshot.json")
