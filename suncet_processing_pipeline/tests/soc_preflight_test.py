from pathlib import Path

import pytest

from suncet_processing_pipeline.soc_preflight import (
    GIB,
    DiskUsage,
    InodeUsage,
    MountInfo,
    SOCPreflightError,
    StorageTarget,
    check_storage,
    check_storage_target,
    load_storage_targets,
)


def _target(path: Path, **overrides) -> StorageTarget:
    values = {
        "name": "data",
        "path": path,
        "expected_mountpoint": None,
        "warning_free_bytes": 300 * GIB,
        "critical_free_bytes": 150 * GIB,
        "warning_free_percent": 20.0,
        "critical_free_percent": 10.0,
        "work_multiplier": 3.0,
        "accepts_workload": True,
    }
    values.update(overrides)
    return StorageTarget(**values)


def test_storage_check_applies_expansion_reserve_and_thresholds(tmp_path):
    usage = lambda _path: DiskUsage(2_000 * GIB, 1_000 * GIB, 1_000 * GIB)
    result = check_storage_target(
        _target(tmp_path), planned_input_bytes=100 * GIB, usage_provider=usage
    )

    assert result.status == "OK"
    assert result.reserved_work_bytes == 300 * GIB
    assert result.projected_free_bytes == 700 * GIB

    warning = check_storage_target(
        _target(tmp_path), planned_input_bytes=250 * GIB, usage_provider=usage
    )
    assert warning.status == "WARNING"

    critical = check_storage_target(
        _target(tmp_path), planned_input_bytes=300 * GIB, usage_provider=usage
    )
    assert critical.status == "CRITICAL"


def test_expected_mountpoint_failure_is_critical(tmp_path):
    usage = lambda _path: DiskUsage(2_000 * GIB, 500 * GIB, 1_500 * GIB)
    result = check_storage_target(
        _target(tmp_path, expected_mountpoint=tmp_path),
        usage_provider=usage,
        mount_checker=lambda _path: False,
    )

    assert result.status == "CRITICAL"
    assert "not mounted" in result.messages[0]


def test_load_storage_targets_expands_environment_and_validates(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("suncet_data", str(data_root))
    config = tmp_path / "soc_operations.ini"
    config.write_text(
        "[storage:data]\n"
        "path = ${suncet_data}\n"
        "expected_mountpoint = /srv/suncet\n"
        "warning_free_gib = 300\n"
        "critical_free_gib = 150\n"
        "warning_free_percent = 20\n"
        "critical_free_percent = 10\n"
        "work_multiplier = 3\n"
        "accepts_workload = true\n",
        encoding="utf-8",
    )

    targets = load_storage_targets(config)
    assert targets[0].path == data_root.resolve()
    assert targets[0].expected_mountpoint == Path("/srv/suncet")
    assert targets[0].accepts_workload

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "warning_free_percent = 20", "warning_free_percent = 5"
        ),
        encoding="utf-8",
    )
    with pytest.raises(SOCPreflightError, match="warning thresholds"):
        load_storage_targets(config)


def test_commissioning_policy_probes_data_but_not_system_root(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))
    config = (
        Path(__file__).resolve().parents[2]
        / "operations"
        / "soc_operations.example.ini"
    )

    targets = {target.name: target for target in load_storage_targets(config)}

    assert targets["data"].require_writable
    assert targets["data"].write_probe
    assert not targets["system"].require_writable
    assert not targets["system"].write_probe


def test_planned_input_requires_one_workload_target(tmp_path):
    target = _target(tmp_path, accepts_workload=False)
    with pytest.raises(SOCPreflightError, match="Exactly one"):
        check_storage((target,), planned_input_bytes=1)


def test_load_rejects_zero_work_multiplier_for_workload(tmp_path):
    config = tmp_path / "soc_operations.ini"
    config.write_text(
        "[storage:data]\n"
        f"path = {tmp_path}\n"
        "warning_free_gib = 1\ncritical_free_gib = 0\n"
        "warning_free_percent = 1\ncritical_free_percent = 0\n"
        "work_multiplier = 0\naccepts_workload = true\n",
        encoding="utf-8",
    )
    with pytest.raises(SOCPreflightError, match="at least 1"):
        load_storage_targets(config)


def test_mount_identity_and_read_only_fail_closed(tmp_path):
    usage = lambda _path: DiskUsage(2_000 * GIB, 500 * GIB, 1_500 * GIB)
    mount_info = lambda _path: MountInfo(
        source="/dev/wrong",
        fstype="xfs",
        uuid="wrong-uuid",
        options=frozenset({"rw"}),
    )
    result = check_storage_target(
        _target(
            tmp_path,
            expected_mountpoint=tmp_path,
            expected_source="/dev/nvme0n1p1",
            expected_fstype="ext4",
            expected_uuid="expected-uuid",
            require_writable=True,
        ),
        usage_provider=usage,
        mount_checker=lambda _path: True,
        mount_info_provider=mount_info,
    )
    assert result.status == "CRITICAL"
    assert any("source differs" in message for message in result.messages)
    assert any("type differs" in message for message in result.messages)
    assert any("UUID differs" in message for message in result.messages)

    read_only = check_storage_target(
        _target(
            tmp_path,
            expected_mountpoint=tmp_path,
            require_writable=True,
        ),
        usage_provider=usage,
        mount_checker=lambda _path: True,
        mount_info_provider=lambda _path: MountInfo(
            "/dev/nvme0n1p1", "ext4", "expected-uuid", frozenset({"ro"})
        ),
    )
    assert read_only.status == "CRITICAL"
    assert any("read-only" in message for message in read_only.messages)


def test_write_probe_and_inode_headroom_are_gates(tmp_path):
    usage = lambda _path: DiskUsage(2_000 * GIB, 500 * GIB, 1_500 * GIB)

    def failed_probe(_path):
        raise OSError("read-only filesystem")

    probe_failure = check_storage_target(
        _target(tmp_path, require_writable=True, write_probe=True),
        usage_provider=usage,
        write_probe_provider=failed_probe,
    )
    assert probe_failure.status == "CRITICAL"
    assert any("write probe failed" in message for message in probe_failure.messages)

    inode_failure = check_storage_target(
        _target(tmp_path, critical_free_inodes=100),
        usage_provider=usage,
        inode_provider=lambda _path: InodeUsage(total=1_000, free=50),
    )
    assert inode_failure.status == "CRITICAL"
    assert inode_failure.free_inodes == 50
