import json
import re
import subprocess
from pathlib import Path

import pytest

from suncet_processing_pipeline.rclone_public_data import (
    RclonePublicError,
    check_command,
    copy_command,
    create_publication_manifest,
    load_task,
    run_task,
)


PULL_METADATA_FILTER = (
    Path(__file__).resolve().parents[2] / "operations" / "rclone" / "pull_metadata.filter"
)


def _task_config(
    tmp_path,
    data_root,
    *,
    local=".",
    direction="pull",
    filter_path=None,
    remote="public:suncet-data",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rclone_config = tmp_path / "rclone.conf"
    rclone_config.write_text("[public]\ntype = dropbox\n", encoding="utf-8")
    rclone_config.chmod(0o600)
    if filter_path is None:
        filter_file = tmp_path / "public.filter"
        filter_file.write_text(
            "+ /metadata/**\n- /**\n"
            if direction == "pull"
            else "+ /level2/**\n- /**\n",
            encoding="utf-8",
        )
    else:
        filter_file = Path(filter_path)
    task_name = "pull-metadata" if direction == "pull" else "push-products"
    state_directory = tmp_path / "state"
    config = tmp_path / "rclone_public.ini"
    config.write_text(
        "[rclone]\n"
        f"config = {rclone_config}\n"
        "executable = /usr/bin/rclone\n"
        "transfers = 2\n"
        "checkers = 3\n"
        "timeout_seconds = 60\n"
        f"state_directory = {state_directory}\n\n"
        f"[task:{task_name}]\n"
        f"direction = {direction}\n"
        f"remote = {remote}\n"
        f"local = {local}\n"
        f"filter_file = {filter_file}\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def test_load_task_is_host_local_and_copy_is_dry_run_by_default(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(tmp_path, data_root), "pull-metadata", data_root=data_root
    )
    log = tmp_path / "copy.log"

    command = copy_command(task, execute=False, log_path=log)
    assert command[:4] == [
        "/usr/bin/rclone",
        "copy",
        "public:suncet-data",
        str(data_root),
    ]
    assert "--immutable" in command
    assert "--checksum" in command
    assert "--check-first" in command
    assert "--dry-run" in command
    assert check_command(task, log_path=log)[1] == "check"


def test_load_task_rejects_local_escape_and_public_credentials(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _task_config(tmp_path, data_root, local="../ctdb")
    with pytest.raises(RclonePublicError, match="below suncet_data"):
        load_task(config, "pull-metadata", data_root=data_root)

    config = _task_config(tmp_path / "second", data_root)
    config.chmod(0o644)
    with pytest.raises(RclonePublicError, match="group/other"):
        load_task(config, "pull-metadata", data_root=data_root)

    config.chmod(0o600)
    with pytest.raises(RclonePublicError, match="task name"):
        load_task(config, "../../escape", data_root=data_root)

    credential_inside_data = data_root / "level2" / "rclone.conf"
    credential_inside_data.parent.mkdir()
    credential_inside_data.write_text("[public]\ntype = dropbox\n", encoding="utf-8")
    credential_inside_data.chmod(0o600)
    config = _task_config(tmp_path / "third", data_root)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            str(tmp_path / "third" / "rclone.conf"), str(credential_inside_data)
        ),
        encoding="utf-8",
    )
    with pytest.raises(RclonePublicError, match="outside suncet_data"):
        load_task(config, "pull-metadata", data_root=data_root)


def test_load_task_rejects_all_control_files_and_state_below_public_data(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _task_config(tmp_path / "private", data_root)

    public_config = data_root / "level2" / "task.ini"
    public_config.parent.mkdir()
    public_config.write_bytes(config.read_bytes())
    public_config.chmod(0o600)
    with pytest.raises(RclonePublicError, match="outside suncet_data"):
        load_task(public_config, "pull-metadata", data_root=data_root)

    public_filter = data_root / "level2" / "public.filter"
    public_filter.write_text("+ /metadata/**\n- /**\n", encoding="utf-8")
    filter_config = _task_config(tmp_path / "filter-case", data_root)
    filter_config.write_text(
        filter_config.read_text(encoding="utf-8").replace(
            str(tmp_path / "filter-case" / "public.filter"), str(public_filter)
        ),
        encoding="utf-8",
    )
    with pytest.raises(RclonePublicError, match="outside suncet_data"):
        load_task(filter_config, "pull-metadata", data_root=data_root)

    state_config = _task_config(tmp_path / "state-case", data_root)
    state_config.write_text(
        state_config.read_text(encoding="utf-8").replace(
            str(tmp_path / "state-case" / "state"), str(data_root / "ops")
        ),
        encoding="utf-8",
    )
    with pytest.raises(RclonePublicError, match="outside suncet_data"):
        load_task(state_config, "pull-metadata", data_root=data_root)


def test_load_task_rejects_state_directory_that_contains_public_data(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _task_config(tmp_path, data_root)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            str(tmp_path / "state"), str(tmp_path)
        ),
        encoding="utf-8",
    )
    tmp_path.chmod(0o755)
    original_mode = tmp_path.stat().st_mode & 0o777

    with pytest.raises(RclonePublicError, match="must not overlap suncet_data"):
        load_task(config, "pull-metadata", data_root=data_root)

    assert tmp_path.stat().st_mode & 0o777 == original_mode


def test_run_task_previews_once_and_executes_then_checks(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(tmp_path, data_root), "pull-metadata", data_root=data_root
    )
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    _, dry_receipt = run_task(
        task, execute=False, runner=runner, data_root=data_root
    )
    assert len(calls) == 1
    assert "--dry-run" in calls[0]
    filter_argument = calls[0][calls[0].index("--filter-from") + 1]
    assert filter_argument != str(task.filter_file)
    assert Path(filter_argument).is_relative_to(task.state_directory)
    assert json.loads(dry_receipt.read_text(encoding="utf-8"))["mode"] == "dry_run"

    calls.clear()
    _, execute_receipt = run_task(
        task, execute=True, runner=runner, data_root=data_root
    )
    assert [command[1] for command in calls] == ["copy", "check"]
    payload = json.loads(execute_receipt.read_text(encoding="utf-8"))
    assert payload["copy_returncode"] == 0
    assert payload["verification_returncode"] == 0
    assert payload["filter_file_sha256"]
    assert execute_receipt.is_relative_to(task.state_directory)
    assert not (data_root / "transfer_logs").exists()


def test_pull_accepts_actual_metadata_filename_filter(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(
            tmp_path / "config",
            data_root,
            local="metadata",
            filter_path=PULL_METADATA_FILTER,
            remote="public:suncet-data/metadata",
        ),
        "pull-metadata",
        data_root=data_root,
    )
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    run_task(task, runner=runner, data_root=data_root)

    assert task.local_path == data_root / "metadata"
    assert task.remote_path == "public:suncet-data/metadata"
    assert len(calls) == 1
    assert "--dry-run" in calls[0]


def test_actual_metadata_filter_matches_only_versioned_csv_families():
    include_prefix = "+ /{{"
    patterns = [
        re.compile(line.removeprefix(include_prefix).removesuffix("}}"))
        for line in PULL_METADATA_FILTER.read_text(encoding="utf-8").splitlines()
        if line.startswith(include_prefix)
    ]

    assert len(patterns) == 4
    for filename in (
        "suncet_metadata_definition_v1.0.2-FITS.csv",
        "suncet_metadata_definition_v1.0.2dev-FITS.csv",
        "suncet_metadata_definition_v12.34.56-NetCDF-Zarr.csv",
        "suncet_metadata_definition_v12.34.56dev-NetCDF-Zarr.csv",
    ):
        assert any(pattern.fullmatch(filename) for pattern in patterns)
    for filename in (
        "suncet_metadata_definition_v1-FITS.csv",
        "suncet_metadata_definition_v1.0.x-FITS.csv",
        "suncet_metadata_definition_v1.0.2rc1-FITS.csv",
        "other_metadata_definition_v1.0.2-FITS.csv",
        "suncet_metadata_definition_v1.0.2-FITS.csv.partial",
        "nested/suncet_metadata_definition_v1.0.2-FITS.csv",
    ):
        assert not any(pattern.fullmatch(filename) for pattern in patterns)


def test_actual_metadata_filter_checks_literal_parent_for_symlinks(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    private_root = tmp_path / "ctdb"
    private_root.mkdir()
    monkeypatch.setenv("suncet_ctdb", str(private_root))
    task = load_task(
        _task_config(
            tmp_path / "config",
            data_root,
            local="metadata",
            filter_path=PULL_METADATA_FILTER,
            remote="public:suncet-data/metadata",
        ),
        "pull-metadata",
        data_root=data_root,
    )
    (data_root / "metadata").symlink_to(private_root, target_is_directory=True)
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RclonePublicError, match="pull destination preflight failed"):
        run_task(task, runner=runner, data_root=data_root)

    assert calls == []
    receipt = next(task.state_directory.rglob("receipt.json"))
    assert json.loads(receipt.read_text(encoding="utf-8"))["stage"] == (
        "pull_destination_preflight"
    )


def test_pull_rejects_broad_rooted_filename_regex(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _task_config(
        tmp_path / "config",
        data_root,
        local="metadata",
        remote="public:suncet-data/metadata",
    )
    filter_file = tmp_path / "config" / "public.filter"
    filter_file.write_text("+ /{{.*\\.csv}}\n- /**\n", encoding="utf-8")
    task = load_task(config, "pull-metadata", data_root=data_root)
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RclonePublicError, match="pull destination preflight failed"):
        run_task(task, runner=runner, data_root=data_root)

    assert calls == []
    receipt = next(task.state_directory.rglob("receipt.json"))
    error = json.loads(receipt.read_text(encoding="utf-8"))["error"]
    assert "rooted versioned-filename regular expressions" in error["message"]


def test_filter_is_snapshotted_and_mutation_fails_with_receipt(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(tmp_path, data_root), "pull-metadata", data_root=data_root
    )

    def runner(command, **_kwargs):
        task.filter_file.write_text("+ /different/**\n- /**\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RclonePublicError, match="filter changed"):
        run_task(task, runner=runner, data_root=data_root)
    receipts = list(task.state_directory.rglob("receipt.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["filter_unchanged"] is False


def test_push_requires_frozen_manifest_and_detects_source_change(tmp_path):
    data_root = tmp_path / "data"
    product = data_root / "level2" / "product.fits"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"final product")
    task = load_task(
        _task_config(tmp_path, data_root, direction="push"),
        "push-products",
        data_root=data_root,
    )
    manifest = tmp_path / "publication.json"
    create_publication_manifest(task, manifest, data_root=data_root)
    assert manifest.stat().st_mode & 0o077 == 0

    with pytest.raises(RclonePublicError, match="publication preflight failed"):
        run_task(task, data_root=data_root, runner=lambda *_args, **_kwargs: None)

    calls = []

    def stable_runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    run_task(
        task,
        execute=True,
        runner=stable_runner,
        data_root=data_root,
        publication_manifest=manifest,
    )
    assert [command[1] for command in calls] == ["copy", "check"]
    assert all("--files-from-raw" in command for command in calls)
    assert "--checksum" not in calls[0]

    def mutating_runner(command, **_kwargs):
        product.write_bytes(b"changed during upload")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RclonePublicError, match="publication files changed"):
        run_task(
            task,
            runner=mutating_runner,
            data_root=data_root,
            publication_manifest=manifest,
        )


def test_push_manifest_rejects_unmodeled_filter_exclusions(tmp_path):
    data_root = tmp_path / "data"
    product = data_root / "level2" / "product.fits"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"final product")
    task = load_task(
        _task_config(tmp_path, data_root, direction="push"),
        "push-products",
        data_root=data_root,
    )
    task.filter_file.write_text(
        "- **/private/**\n+ /level2/**\n- /**\n", encoding="utf-8"
    )
    with pytest.raises(RclonePublicError, match="do not support"):
        create_publication_manifest(task, tmp_path / "publication.json", data_root=data_root)


def test_push_manifest_rejects_actual_metadata_filename_filter(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(
            tmp_path / "config",
            data_root,
            direction="push",
            local="metadata",
            filter_path=PULL_METADATA_FILTER,
        ),
        "push-products",
        data_root=data_root,
    )

    with pytest.raises(RclonePublicError, match="literal"):
        create_publication_manifest(
            task,
            tmp_path / "publication.json",
            data_root=data_root,
        )


def test_runner_failure_is_receipted(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(tmp_path, data_root), "pull-metadata", data_root=data_root
    )

    def runner(_command, **_kwargs):
        raise OSError("rclone executable disappeared")

    with pytest.raises(RclonePublicError, match="before completion"):
        run_task(task, runner=runner, data_root=data_root)
    receipt = next(task.state_directory.rglob("receipt.json"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["error"]["type"] == "OSError"
    assert payload["outcome"] == "failed"


def test_concurrent_operation_lock_fails_closed_and_is_receipted(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    task = load_task(
        _task_config(tmp_path, data_root), "pull-metadata", data_root=data_root
    )
    task.state_directory.mkdir(mode=0o700)
    (task.state_directory / ".operation.lock").write_text("held\n", encoding="utf-8")

    with pytest.raises(RclonePublicError, match="operation is active"):
        run_task(
            task,
            runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
            data_root=data_root,
        )
    receipt = next(task.state_directory.rglob("receipt.json"))
    assert json.loads(receipt.read_text(encoding="utf-8"))["stage"] == "operation_lock"
