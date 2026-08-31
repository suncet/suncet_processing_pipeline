"""Dry-run-first, one-way public-data copies through rclone."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from suncet_processing_pipeline.data_paths import (
    SuncetDataPathError,
    get_ctdb_root,
    get_data_root,
)


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "suncet" / "rclone_public.ini"
DEFAULT_STATE_DIRECTORY = Path.home() / ".local" / "state" / "suncet" / "rclone"
REMOTE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+:.*$")
TASK_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PUBLICATION_SCHEMA_VERSION = 1
PUBLICATION_INCLUDE_PATTERN = re.compile(r"^\+ /([^*?\[\]{}\\]+)/\*\*$")
# Rclone anchors /{{REGEXP}} to the task root.  Pulls accept only this
# deliberately small versioned-basename grammar; arbitrary regexes stay invalid.
PULL_VERSIONED_FILENAME_INCLUDE_PATTERN = re.compile(
    r"^\+ /\{\{"
    r"[A-Za-z0-9][A-Za-z0-9_-]*_v"
    r"\[0-9\]\+\\\.\[0-9\]\+\\\.\[0-9\]\+"
    r"(?:dev)?"
    r"-[A-Za-z0-9][A-Za-z0-9_-]*\\\.[A-Za-z0-9][A-Za-z0-9_-]*"
    r"\}\}$"
)
SUPPORTED_PUBLICATION_EXCLUSIONS = {
    "- **/*.partial",
    "- **/*.partial.*",
    "- **/*.tmp",
    "- **/.*/**",
    "- /**",
}


class RclonePublicError(RuntimeError):
    """Raised when a public-data copy is unsafe or fails verification."""


@dataclass(frozen=True)
class RcloneTask:
    name: str
    direction: str
    local_path: Path
    remote_path: str
    filter_file: Path
    rclone_config: Path
    executable: str
    transfers: int
    checkers: int
    state_directory: Path
    timeout_seconds: int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_path(raw_value: str, *, field: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_value.strip()))
    if "$" in expanded:
        raise RclonePublicError(f"{field} contains an undefined environment variable")
    path = Path(expanded)
    if not path.is_absolute():
        raise RclonePublicError(f"{field} must be absolute: {raw_value!r}")
    return path.resolve(strict=False)


def _private_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise RclonePublicError(f"{label} is not a file: {path}")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RclonePublicError(f"{label} must not be accessible by group/other: {path}")


def _reject_below(path: Path, root: Path, *, label: str) -> None:
    resolved = path.resolve(strict=False)
    if resolved == root or resolved.is_relative_to(root):
        raise RclonePublicError(f"{label} must remain outside suncet_data: {resolved}")


def _reject_ctdb_path(path: Path, ctdb_root: Path | None, *, label: str) -> None:
    if ctdb_root is None:
        return
    resolved = path.resolve(strict=False)
    if resolved == ctdb_root or resolved.is_relative_to(ctdb_root):
        raise RclonePublicError(f"{label} must remain outside suncet_ctdb: {resolved}")


def _validate_private_state_path(
    path: Path, data_root: Path, ctdb_root: Path | None
) -> None:
    """Reject state paths that contain or are contained by managed data roots."""

    resolved = path.resolve(strict=False)
    for protected_root, name in (
        (data_root, "suncet_data"),
        (ctdb_root, "suncet_ctdb"),
    ):
        if protected_root is None:
            continue
        protected = protected_root.resolve(strict=False)
        if (
            resolved == protected
            or resolved.is_relative_to(protected)
            or protected.is_relative_to(resolved)
        ):
            raise RclonePublicError(
                "rclone private state directory must not overlap "
                f"{name}: {resolved}"
            )


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise RclonePublicError(f"rclone state path is not a directory: {path}")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RclonePublicError(
            f"rclone state path must not be accessible by group/other: {path}"
        )


def _write_private_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_private_json(
    path: Path, payload: dict[str, object], *, replace: bool = False
) -> None:
    if path.exists() and not replace:
        raise RclonePublicError(f"Refusing to replace existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise RclonePublicError(f"Refusing to replace existing file: {path}") from exc
            temporary.unlink()
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _literal_filter_roots(
    filter_file: Path,
    *,
    label: str,
    allow_versioned_filename_regexes: bool = False,
) -> tuple[Path, ...]:
    """Extract literal destination roots from a deliberately simple allowlist."""

    roots: list[Path] = []
    active_lines = [
        raw_line.strip()
        for raw_line in filter_file.read_text(encoding="utf-8").splitlines()
        if raw_line.strip() and not raw_line.strip().startswith("#")
    ]
    if not active_lines or active_lines[-1] != "- /**":
        raise RclonePublicError(
            f"{label} filter must end with the catch-all '- /**' rule"
        )
    for line in active_lines:
        if line.startswith("-"):
            if line not in SUPPORTED_PUBLICATION_EXCLUSIONS:
                raise RclonePublicError(
                    f"{label} filters do not support this exclusion rule; use a "
                    f"narrower literal include root instead: {line!r}"
                )
            continue
        literal_match = PUBLICATION_INCLUDE_PATTERN.fullmatch(line)
        filename_match = None
        if literal_match is None and allow_versioned_filename_regexes:
            filename_match = PULL_VERSIONED_FILENAME_INCLUDE_PATTERN.fullmatch(line)
        if literal_match is None and filename_match is None:
            supported_includes = "literal '/PATH/**' rules"
            if allow_versioned_filename_regexes:
                supported_includes += (
                    " or rooted versioned-filename regular expressions"
                )
            raise RclonePublicError(
                f"{label} filter includes must be {supported_includes}: {line!r}"
            )
        if literal_match is None:
            root = Path(".")
        else:
            root = Path(literal_match.group(1))
            if root.is_absolute() or not root.parts or ".." in root.parts:
                raise RclonePublicError(f"unsafe publication root in filter: {line!r}")
        roots.append(root)
    if not roots:
        raise RclonePublicError(f"{label} filter contains no allowlist roots")
    unique = tuple(sorted(set(roots), key=lambda item: item.as_posix()))
    for index, root in enumerate(unique):
        if any(root.is_relative_to(other) for other in unique[:index]):
            raise RclonePublicError(f"nested publication roots are not allowed: {root}")
    return unique


def _publication_roots(filter_file: Path) -> tuple[Path, ...]:
    """Extract the deliberately simple top-level push allowlist."""

    return _literal_filter_roots(filter_file, label="push manifest")


def _validate_pull_destination_tree(local_root: Path, filter_file: Path) -> None:
    """Reject symlinks that could redirect an allowed pull outside its root."""

    for relative_root in _literal_filter_roots(
        filter_file,
        label="pull safety",
        allow_versioned_filename_regexes=True,
    ):
        destination_root = local_root / relative_root
        if destination_root.is_symlink():
            raise RclonePublicError(
                f"pull destination contains a symlink: {destination_root}"
            )
        if destination_root.exists() and not destination_root.is_dir():
            raise RclonePublicError(
                f"pull destination root is not a directory: {destination_root}"
            )
        if not destination_root.exists():
            continue
        for path in destination_root.rglob("*"):
            if path.is_symlink():
                raise RclonePublicError(
                    f"pull destination contains a symlink: {path}"
                )


def _publication_tree_digest(entries: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_publication_record(root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    if any(character in relative for character in "\r\n\x00"):
        raise RclonePublicError(f"publication path has unsupported characters: {relative!r}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RclonePublicError(f"could not open publication file safely: {relative}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RclonePublicError(f"publication entry is not a regular file: {relative}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise RclonePublicError(
                f"publication file changed while hashing: {relative}"
            ) from exc
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in fields) or any(
            getattr(after, name) != getattr(path_after, name) for name in fields
        ):
            raise RclonePublicError(f"publication file changed while hashing: {relative}")
        return {
            "path": relative,
            "size_bytes": after.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _inventory_publication(task: RcloneTask) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for relative_root in _publication_roots(task.filter_file):
        root = task.local_path / relative_root
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise RclonePublicError(f"publication root is not a real directory: {root}")
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(task.local_path)
            if path.is_symlink():
                raise RclonePublicError(f"publication tree contains a symlink: {relative}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise RclonePublicError(
                    f"publication tree contains a special filesystem entry: {relative}"
                )
            if (
                any(part.startswith(".") for part in relative.parts)
                or path.name.endswith((".partial", ".tmp"))
                or ".partial." in path.name
            ):
                raise RclonePublicError(
                    f"publication tree contains an incomplete/private artifact: {relative}"
                )
            entries.append(_stable_publication_record(task.local_path, path))
    if not entries:
        raise RclonePublicError("publication allowlist contains no files to freeze")
    return entries


def create_publication_manifest(
    task: RcloneTask,
    output: str | os.PathLike[str],
    *,
    replace: bool = False,
    data_root: Path | None = None,
) -> dict[str, object]:
    """Freeze the exact public files approved for one push operation."""

    if task.direction != "push":
        raise RclonePublicError("publication manifests apply only to push tasks")
    root = (data_root or get_data_root()).resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    _reject_below(output_path, root, label="publication manifest")
    try:
        ctdb_root = get_ctdb_root(must_exist=False).resolve(strict=False)
    except SuncetDataPathError:
        ctdb_root = None
    _reject_ctdb_path(output_path, ctdb_root, label="publication manifest")
    entries = _inventory_publication(task)
    payload: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task.name,
        "filter_file_sha256": _sha256(task.filter_file),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": _publication_tree_digest(entries),
    }
    _write_private_json(output_path, payload, replace=replace)
    return payload


def _load_publication_manifest(
    path: Path, task: RcloneTask, root: Path
) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    _reject_below(resolved, root, label="publication manifest")
    try:
        ctdb_root = get_ctdb_root(must_exist=False).resolve(strict=False)
    except SuncetDataPathError:
        ctdb_root = None
    _reject_ctdb_path(resolved, ctdb_root, label="publication manifest")
    _private_file(resolved, label="publication manifest")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RclonePublicError(f"Could not read publication manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PUBLICATION_SCHEMA_VERSION:
        raise RclonePublicError("unsupported publication manifest schema")
    if payload.get("task") != task.name:
        raise RclonePublicError("publication manifest task does not match requested task")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RclonePublicError("publication manifest files must be a nonempty list")
    names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise RclonePublicError("publication manifest entry must be an object")
        name = entry.get("path")
        relative = Path(name) if isinstance(name, str) else Path("..")
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or any(character in name for character in "\r\n\x00")
        ):
            raise RclonePublicError(f"unsafe publication manifest path: {name!r}")
        if name in names:
            raise RclonePublicError(f"duplicate publication manifest path: {name}")
        names.add(name)
        size = entry.get("size_bytes")
        checksum = entry.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise RclonePublicError(f"invalid publication size for {name}")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise RclonePublicError(f"invalid publication SHA-256 for {name}")
    if payload.get("file_count") != len(files):
        raise RclonePublicError("publication manifest file_count is invalid")
    if payload.get("total_bytes") != sum(int(entry["size_bytes"]) for entry in files):
        raise RclonePublicError("publication manifest total_bytes is invalid")
    if payload.get("tree_sha256") != _publication_tree_digest(files):
        raise RclonePublicError("publication manifest tree checksum is invalid")
    if payload.get("filter_file_sha256") != _sha256(task.filter_file):
        raise RclonePublicError("push filter changed after publication approval")
    return payload


def _verify_publication_manifest(payload: dict[str, object], task: RcloneTask) -> None:
    actual = _inventory_publication(task)
    if actual != payload["files"]:
        raise RclonePublicError(
            "publication files differ from the frozen manifest; create and review a new manifest"
        )


def _safe_local_path(data_root: Path, value: str) -> Path:
    relative = Path(value.strip() or ".")
    if relative.is_absolute() or ".." in relative.parts:
        raise RclonePublicError(
            f"task local path must remain below suncet_data: {value!r}"
        )
    resolved = (data_root / relative).resolve(strict=False)
    if not resolved.is_relative_to(data_root):
        raise RclonePublicError(
            f"task local path escapes suncet_data: {value!r}"
        )
    return resolved


def load_task(
    config_path: str | os.PathLike[str],
    task_name: str,
    *,
    data_root: Path | None = None,
) -> RcloneTask:
    """Load one reviewed copy task from a private host-local INI file."""

    if not TASK_PATTERN.fullmatch(task_name):
        raise RclonePublicError(
            "task name must contain only letters, numbers, underscores, and hyphens"
        )
    root = (data_root or get_data_root()).resolve(strict=True)
    resolved_config = Path(config_path).expanduser().resolve(strict=False)
    _reject_below(resolved_config, root, label="rclone public task config")
    _private_file(resolved_config, label="rclone public task config")
    parser = configparser.ConfigParser()
    try:
        with resolved_config.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise RclonePublicError(
            f"Could not read rclone public config {resolved_config}: {exc}"
        ) from exc
    section = f"task:{task_name}"
    if not parser.has_section("rclone") or not parser.has_section(section):
        raise RclonePublicError(
            f"Config must contain [rclone] and [{section}] sections"
        )

    direction = parser.get(section, "direction", fallback="").strip().lower()
    if direction not in {"pull", "push"}:
        raise RclonePublicError(f"{section}.direction must be pull or push")
    remote = parser.get(section, "remote", fallback="").strip()
    if not REMOTE_PATTERN.fullmatch(remote) or any(
        character in remote for character in "\r\n\x00"
    ):
        raise RclonePublicError(f"{section}.remote is not an rclone remote path")

    try:
        ctdb_root = get_ctdb_root(must_exist=False).resolve(strict=False)
    except SuncetDataPathError:
        ctdb_root = None
    if ctdb_root is not None and (
        ctdb_root == root
        or ctdb_root.is_relative_to(root)
        or root.is_relative_to(ctdb_root)
    ):
        raise RclonePublicError(
            "suncet_data and private suncet_ctdb roots must not overlap"
        )
    local = _safe_local_path(root, parser.get(section, "local", fallback="."))
    filter_file = _expand_path(
        parser.get(section, "filter_file", fallback=""),
        field=f"{section}.filter_file",
    )
    if not filter_file.is_file():
        raise RclonePublicError(f"rclone filter file does not exist: {filter_file}")
    _reject_below(filter_file, root, label="rclone filter file")
    rclone_config = _expand_path(
        parser.get("rclone", "config", fallback=""), field="rclone.config"
    )
    _private_file(rclone_config, label="rclone credential config")
    _reject_below(rclone_config, root, label="rclone credential config")
    state_directory = _expand_path(
        parser.get(
            "rclone", "state_directory", fallback=str(DEFAULT_STATE_DIRECTORY)
        ),
        field="rclone.state_directory",
    )
    _reject_below(state_directory, root, label="rclone private state directory")
    _validate_private_state_path(state_directory, root, ctdb_root)
    for private_path, label in (
        (resolved_config, "rclone public task config"),
        (filter_file, "rclone filter file"),
        (rclone_config, "rclone credential config"),
        (state_directory, "rclone private state directory"),
    ):
        _reject_ctdb_path(private_path, ctdb_root, label=label)

    executable = parser.get("rclone", "executable", fallback="").strip()
    if not executable:
        executable = shutil.which("rclone") or "rclone"
    try:
        transfers = parser.getint("rclone", "transfers", fallback=4)
        checkers = parser.getint("rclone", "checkers", fallback=8)
        timeout_seconds = parser.getint("rclone", "timeout_seconds", fallback=21600)
    except ValueError as exc:
        raise RclonePublicError("rclone transfers/checkers must be integers") from exc
    if transfers <= 0 or checkers <= 0 or timeout_seconds <= 0:
        raise RclonePublicError("rclone transfers/checkers/timeout must be positive")
    return RcloneTask(
        name=task_name,
        direction=direction,
        local_path=local,
        remote_path=remote,
        filter_file=filter_file,
        rclone_config=rclone_config,
        executable=executable,
        transfers=transfers,
        checkers=checkers,
        state_directory=state_directory,
        timeout_seconds=timeout_seconds,
    )


def copy_command(
    task: RcloneTask,
    *,
    execute: bool,
    log_path: Path,
    filter_path: Path | None = None,
    files_from_path: Path | None = None,
) -> list[str]:
    source, destination = (
        (task.remote_path, str(task.local_path))
        if task.direction == "pull"
        else (str(task.local_path), task.remote_path)
    )
    command = [
        task.executable,
        "copy",
        source,
        destination,
        "--config",
        str(task.rclone_config),
        "--immutable",
        "--check-first",
        "--transfers",
        str(task.transfers),
        "--checkers",
        str(task.checkers),
        "--stats",
        "30s",
        "--log-level",
        "INFO",
        "--log-file",
        str(log_path),
    ]
    if files_from_path is not None:
        command.extend(["--files-from-raw", str(files_from_path)])
    else:
        command.extend(["--filter-from", str(filter_path or task.filter_file)])
    if not execute:
        command.append("--dry-run")
    return command


def check_command(
    task: RcloneTask,
    *,
    log_path: Path,
    filter_path: Path | None = None,
    files_from_path: Path | None = None,
) -> list[str]:
    source, destination = (
        (task.remote_path, str(task.local_path))
        if task.direction == "pull"
        else (str(task.local_path), task.remote_path)
    )
    command = [
        task.executable,
        "check",
        source,
        destination,
        "--one-way",
        "--config",
        str(task.rclone_config),
        "--checkers",
        str(task.checkers),
        "--log-level",
        "INFO",
        "--log-file",
        str(log_path),
    ]
    if files_from_path is not None:
        command.extend(["--files-from-raw", str(files_from_path)])
    else:
        command.extend(["--filter-from", str(filter_path or task.filter_file)])
    return command


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    _write_private_json(path, payload)


def run_task(
    task: RcloneTask,
    *,
    execute: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    data_root: Path | None = None,
    publication_manifest: Path | None = None,
) -> tuple[Path, Path]:
    """Run a private, receipted copy from an immutable invocation snapshot."""

    root = (data_root or get_data_root()).resolve(strict=True)
    if not (
        task.local_path == root or task.local_path.is_relative_to(root)
    ):
        raise RclonePublicError("task local path escaped suncet_data before execution")
    try:
        ctdb_root = get_ctdb_root(must_exist=False).resolve(strict=False)
    except SuncetDataPathError:
        ctdb_root = None
    _validate_private_state_path(task.state_directory, root, ctdb_root)
    _private_directory(task.state_directory)
    stamp = _utc_stamp()
    mode = "execute" if execute else "dry_run"
    run_directory = task.state_directory / f"{stamp}_{task.name}_{mode}"
    run_directory.mkdir(mode=0o700)
    log_path = run_directory / "rclone.log"
    receipt_path = run_directory / "receipt.json"
    filter_snapshot = run_directory / "filter.rules"
    filter_content = task.filter_file.read_bytes()
    filter_digest = hashlib.sha256(filter_content).hexdigest()
    _write_private_bytes(filter_snapshot, filter_content)

    manifest_payload: dict[str, object] | None = None
    manifest_digest: str | None = None
    files_from_path: Path | None = None
    preflight_stage = (
        "publication_preflight"
        if task.direction == "push"
        else "pull_destination_preflight"
    )
    try:
        if task.direction == "pull":
            _validate_pull_destination_tree(task.local_path, filter_snapshot)
        else:
            if publication_manifest is None:
                raise RclonePublicError(
                    "push tasks require a reviewed --manifest frozen with --create-manifest"
                )
            manifest_payload = _load_publication_manifest(
                publication_manifest, task, root
            )
            manifest_digest = _sha256(publication_manifest)
            _verify_publication_manifest(manifest_payload, task)
            files_from_path = run_directory / "publication-files.txt"
            files_content = "".join(
                f"{entry['path']}\n" for entry in manifest_payload["files"]
            ).encode("utf-8")
            _write_private_bytes(files_from_path, files_content)
    except BaseException as exc:
        _write_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "outcome": "failed",
                "stage": preflight_stage,
                "task": task.name,
                "direction": task.direction,
                "mode": mode,
                "filter_file_sha256": filter_digest,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RclonePublicError(
            f"{preflight_stage.replace('_', ' ')} failed; see {receipt_path}"
        ) from exc

    command = copy_command(
        task,
        execute=execute,
        log_path=log_path,
        filter_path=filter_snapshot,
        files_from_path=files_from_path,
    )
    operation_lock = task.state_directory / ".operation.lock"
    try:
        lock_descriptor = os.open(
            operation_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        _write_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "outcome": "failed",
                "stage": "operation_lock",
                "task": task.name,
                "direction": task.direction,
                "mode": mode,
                "error": {
                    "type": type(exc).__name__,
                    "message": f"another rclone public-data operation holds {operation_lock}",
                },
            },
        )
        raise RclonePublicError(
            f"another rclone public-data operation is active; see {receipt_path}"
        ) from exc
    else:
        os.close(lock_descriptor)
    verification_command: Sequence[str] | None = None
    copy_returncode: int | None = None
    verification_returncode: int | None = None
    filter_unchanged = False
    publication_unchanged: bool | None = None
    manifest_file_unchanged: bool | None = None
    error: BaseException | None = None
    try:
        copy_result = runner(
            command, check=False, text=True, timeout=task.timeout_seconds
        )
        copy_returncode = copy_result.returncode
        if copy_returncode == 0 and execute:
            verification_command = check_command(
                task,
                log_path=log_path,
                filter_path=filter_snapshot,
                files_from_path=files_from_path,
            )
            check_result = runner(
                verification_command,
                check=False,
                text=True,
                timeout=task.timeout_seconds,
            )
            verification_returncode = check_result.returncode
    except BaseException as exc:
        error = exc
    finally:
        try:
            filter_unchanged = _sha256(task.filter_file) == filter_digest
            if manifest_payload is not None:
                manifest_file_unchanged = (
                    publication_manifest is not None
                    and publication_manifest.is_file()
                    and _sha256(publication_manifest) == manifest_digest
                )
                try:
                    _verify_publication_manifest(manifest_payload, task)
                except RclonePublicError:
                    publication_unchanged = False
                else:
                    publication_unchanged = True
        except BaseException as validation_error:
            if error is None:
                error = validation_error
        policy_failures = []
        if not filter_unchanged:
            policy_failures.append("filter_changed")
        if publication_unchanged is False:
            policy_failures.append("publication_files_changed")
        if manifest_file_unchanged is False:
            policy_failures.append("publication_manifest_changed")
        outcome = "succeeded"
        if (
            error is not None
            or policy_failures
            or copy_returncode not in {0}
            or (execute and verification_returncode not in {0})
        ):
            outcome = "failed"
        payload: dict[str, object] = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "task": task.name,
            "direction": task.direction,
            "mode": mode,
            "local_path": str(task.local_path.relative_to(root) or "."),
            "remote_path": task.remote_path,
            "filter_file_sha256": filter_digest,
            "filter_unchanged": filter_unchanged,
            "publication_manifest_sha256": (
                manifest_digest
            ),
            "publication_manifest_unchanged": manifest_file_unchanged,
            "publication_tree_sha256": (
                manifest_payload.get("tree_sha256")
                if manifest_payload is not None
                else None
            ),
            "publication_unchanged": publication_unchanged,
            "policy_failures": policy_failures,
            "copy_command": list(command),
            "copy_returncode": copy_returncode,
            "verification_command": (
                list(verification_command) if verification_command is not None else None
            ),
            "verification_returncode": verification_returncode,
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        }
        try:
            _write_receipt(receipt_path, payload)
        finally:
            operation_lock.unlink(missing_ok=True)

    if error is not None:
        if isinstance(error, KeyboardInterrupt):
            raise error
        raise RclonePublicError(
            f"rclone operation failed before completion; see {receipt_path}"
        ) from error
    if not filter_unchanged:
        raise RclonePublicError(
            f"rclone filter changed during the operation; see {receipt_path}"
        )
    if publication_unchanged is False:
        raise RclonePublicError(
            f"publication files changed during the operation; see {receipt_path}"
        )
    if manifest_file_unchanged is False:
        raise RclonePublicError(
            f"publication manifest changed during the operation; see {receipt_path}"
        )
    if copy_returncode:
        raise RclonePublicError(
            f"rclone copy failed with exit code {copy_returncode}; see {log_path}"
        )
    if execute and verification_returncode:
        raise RclonePublicError(
            "rclone post-copy verification failed with exit code "
            f"{verification_returncode}; see {log_path}"
        )
    return log_path, receipt_path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or execute a reviewed one-way public-data rclone task."
    )
    parser.add_argument("task")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--create-manifest",
        type=Path,
        metavar="PATH",
        help="Freeze the current push allowlist into a private reviewed manifest.",
    )
    action.add_argument(
        "--manifest",
        type=Path,
        help="Reviewed private publication manifest required by a push task.",
    )
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help="Allow --create-manifest to replace an existing private manifest.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Transfer and then verify files; omission is a dry run.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    task = load_task(args.config, args.task)
    if args.create_manifest is not None:
        if args.execute:
            raise RclonePublicError("--create-manifest cannot be combined with --execute")
        payload = create_publication_manifest(
            task, args.create_manifest, replace=args.replace_manifest
        )
        print(f"Wrote private publication manifest: {args.create_manifest.expanduser().resolve()}")
        print(f"Files: {payload['file_count']}")
        print(f"Bytes: {payload['total_bytes']}")
        print(f"Tree SHA-256: {payload['tree_sha256']}")
        print("Review this manifest before using it with --manifest.")
        return 0
    if args.replace_manifest:
        raise RclonePublicError("--replace-manifest requires --create-manifest")
    if task.direction == "pull" and args.manifest is not None:
        raise RclonePublicError("--manifest applies only to push tasks")
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: {task.direction} task {task.name}")
    print(f"  local:  {task.local_path}")
    print(f"  remote: {task.remote_path}")
    print(f"  filter: {task.filter_file}")
    log_path, receipt_path = run_task(
        task,
        execute=args.execute,
        publication_manifest=args.manifest,
    )
    print(f"Log: {log_path}")
    print(f"Receipt: {receipt_path}")
    if not args.execute:
        print("No files were transferred. Review the log, then repeat with --execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
