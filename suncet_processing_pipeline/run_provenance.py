"""Reproducible, atomic provenance manifests for SunCET processing runs."""

from __future__ import annotations

import configparser
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from suncet_processing_pipeline.data_paths import get_data_root


SCHEMA_VERSION = 1
MANIFEST_DIRNAME = "processing_manifests"
_SENSITIVE_TERMS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "access_key",
    "private_key",
    "credential",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _is_sensitive(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return any(term in normalized for term in _SENSITIVE_TERMS)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _is_sensitive(key) else _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sanitize_argv(argv: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for item in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if item.startswith("--") and "=" in item:
            name, _value = item.split("=", 1)
            sanitized.append(f"{name}=<redacted>" if _is_sensitive(name) else item)
            continue
        sanitized.append(item)
        if item.startswith("-") and _is_sensitive(item):
            redact_next = True
    return sanitized


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> tuple[str, bool]:
    try:
        return str(path.relative_to(root)), True
    except ValueError:
        return str(path), False


def _file_record(path: Path, root: Path, *, include_hash: bool) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    display_path, relative = _display_path(resolved, root)
    record: dict[str, object] = {
        "path": display_path,
        "relative_to_data_root": relative,
    }
    try:
        stat = resolved.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            }
        )
        if include_hash:
            record["sha256"] = sha256_file(resolved)
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _config_snapshot(config_path: Path | None) -> dict[str, object] | None:
    if config_path is None:
        return None
    resolved = config_path.expanduser().resolve()
    snapshot: dict[str, object] = {"path": str(resolved)}
    try:
        snapshot["sha256"] = sha256_file(resolved)
        parser = configparser.ConfigParser()
        with resolved.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
        snapshot["values"] = {
            section: {
                option: "<redacted>" if _is_sensitive(option) else value
                for option, value in parser.items(section)
            }
            for section in parser.sections()
        }
    except (OSError, configparser.Error, UnicodeError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def resolved_config_snapshot(config: object, data_root: Path) -> dict[str, object]:
    """Capture resolved operational values from ``config_parser.Config``."""
    names = (
        "make_level0_5",
        "make_level1",
        "make_level2",
        "make_level3",
        "make_level4",
        "ignore_realtime",
        "save_png",
        "save_jpeg2000",
        "also_save_csie_meta_json",
        "version_pipeline",
        "version_bus",
        "version_csie",
        "output_suffix",
        "base_metadata_filename",
        "ctdb_base",
        "calibration_path",
        "dark_filename",
        "flat_filename",
        "badpix_filename",
        "cosmic_ray_removal",
        "packet_definitions_path",
        "bus_ctdb_path",
        "csie_ctdb_path",
    )
    values = {name: getattr(config, name) for name in names if hasattr(config, name)}
    ctdb_base = Path(values["ctdb_base"]) if "ctdb_base" in values else None
    if ctdb_base is not None:
        values["ctdb_base"] = "$suncet_ctdb"
        for name in ("packet_definitions_path", "bus_ctdb_path", "csie_ctdb_path"):
            if name not in values:
                continue
            try:
                relative = Path(values[name]).relative_to(ctdb_base)
            except (TypeError, ValueError):
                values[name] = "<private-ctdb-path>"
            else:
                values[name] = str(Path("$suncet_ctdb") / relative)
    values["data_root"] = str(data_root)
    return _json_safe(values)  # type: ignore[return-value]


def _git_snapshot(start: Path) -> dict[str, object]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(start), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        root = Path(run_git("rev-parse", "--show-toplevel")).resolve()
        status_lines = [
            line
            for line in run_git("status", "--porcelain=v1", "--untracked-files=normal").splitlines()
            if line
        ]
        branch = run_git("branch", "--show-current")
        return {
            "available": True,
            "repository_root": str(root),
            "commit": run_git("rev-parse", "HEAD"),
            "branch": branch or None,
            "dirty": bool(status_lines),
            "status": status_lines,
        }
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _system_snapshot() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "suncet_data": str(get_data_root()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _tree_snapshot(root: Path, excluded_root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return snapshot
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(excluded_root)
            continue
        except ValueError:
            pass
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class ProcessingRunProvenance:
    """Context manager that records one processing invocation and its products."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        run_kind: str,
        config_path: str | Path | None = None,
        resolved_config: Mapping[str, object] | None = None,
        arguments: Mapping[str, object] | None = None,
        argv: Sequence[str] | None = None,
        repository_hint: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest_dir = self.data_root / MANIFEST_DIRNAME
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.manifest_path = self.manifest_dir / f"processing_run_{self.run_id}.json"
        self._started_monotonic: float | None = None
        self._before_files: dict[str, tuple[int, int]] = {}
        self._payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_kind": run_kind,
            "status": "initialized",
            "data_root": str(self.data_root),
            "invocation": {
                "working_directory": str(Path.cwd()),
                "argv": _sanitize_argv(list(argv if argv is not None else sys.argv)),
                "arguments": _json_safe(arguments or {}),
            },
            "configuration": _config_snapshot(
                Path(config_path) if config_path is not None else None
            ),
            "resolved_configuration": _json_safe(resolved_config or {}),
            "git": _git_snapshot(
                Path(repository_hint).expanduser().resolve()
                if repository_hint is not None
                else Path.cwd()
            ),
            "system": _system_snapshot(),
            "packages": _installed_packages(),
            "inputs": [],
            "outputs": {"created_or_modified": [], "deleted": []},
        }

    def __enter__(self) -> "ProcessingRunProvenance":
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._before_files = _tree_snapshot(self.data_root, self.manifest_dir)
        self._started_monotonic = time.monotonic()
        self._payload["status"] = "running"
        self._payload["started_utc"] = _utc_now()
        self._write()
        print(f"Processing provenance manifest: {self.manifest_path}")
        return self

    def record_inputs(self, paths: Iterable[str | Path]) -> None:
        unique = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
        self._payload["inputs"] = [
            _file_record(path, self.data_root, include_hash=True) for path in unique
        ]
        self._write()

    def _output_inventory(self) -> dict[str, object]:
        after = _tree_snapshot(self.data_root, self.manifest_dir)
        changed = [
            relative
            for relative, signature in after.items()
            if self._before_files.get(relative) != signature
        ]
        deleted = sorted(relative for relative in self._before_files if relative not in after)
        return {
            "created_or_modified": [
                _file_record(
                    self.data_root / relative,
                    self.data_root,
                    include_hash=True,
                )
                for relative in sorted(changed)
            ],
            "deleted": deleted,
        }

    def _write(self) -> None:
        _atomic_write_json(self.manifest_path, self._payload)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        self._payload["finished_utc"] = _utc_now()
        if self._started_monotonic is not None:
            self._payload["duration_seconds"] = round(
                time.monotonic() - self._started_monotonic, 6
            )
        self._payload["outputs"] = self._output_inventory()
        if exc_type is None:
            self._payload["status"] = "succeeded"
            self._payload["error"] = None
        else:
            self._payload["status"] = "failed"
            self._payload["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                ),
            }
        try:
            self._write()
        except Exception as manifest_error:
            if exc_type is None:
                raise
            print(
                f"WARNING: failed to finalize provenance manifest: {manifest_error}",
                file=sys.stderr,
            )
        return False
