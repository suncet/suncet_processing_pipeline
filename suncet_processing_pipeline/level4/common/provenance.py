"""Small, dependency-light processing-run provenance helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


def _canonical_json(values: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Configuration must be finite and JSON serializable.") from exc
    return text.encode("utf-8")


def configuration_sha256(values: Mapping[str, Any]) -> str:
    """Hash a configuration independently of dictionary insertion order."""

    return hashlib.sha256(_canonical_json(values)).hexdigest()


def _git_value(repository: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _source_tree_sha256(repository: Path) -> str | None:
    """Hash tracked and untracked, non-ignored files without exposing contents."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    digest = hashlib.sha256()
    for encoded_path in sorted(filter(None, result.stdout.split(b"\0"))):
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        path = repository / relative
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"SYMLINK\0")
            digest.update(path.readlink().as_posix().encode("utf-8"))
            continue
        if not path.is_file():
            digest.update(b"MISSING\0")
            continue
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "astropy", "matplotlib", "suncet"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions


def collect_run_provenance(
    *,
    repository: str | Path | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture enough identity to reproduce or diagnose one algorithm run."""

    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    commit = _git_value(root, ["rev-parse", "HEAD"])
    status = _git_value(root, ["status", "--porcelain"])
    provenance: dict[str, Any] = {
        "processed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "source_tree_sha256": _source_tree_sha256(root),
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": _package_versions(),
    }
    if configuration is not None:
        provenance["configuration"] = dict(configuration)
        provenance["configuration_sha256"] = configuration_sha256(configuration)
    return provenance
