"""Checksum inventories for private CTDB refreshes.

The manifest deliberately records paths relative to ``suncet_ctdb`` and must
remain with the private CTDB material.  It is intended to cross the host
boundary with a staged CTDB refresh so the receiving host can prove that it has
the exact file set produced by the authoritative host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

from suncet_processing_pipeline.data_paths import (
    SuncetDataPathError,
    get_ctdb_root,
    get_data_root,
)
SCHEMA_VERSION = 1
IGNORED_FILENAMES = {".DS_Store"}
IGNORED_DIRECTORY_NAMES = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


class CTDBSnapshotError(RuntimeError):
    """Raised when a CTDB tree or snapshot is unsafe or invalid."""


@dataclass(frozen=True)
class VerificationResult:
    """Differences between a CTDB tree and its expected manifest."""

    expected_tree_sha256: str
    actual_tree_sha256: str
    file_count: int
    total_bytes: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    mismatched: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.mismatched)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _relative_name(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _validate_relative_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CTDBSnapshotError(f"Invalid CTDB manifest path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise CTDBSnapshotError(f"Unsafe CTDB manifest path: {value!r}")
    return value


def _tree_digest(entries: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_ignored_artifact(root: Path, path: Path) -> bool:
    """Return whether *path* is a host cache rather than CTDB authority data."""

    relative = path.relative_to(root)
    return (
        path.name in IGNORED_FILENAMES
        or any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts)
        or path.suffix.lower() in IGNORED_SUFFIXES
    )


def _stable_file_record(root: Path, path: Path) -> dict[str, object]:
    """Hash one regular file while proving it did not change under the reader."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CTDBSnapshotError(f"Could not open CTDB file safely: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CTDBSnapshotError(
                f"Unsupported CTDB filesystem entry: {_relative_name(root, path)}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise CTDBSnapshotError(
                f"CTDB file changed while being inventoried: {_relative_name(root, path)}"
            ) from exc
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields) or any(
            getattr(after, name) != getattr(path_after, name) for name in stable_fields
        ):
            raise CTDBSnapshotError(
                f"CTDB file changed while being inventoried: {_relative_name(root, path)}"
            )
        return {
            "path": _relative_name(root, path),
            "size_bytes": after.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inventory_tree(
    root: str | os.PathLike[str],
    *,
    excluded_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, object]:
    """Return a deterministic, content-addressed inventory of one CTDB tree.

    Symbolic links are rejected.  Generated packet definitions are executable
    Python as well as data, so following a link outside the reviewed private
    root would make the snapshot ambiguous and unsafe.
    """

    resolved_root = Path(root).expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise CTDBSnapshotError(f"CTDB root is not a directory: {resolved_root}")

    excluded: set[Path] = set()
    for value in excluded_paths:
        path = Path(value).expanduser().resolve(strict=False)
        if path.is_relative_to(resolved_root):
            excluded.add(path)

    entries: list[dict[str, object]] = []
    for path in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
        resolved = path.resolve(strict=False)
        if resolved in excluded or _is_ignored_artifact(resolved_root, path):
            continue
        if path.is_symlink():
            raise CTDBSnapshotError(
                f"CTDB snapshots do not follow symbolic links: "
                f"{_relative_name(resolved_root, path)}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CTDBSnapshotError(
                f"Unsupported CTDB filesystem entry: "
                f"{_relative_name(resolved_root, path)}"
            )
        entries.append(_stable_file_record(resolved_root, path))

    return {
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "tree_sha256": _tree_digest(entries),
    }


def build_manifest(
    root: str | os.PathLike[str],
    *,
    excluded_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, object]:
    """Build a private CTDB manifest without exposing its absolute host path."""

    inventory = inventory_tree(root, excluded_paths=excluded_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        **inventory,
    }


def write_manifest(
    root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    replace: bool = False,
) -> dict[str, object]:
    """Atomically write a CTDB manifest, refusing replacement by default."""

    output_path = Path(output).expanduser().resolve(strict=False)
    try:
        public_root = get_data_root(must_exist=False)
    except SuncetDataPathError:
        public_root = None
    if public_root is not None:
        if output_path == public_root or output_path.is_relative_to(public_root):
            raise CTDBSnapshotError(
                "Private CTDB manifests must not be written below suncet_data"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    lock_acquired = False
    try:
        try:
            lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise CTDBSnapshotError(
                f"Another CTDB snapshot writer holds the lock: {lock_path}"
            ) from exc
        lock_acquired = True
        os.close(lock_descriptor)
        if output_path.exists() and not replace:
            raise CTDBSnapshotError(
                f"Refusing to replace existing CTDB manifest: {output_path}"
            )
        manifest = build_manifest(
            root, excluded_paths=(output_path, temporary, lock_path)
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, output_path)
        else:
            try:
                os.link(temporary, output_path)
            except FileExistsError as exc:
                raise CTDBSnapshotError(
                    f"Refusing to replace existing CTDB manifest: {output_path}"
                ) from exc
            temporary.unlink()
        output_path.chmod(0o600)
        _fsync_directory(output_path.parent)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


def load_manifest(path: str | os.PathLike[str]) -> dict[str, object]:
    """Load and structurally validate a CTDB checksum manifest."""

    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CTDBSnapshotError(
            f"Could not read CTDB manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CTDBSnapshotError(
            f"Unsupported CTDB manifest schema in {manifest_path}"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise CTDBSnapshotError("CTDB manifest files must be a list")
    names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise CTDBSnapshotError("CTDB manifest file entry must be an object")
        name = _validate_relative_name(entry.get("path"))
        if name in names:
            raise CTDBSnapshotError(f"Duplicate CTDB manifest path: {name}")
        names.add(name)
        size = entry.get("size_bytes")
        checksum = entry.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise CTDBSnapshotError(f"Invalid size for CTDB manifest path {name}")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise CTDBSnapshotError(f"Invalid SHA-256 for CTDB manifest path {name}")
    if payload.get("file_count") != len(files):
        raise CTDBSnapshotError("CTDB manifest file_count does not match files")
    if payload.get("total_bytes") != sum(int(entry["size_bytes"]) for entry in files):
        raise CTDBSnapshotError("CTDB manifest total_bytes does not match files")
    expected_digest = _tree_digest(files)
    if payload.get("tree_sha256") != expected_digest:
        raise CTDBSnapshotError("CTDB manifest tree checksum is invalid")
    return payload


def verify_manifest(
    root: str | os.PathLike[str], manifest_path: str | os.PathLike[str]
) -> VerificationResult:
    """Compare an exact CTDB tree with a previously generated manifest."""

    resolved_root = Path(root).expanduser().resolve(strict=True)
    resolved_manifest = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_manifest(resolved_manifest)
    actual = inventory_tree(resolved_root, excluded_paths=(resolved_manifest,))

    expected_by_name = {str(item["path"]): item for item in manifest["files"]}
    actual_by_name = {str(item["path"]): item for item in actual["files"]}
    expected_names = set(expected_by_name)
    actual_names = set(actual_by_name)
    common = expected_names & actual_names
    mismatched = tuple(
        sorted(
            name
            for name in common
            if expected_by_name[name]["size_bytes"]
            != actual_by_name[name]["size_bytes"]
            or expected_by_name[name]["sha256"] != actual_by_name[name]["sha256"]
        )
    )
    return VerificationResult(
        expected_tree_sha256=str(manifest["tree_sha256"]),
        actual_tree_sha256=str(actual["tree_sha256"]),
        file_count=int(actual["file_count"]),
        total_bytes=int(actual["total_bytes"]),
        missing=tuple(sorted(expected_names - actual_names)),
        unexpected=tuple(sorted(actual_names - expected_names)),
        mismatched=mismatched,
    )


def _print_inventory(payload: dict[str, object]) -> None:
    print(f"files: {payload['file_count']}")
    print(f"bytes: {payload['total_bytes']}")
    print(f"tree_sha256: {payload['tree_sha256']}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify a private SunCET CTDB checksum snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Write a checksum manifest")
    snapshot.add_argument("--root", type=Path, default=None)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--replace", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify a tree against a manifest")
    verify.add_argument("--root", type=Path, default=None)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    root = args.root if args.root is not None else get_ctdb_root()
    if args.command == "snapshot":
        manifest = write_manifest(root, args.output, replace=args.replace)
        print(f"Wrote private CTDB manifest: {args.output.expanduser().resolve()}")
        _print_inventory(manifest)
        return 0

    result = verify_manifest(root, args.manifest)
    print(f"files: {result.file_count}")
    print(f"bytes: {result.total_bytes}")
    print(f"expected_tree_sha256: {result.expected_tree_sha256}")
    print(f"actual_tree_sha256: {result.actual_tree_sha256}")
    for label, values in (
        ("missing", result.missing),
        ("unexpected", result.unexpected),
        ("mismatched", result.mismatched),
    ):
        print(f"{label}: {len(values)}")
        for value in values:
            print(f"  {value}")
    print("CTDB verification passed" if result.ok else "CTDB verification FAILED")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(run())
