"""Manual, auditable ingest from the SunCET AWS delivery buckets.

AWS resource names remain in a host-local INI file. Downloads are written to a
temporary file, hashed, and atomically finalized below ``suncet_data``. The
source object is never modified or deleted.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence

from suncet_processing_pipeline.data_paths import (
    SuncetDataPathError,
    data_path,
    get_ctdb_root,
    get_data_root,
)
from suncet_processing_pipeline.run_provenance import sha256_file


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "suncet" / "aws_ingest.ini"
DEFAULT_STATE_DIRECTORY = Path.home() / ".local" / "state" / "suncet" / "aws_ingest"


class S3IngestError(RuntimeError):
    """Raised when an S3 delivery object cannot be ingested safely."""


class S3IngestConflictError(S3IngestError):
    """Raised rather than overwriting different local content."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    bucket: str
    prefix: str
    destination: Path
    profile: str
    region: str


class AwsCli:
    """Small JSON adapter around the installed AWS CLI."""

    def __init__(self, executable: str, *, profile: str, region: str) -> None:
        self.executable = executable
        self.profile = profile
        self.region = region

    def run_json(self, arguments: Sequence[str]) -> dict[str, object]:
        command = [
            self.executable,
            *arguments,
            "--profile",
            self.profile,
            "--region",
            self.region,
            "--output",
            "json",
            "--no-cli-pager",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise S3IngestError(f"Could not run AWS CLI: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise S3IngestError(
                f"AWS CLI failed with exit code {result.returncode}: {detail}"
            )
        if not result.stdout.strip():
            return {}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise S3IngestError("AWS CLI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise S3IngestError("AWS CLI returned an unexpected JSON value")
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _safe_relative_directory(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise S3IngestError(
            f"Ingest destination must be a safe path below suncet_data: {value!r}"
        )
    return path


def load_source_config(config_path: Path, source_name: str) -> SourceConfig:
    """Load one delivery source without embedding resource names in the repo."""

    resolved = config_path.expanduser()
    parser = configparser.ConfigParser()
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise S3IngestError(f"Could not read ingest config {resolved}: {exc}") from exc

    if not parser.has_section("aws") or not parser.has_section(source_name):
        raise S3IngestError(
            f"Config {resolved} must contain [aws] and [{source_name}] sections"
        )
    profile = parser.get("aws", "profile", fallback="").strip()
    region = parser.get("aws", "region", fallback="").strip()
    bucket = parser.get(source_name, "bucket", fallback="").strip()
    prefix = parser.get(source_name, "prefix", fallback="").strip().lstrip("/")
    destination = _safe_relative_directory(
        parser.get(source_name, "destination", fallback="").strip()
    )
    if not profile or not region or not bucket:
        raise S3IngestError(
            f"Config {resolved} has an empty profile, region, or bucket value"
        )
    return SourceConfig(
        name=source_name,
        bucket=bucket,
        prefix=prefix,
        destination=destination,
        profile=profile,
        region=region,
    )


def list_objects(client: AwsCli, source: SourceConfig) -> list[dict[str, object]]:
    """Return current objects under the configured delivery prefix."""

    arguments = ["s3api", "list-objects-v2", "--bucket", source.bucket]
    if source.prefix:
        arguments.extend(["--prefix", source.prefix])
    payload = client.run_json(arguments)
    contents = payload.get("Contents", [])
    if not isinstance(contents, list):
        raise S3IngestError("S3 object listing had an unexpected Contents value")
    return [item for item in contents if isinstance(item, dict)]


def _object_name(key: str) -> str:
    if not key or key.endswith("/") or "\x00" in key:
        raise S3IngestError(f"S3 key does not name a file: {key!r}")
    name = PurePosixPath(key).name
    if name in {"", ".", ".."}:
        raise S3IngestError(f"S3 key has no safe filename: {key!r}")
    return name


def _prepare_private_state_directory(state_directory: Path, data_root: Path) -> Path:
    """Validate and create a dedicated private state directory."""

    resolved = state_directory.expanduser().resolve(strict=False)
    try:
        ctdb_root = get_ctdb_root(must_exist=False)
    except SuncetDataPathError:
        ctdb_root = None
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
            raise S3IngestError(
                f"AWS ingest private state must not overlap {name}: {resolved}"
            )
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not resolved.is_dir():
        raise S3IngestError(f"AWS ingest private state is not a directory: {resolved}")
    if os.name == "posix" and stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise S3IngestError(
            "AWS ingest private state must not be accessible by group/other: "
            f"{resolved}"
        )
    return resolved


def _write_receipt(state_directory: Path, payload: dict[str, object]) -> Path:
    log_directory = state_directory.expanduser().resolve(strict=False)
    if not log_directory.is_dir():
        raise S3IngestError(
            f"AWS ingest private state is not a directory: {log_directory}"
        )
    identity = "\0".join(
        str(payload.get(name, ""))
        for name in ("source", "bucket", "key", "version_id", "sha256")
    )
    identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = log_directory / f"{timestamp}_{payload['source']}_{identity_hash}.json"
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ingest_object(
    client: AwsCli,
    source: SourceConfig,
    key: str,
    *,
    version_id: str | None = None,
    state_directory: Path | None = None,
) -> tuple[Path, Path, str]:
    """Download, hash, atomically finalize, and receipt one delivery object."""

    if source.prefix and not key.startswith(source.prefix):
        raise S3IngestError(
            f"Key {key!r} is outside configured prefix {source.prefix!r}"
        )
    data_root = get_data_root()
    private_state = _prepare_private_state_directory(
        state_directory or DEFAULT_STATE_DIRECTORY,
        data_root,
    )
    destination_directory = data_path(*source.destination.parts)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / _object_name(key)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")

    arguments = [
        "s3api",
        "get-object",
        "--bucket",
        source.bucket,
        "--key",
        key,
        "--checksum-mode",
        "ENABLED",
    ]
    if version_id:
        arguments.extend(["--version-id", version_id])
    arguments.append(str(temporary))

    started = _utc_now()
    try:
        metadata = client.run_json(arguments)
        if not temporary.is_file():
            raise S3IngestError("AWS CLI reported success without a downloaded file")
        actual_size = temporary.stat().st_size
        declared_size = metadata.get("ContentLength")
        if isinstance(declared_size, int) and actual_size != declared_size:
            raise S3IngestError(
                f"Downloaded size {actual_size} differs from S3 size {declared_size}"
            )
        digest = sha256_file(temporary)
        status = "downloaded"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or sha256_file(destination) != digest
            ):
                raise S3IngestConflictError(
                    f"Refusing to overwrite different local content: {destination}"
                )
            status = "already_present"
        else:
            temporary.unlink()

        receipt_payload: dict[str, object] = {
            "schema_version": 1,
            "started_utc": started,
            "finished_utc": _utc_now(),
            "status": status,
            "source": source.name,
            "bucket": source.bucket,
            "key": key,
            "version_id": metadata.get("VersionId") or version_id,
            "etag": metadata.get("ETag"),
            "last_modified": metadata.get("LastModified"),
            "size_bytes": actual_size,
            "sha256": digest,
            "local_path": str(destination.relative_to(data_root)),
            "s3_checksums": {
                name: metadata[name]
                for name in (
                    "ChecksumCRC32",
                    "ChecksumCRC32C",
                    "ChecksumCRC64NVME",
                    "ChecksumSHA1",
                    "ChecksumSHA256",
                )
                if name in metadata
            },
        }
        receipt = _write_receipt(private_state, receipt_payload)
        return destination, receipt, status
    finally:
        temporary.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Host-local AWS ingest INI file",
    )
    parser.add_argument(
        "--aws-cli",
        default="aws",
        help="AWS CLI executable (default: aws on PATH)",
    )
    parser.add_argument(
        "--state-directory",
        type=Path,
        default=DEFAULT_STATE_DIRECTORY,
        help="Private receipt directory outside suncet_data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available delivery objects")
    list_parser.add_argument("source", help="Config section, such as xband or uhf")

    pull_parser = subparsers.add_parser("pull", help="Pull one exact delivery object")
    pull_parser.add_argument("source", help="Config section, such as xband or uhf")
    pull_parser.add_argument("key", help="Exact S3 object key")
    pull_parser.add_argument("--version-id")
    pull_parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the download; otherwise print the planned operation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    source = load_source_config(args.config, args.source)
    client = AwsCli(args.aws_cli, profile=source.profile, region=source.region)

    if args.command == "list":
        for item in list_objects(client, source):
            print(
                "\t".join(
                    str(item.get(name, ""))
                    for name in ("LastModified", "Size", "Key")
                )
            )
        return 0

    if not args.execute:
        version = f" version {args.version_id}" if args.version_id else ""
        print(
            f"PLAN s3://{source.bucket}/{args.key}{version} -> "
            f"{source.destination / _object_name(args.key)}"
        )
        return 0
    destination, receipt, status = ingest_object(
        client,
        source,
        args.key,
        version_id=args.version_id,
        state_directory=args.state_directory,
    )
    print(f"{status.upper()} {destination}")
    print(f"Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
