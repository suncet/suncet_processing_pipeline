"""Daily read-only UHF delivery collection using the existing AWS ingest profile.

Full version listings avoid timestamp-watermark gaps. Local receipts and hashes
avoid downloading the same version again. No S3 writes or deletions are used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from .data_paths import data_path, get_data_root, SuncetDataPathError
from .ingest_s3 import (
    AwsCli, DEFAULT_CONFIG_PATH, S3IngestError, SourceConfig,
    _object_name, _prepare_private_state_directory, ingest_object, load_source_config,
)
from .run_provenance import sha256_file
from .soc_preflight import (
    DEFAULT_CONFIG_PATH as STORAGE_CONFIG, SOCPreflightError,
    check_storage, load_storage_targets,
)

DEFAULT_STATE = Path.home() / ".local/state/suncet/aws_uhf_sync"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def versions(client, source: SourceConfig, *, max_pages: int):
    """Explicitly paginate both S3 version markers; never assume first-page coverage."""
    markers = None
    seen = set()
    for _ in range(max_pages):
        arguments = ["s3api", "list-object-versions", "--bucket", source.bucket,
                     "--max-keys", "1000", "--no-paginate"]
        if source.prefix:
            arguments.extend(["--prefix", source.prefix])
        if markers:
            arguments.extend(["--key-marker", markers[0]])
            if markers[1]:
                arguments.extend(["--version-id-marker", markers[1]])
        page = client.run_json(arguments)
        rows = page.get("Versions", [])
        if not isinstance(rows, list) or not isinstance(page.get("IsTruncated"), bool):
            raise S3IngestError("Malformed object-version listing")
        for row in rows:
            if not isinstance(row, dict):
                raise S3IngestError("Malformed object-version record")
            key, version, size = row.get("Key"), row.get("VersionId"), row.get("Size")
            if (not isinstance(key, str) or not key.startswith(source.prefix)
                    or not isinstance(version, str) or not version
                    or type(size) is not int or size < 0
                    or not isinstance(row.get("ETag"), str) or not row["ETag"]
                    or not isinstance(row.get("LastModified"), str)):
                raise S3IngestError("Invalid object identity in version listing")
            if key.endswith("/") and size == 0:
                continue  # S3 folder marker, not a delivery file.
            _object_name(key)
            yield row
        if page["IsTruncated"] is False:
            return
        key_marker = page.get("NextKeyMarker")
        version_marker = page.get("NextVersionIdMarker", "")
        if not isinstance(key_marker, str) or not key_marker or not isinstance(version_marker, str):
            raise S3IngestError("Missing S3 pagination marker")
        markers = (key_marker, version_marker)
        if markers in seen:
            raise S3IngestError("Repeated S3 pagination marker")
        seen.add(markers)
    raise S3IngestError("S3 listing page budget exhausted; collection is incomplete")


def object_identity(source: SourceConfig, item: dict) -> str:
    # ETag/mtime/size distinguish replacements of the special unversioned 'null' version.
    identity = [source.bucket, item["Key"], item["VersionId"], item["ETag"],
                item["LastModified"], item["Size"]]
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


@contextmanager
def database(directory: Path):
    directory = _prepare_private_state_directory(directory, get_data_root())
    with (directory / "sync.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise S3IngestError("Another UHF S3 collection is already running") from None
        connection = sqlite3.connect(directory / "index.sqlite3")
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    identity TEXT PRIMARY KEY, receipt_json TEXT NOT NULL,
                    receipt_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            yield connection
        finally:
            connection.close()


def cached_file_is_valid(payload: str) -> bool:
    receipt = json.loads(payload)
    relative = Path(receipt["local_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise S3IngestError("Unsafe path in cached receipt")
    unresolved = get_data_root() / relative
    if unresolved.is_symlink():
        raise S3IngestError("Symlink found at a cached delivery path")
    path = data_path(*relative.parts)
    return (path.is_file() and path.stat().st_size == receipt["size_bytes"]
            and sha256_file(path) == receipt["sha256"])


def sync(client, source: SourceConfig, state: Path, preflight, *,
         max_pages=1000, max_download_bytes=50 * 2**30) -> dict:
    if max_pages < 1 or max_download_bytes < 1:
        raise S3IngestError("Collection limits must be positive")
    preflight(0)
    counts = {"versions_seen": 0, "downloaded": 0, "already_present": 0,
              "bytes_downloaded": 0}
    with database(state) as connection:
        for item in versions(client, source, max_pages=max_pages):
            counts["versions_seen"] += 1
            identity = object_identity(source, item)
            stored = connection.execute(
                "SELECT receipt_json FROM objects WHERE identity = ?", (identity,)
            ).fetchone()
            if stored and cached_file_is_valid(stored[0]):
                counts["already_present"] += 1
                continue
            if counts["bytes_downloaded"] + item["Size"] > max_download_bytes:
                raise S3IngestError("Daily download budget reached; remaining objects await next run")
            preflight(item["Size"])
            destination = source.destination / "objects" / identity
            _, receipt_path, disposition = ingest_object(
                client, replace(source, destination=destination), item["Key"],
                version_id=item["VersionId"], state_directory=state / "receipts",
                expected_etag=item["ETag"], expected_size=item["Size"],
            )
            payload = receipt_path.read_text(encoding="utf-8")
            with connection:
                connection.execute("INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                                   (identity, payload, str(receipt_path)))
            counts["bytes_downloaded"] += item["Size"]
            counts["downloaded" if disposition == "downloaded" else "already_present"] += 1
        summary = {"finished_utc": now(), **counts}
        with connection:
            connection.execute("INSERT OR REPLACE INTO state VALUES ('last_success', ?)",
                               (canonical(summary),))
    return summary


def status(state: Path) -> dict:
    path = state / "index.sqlite3"
    if not path.exists():
        return {"last_success": None, "retained_versions": 0}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT value FROM state WHERE key = 'last_success'").fetchone()
        return {"last_success": json.loads(row[0]) if row else None,
                "retained_versions": connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--storage-config", type=Path, default=STORAGE_CONFIG)
    parser.add_argument("--state-directory", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--aws-cli", default=str(Path.home() / ".local/bin/aws"))
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-download-gib", type=int, default=50)
    args = parser.parse_args(argv)
    os.umask(0o077)
    os.environ.setdefault("AWS_RETRY_MODE", "standard")
    os.environ.setdefault("AWS_MAX_ATTEMPTS", "3")
    state = args.state_directory.expanduser().resolve()
    try:
        if args.command == "status":
            result = status(state)
        else:
            source = load_source_config(args.config, "uhf")
            targets = load_storage_targets(args.storage_config)
            workload = [target for target in targets if target.accepts_workload]
            if len(workload) != 1 or workload[0].path != get_data_root():
                raise S3IngestError("Storage workload target must match suncet_data")

            def preflight(size):
                if any(result.exit_code for result in check_storage(targets, planned_input_bytes=size)):
                    raise S3IngestError("SOC storage preflight requires operator attention")

            result = sync(AwsCli(args.aws_cli, profile=source.profile, region=source.region),
                          source, state, preflight, max_pages=args.max_pages,
                          max_download_bytes=args.max_download_gib * 2**30)
        print(canonical(result))
        return 0
    except (S3IngestError, SuncetDataPathError, SOCPreflightError, OSError, ValueError,
            KeyError, sqlite3.Error) as error:
        # AWS diagnostics contain private bucket/key identifiers: keep them out of the journal.
        try:
            private = _prepare_private_state_directory(state, get_data_root())
            temporary = private / f"failure-{os.getpid()}.partial"
            temporary.write_text(canonical({"failed_utc": now(), "error": str(error)}) + "\n")
            temporary.replace(private / "last_failure.json")
        except (OSError, S3IngestError, SuncetDataPathError):
            pass
        print("UHF S3 collection failed; inspect private aws_uhf_sync/last_failure.json", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
