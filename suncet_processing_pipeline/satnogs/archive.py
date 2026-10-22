"""Private S3 Deep Archive custody for receipts retrieved from SatNOGS DB only."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


class ArchiveError(RuntimeError):
    """Safe diagnostic: never includes AWS stderr, credentials or frame contents."""


@dataclass(frozen=True)
class ArchiveConfig:
    bucket: str
    profile: str
    region: str
    prefix: str = "satnogs/"
    executable: str = str(Path.home() / ".local/bin/aws")

    @classmethod
    def read(cls, section):
        result = cls(section["bucket"].strip(), section["profile"].strip(),
                     section["region"].strip(), section.get("prefix", "satnogs/").strip(),
                     str(Path(section.get("aws_cli", cls.executable)).expanduser()))
        if not all((result.bucket, result.profile, result.region)):
            raise ArchiveError("Archive bucket, profile and region are required")
        if (not result.prefix.startswith("satnogs/") or not result.prefix.endswith("/")
                or any(p in (".", "..", "") for p in result.prefix[:-1].split("/"))):
            raise ArchiveError("Archive prefix must be a safe namespace below satnogs/")
        return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def initialize(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS archive_objects (
            bucket TEXT NOT NULL, object_key TEXT NOT NULL, sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, version_id TEXT NOT NULL,
            verified_at TEXT NOT NULL, PRIMARY KEY (bucket, object_key)
        );
        CREATE TABLE IF NOT EXISTS archived_receipts (
            bucket TEXT NOT NULL, prefix TEXT NOT NULL, digest TEXT NOT NULL,
            PRIMARY KEY (bucket, prefix, digest)
        );
    """)


class S3Archive:
    def __init__(self, config: ArchiveConfig, directory: Path):
        self.config, self.directory = config, directory

    def call(self, arguments, *, allow_existing=False):
        env = dict(os.environ, AWS_RETRY_MODE="standard", AWS_MAX_ATTEMPTS="3")
        try:
            response = subprocess.run(
                [self.config.executable, "s3api", *arguments, "--profile", self.config.profile,
                 "--region", self.config.region, "--output", "json", "--no-cli-pager"],
                capture_output=True, text=True, timeout=120, env=env,
            )
        except (OSError, subprocess.SubprocessError):
            raise ArchiveError("Archive AWS request could not complete") from None
        if response.returncode:
            # A failed conditional create is expected after an interrupted local commit.
            if allow_existing and "(PreconditionFailed)" in response.stderr:
                return None
            raise ArchiveError("Archive AWS request failed; check scoped permissions and connectivity")
        try:
            value = json.loads(response.stdout)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, TypeError):
            raise ArchiveError("Malformed archive AWS response") from None

    def ensure(self, key: str, body: bytes, content_type: str) -> dict:
        digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
        # Conditional PUT avoids requiring ListBucket merely to distinguish 403 from 404.
        # Never overwrite an existing key, including during crash recovery.
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix="archive-", suffix=".partial") as f:
            f.write(body)
            f.flush()
            self.call(["put-object", "--bucket", self.config.bucket, "--key", key,
                       "--body", f.name, "--if-none-match", "*", "--storage-class", "DEEP_ARCHIVE",
                       "--checksum-algorithm", "SHA256", "--checksum-sha256", digest,
                       "--content-type", content_type], allow_existing=True)
        head = self.call(["head-object", "--bucket", self.config.bucket, "--key", key,
                          "--checksum-mode", "ENABLED"])
        if (head.get("ContentLength") != len(body) or head.get("ChecksumSHA256") != digest
                or head.get("ChecksumType") != "FULL_OBJECT"
                or head.get("StorageClass") != "DEEP_ARCHIVE"
                or not head.get("VersionId") or head["VersionId"] == "null"):
            raise ArchiveError("Archive object checksum, size, storage class or version verification failed")
        return head


def drain(connection, config: ArchiveConfig, uploader) -> int:
    """Recover old pending receipts and archive each newly committed page immediately."""
    initialize(connection)
    completed = 0
    rows = connection.execute("""
        SELECT r.digest, r.received_at, r.raw_json, r.frame_sha256,
               r.decoded_json, r.rejection, r.decoder_revision
        FROM receipts r WHERE NOT EXISTS (
            SELECT 1 FROM archived_receipts a
            WHERE a.bucket=? AND a.prefix=? AND a.digest=r.digest
        ) ORDER BY r.digest
    """, (config.bucket, config.prefix))

    def store(key, body, content_type):
        sha = hashlib.sha256(body).hexdigest()
        cached = connection.execute(
            "SELECT sha256, size_bytes FROM archive_objects WHERE bucket=? AND object_key=?",
            (config.bucket, key)).fetchone()
        if cached:
            if cached != (sha, len(body)):
                raise ArchiveError("Local archive receipt conflicts with object content")
            return
        head = uploader.ensure(key, body, content_type)
        with connection:
            connection.execute("INSERT INTO archive_objects VALUES (?, ?, ?, ?, ?, ?)",
                               (config.bucket, key, sha, len(body), head["VersionId"],
                                datetime.now(timezone.utc).isoformat()))

    for digest, received, raw, frame_sha, decoded, rejection, revision in rows:
        if hashlib.sha256(raw.encode()).hexdigest() != digest:
            raise ArchiveError("Local SatNOGS receipt checksum mismatch")
        row = json.loads(raw)
        if row.get("sat_id") != "MNRC-9829-4319-5529-8975":
            raise ArchiveError("Archive receipt is not SunCET SatNOGS data")
        frame_key = None
        if frame_sha:
            frame = bytes.fromhex(row["frame"])
            if hashlib.sha256(frame).hexdigest() != frame_sha:
                raise ArchiveError("Local SatNOGS frame checksum mismatch")
            frame_key = config.prefix + "frames/" + frame_sha + ".bin"
            store(frame_key, frame, "application/octet-stream")
        manifest = {"schema": "suncet-satnogs-receipt-v1", "source": "SatNOGS DB",
                    "api_record": row, "receipt_sha256": digest, "retrieved_at": received,
                    "frame_sha256": frame_sha, "frame_key": frame_key,
                    "decoded_public": json.loads(decoded) if decoded else None,
                    "rejection": rejection, "decoder_revision": revision}
        store(config.prefix + "receipts/" + digest + ".json",
              (canonical(manifest) + "\n").encode(), "application/json")
        with connection:
            connection.execute("INSERT INTO archived_receipts VALUES (?, ?, ?)",
                               (config.bucket, config.prefix, digest))
        completed += 1
    return completed
