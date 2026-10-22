"""Restartable SatNOGS DB collection with private S3 archive custody.

Run ``python -m suncet_processing_pipeline.satnogs.collect --help``. This
collector never uploads to SatNOGS, publishes products, or changes upstream data.
"""

from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .ax25 import decode_public_ax25_beacon
from .beacon_contract import BeaconValidationError
from .archive import ArchiveConfig, ArchiveError, S3Archive, drain

SAT_ID = "MNRC-9829-4319-5529-8975"
ENDPOINT = "https://db.satnogs.org/api/telemetry/"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class CollectionError(RuntimeError):
    """An actionable collection failure, without credentials or frame contents."""


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Config:
    directory: Path
    token_file: Path
    initial_start: datetime
    required_mount: Path
    overlap_days: int = 7
    reconcile_days: int = 30
    min_free_gib: float = 150
    max_pages: int = 1000
    archive: ArchiveConfig | None = None

    @classmethod
    def read(cls, path: Path) -> Config:
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(path):
            raise CollectionError("Collector configuration file is missing")
        section = parser["satnogs"]
        result = cls(
            directory=Path(section["directory"]).expanduser().resolve(),
            token_file=Path(section.get(
                "token_file", str(Path.home() / ".config/suncet/satnogs.token")
            )).expanduser().resolve(),
            initial_start=utc(section.get("initial_start", "2026-10-01T00:00:00Z")),
            required_mount=Path(section["required_mount"]).expanduser().resolve(),
            overlap_days=section.getint("overlap_days", 7),
            reconcile_days=section.getint("reconcile_days", 30),
            min_free_gib=section.getfloat("min_free_gib", 150),
            max_pages=section.getint("max_pages", 1000),
            archive=ArchiveConfig.read(parser["archive"]) if parser.has_section("archive") else None,
        )
        if min(result.overlap_days, result.reconcile_days, result.max_pages) < 1:
            raise CollectionError("Window and page limits must be positive")
        if not 0 <= result.min_free_gib < float("inf"):
            raise CollectionError("Invalid free-space reserve")
        if not result.directory.is_relative_to(result.required_mount):
            raise CollectionError("Collection directory must be below the required mount")
        return result


def check_storage(config: Config) -> None:
    if not config.required_mount.is_mount():
        raise CollectionError("Required data volume is not mounted")
    if shutil.disk_usage(config.required_mount).free < config.min_free_gib * 2**30:
        raise CollectionError("Data volume is below the configured free-space reserve")


def read_token(path: Path) -> str:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise CollectionError("Token file must be owned by this user and mode 0600 or 0400")
    token = path.read_text(encoding="ascii").strip()
    if not token or not token.isascii() or not token.isalnum():
        raise CollectionError("Token file must contain only the SatNOGS API token")
    return token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CollectionError("API redirect refused; review the endpoint before retrying")


class Client:
    """Paced GET requests; token is never sent to a pagination-provided host."""

    def __init__(self, token: str):
        self.token = token
        self.opener = build_opener(NoRedirect())
        self.last_request = None

    def get(self, url: str) -> bytes:
        parts = urlsplit(url)
        expected = urlsplit(ENDPOINT)
        if (parts.scheme, parts.netloc, parts.path) != (
            expected.scheme, expected.netloc, expected.path
        ) or parts.fragment:
            raise CollectionError("Unsafe API pagination URL refused")
        for attempt in range(3):
            if self.last_request is not None:
                time.sleep(max(0, 11 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            request = Request(url, headers={
                "Authorization": "Token " + self.token,
                "Accept": "application/json",
                "User-Agent": "SunCET-SOC-daily-collector/1",
            }, method="GET")
            try:
                with self.opener.open(request, timeout=30) as response:
                    data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise CollectionError("API response exceeds the size limit")
                return data
            except HTTPError as error:
                # Stop on throttling instead of risking an unbounded retry loop.
                # The checkpoint is untouched; the next daily/manual run recovers.
                if error.code not in (500, 502, 503, 504) or attempt == 2:
                    raise CollectionError(f"SatNOGS returned HTTP {error.code}; retry later") from None
            except (URLError, TimeoutError, OSError, IncompleteRead):
                if attempt == 2:
                    raise CollectionError("SatNOGS network request failed after three attempts") from None
            time.sleep(15 * 2**attempt)
        raise AssertionError("Unreachable")


@contextmanager
def locked_database(directory: Path):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_mode & 0o077:
        raise CollectionError("Collection directory must be private (mode 0700)")
    with (directory / "collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CollectionError("Another collection is already running") from None
        connection = sqlite3.connect(directory / "telemetry.sqlite3")
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts (
                    digest TEXT PRIMARY KEY, received_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL, frame_sha256 TEXT,
                    decoded_json TEXT, rejection TEXT, decoder_revision TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    digest TEXT PRIMARY KEY, fetched_at TEXT NOT NULL,
                    request_url TEXT NOT NULL, body BLOB NOT NULL
                );
            """)
            yield connection
        finally:
            connection.close()


def decoder_revision() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("../spacecraft_time.py", "collect.py", "ax25.py", "decoder.py", "beacon_contract.py",
                 "public_schema.py", "generated_suncet_apid1.py", "public_beacon_schema.csv"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return "ctdb-2.0.5:" + digest.hexdigest()


def save_receipt(connection, row: dict, fetched_at: str, revision: str) -> bool:
    if not isinstance(row, dict) or row.get("sat_id") != SAT_ID:
        raise CollectionError("API returned an unexpected satellite or malformed record")
    try:
        utc(row["timestamp"])
        raw = encoded(row)
    except (KeyError, ValueError, TypeError, AttributeError):
        raise CollectionError("API returned invalid record metadata") from None
    digest = hashlib.sha256(raw.encode()).hexdigest()
    frame_digest = decoded = rejection = None
    try:
        frame = bytes.fromhex(row["frame"])
        frame_digest = hashlib.sha256(frame).hexdigest()
        # SatNOGS GFSK/AX.25 produces a 268-byte frame without the FCS.
        # Never guess alternate boundaries or scan for other APIDs.
        decoded = encoded(decode_public_ax25_beacon(frame, fcs_present=False))
    except (KeyError, ValueError, TypeError, BeaconValidationError):
        rejection = "Not a valid CTDB 2.0.5 public AX.25 beacon"
    cursor = connection.execute(
        "INSERT OR IGNORE INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (digest, fetched_at, raw, frame_digest, decoded, rejection, revision),
    )
    return cursor.rowcount == 1


def collect(config: Config, client, *, now=None, since=None, full=False, archiver=None) -> dict:
    now = now or datetime.now(timezone.utc)
    check_storage(config)
    with locked_database(config.directory) as connection:
        archived = 0
        if config.archive:
            archiver = archiver or S3Archive(config.archive, config.directory)
            archived += drain(connection, config.archive, archiver)
        state = dict(connection.execute("SELECT key, value FROM state"))
        last = utc(state["last_success"]) if "last_success" in state else None
        reconciled = utc(state["last_reconcile"]) if "last_reconcile" in state else None
        full = full or reconciled is None or now - reconciled >= timedelta(days=config.reconcile_days)
        start = since or (config.initial_start if full or last is None else
                          max(config.initial_start, last - timedelta(days=config.overlap_days)))
        if start >= now:
            raise CollectionError("Download start must precede the current time")
        url = ENDPOINT + "?" + urlencode({"sat_id": SAT_ID, "start": iso(start), "end": iso(now)})
        seen = set()
        count = added = 0
        rejected_before = connection.execute("SELECT COUNT(rejection) FROM receipts").fetchone()[0]
        revision = decoder_revision()
        while url:
            if url in seen or len(seen) >= config.max_pages:
                raise CollectionError("Pagination loop or page limit reached; checkpoint unchanged")
            seen.add(url)
            check_storage(config)
            body = client.get(url)
            try:
                page = json.loads(body)
                rows, following = page["results"], page["next"]
                if not isinstance(rows, list) or (following is not None and
                                                (not isinstance(following, str) or not following)):
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise CollectionError("Malformed API page; checkpoint unchanged") from None
            fetched_at = iso(datetime.now(timezone.utc))
            with connection:
                connection.execute("INSERT OR IGNORE INTO pages VALUES (?, ?, ?, ?)", (
                    hashlib.sha256(body).hexdigest(), fetched_at, url, body,
                ))
                for row in rows:
                    added += save_receipt(connection, row, fetched_at, revision)
                    count += 1
            url = following
            if config.archive:
                archived += drain(connection, config.archive, archiver)
        # Explicit historical requests never move the daily collection checkpoint.
        if since is None:
            with connection:
                connection.execute("INSERT OR REPLACE INTO state VALUES ('last_success', ?)", (iso(now),))
                if full:
                    connection.execute("INSERT OR REPLACE INTO state VALUES ('last_reconcile', ?)", (iso(now),))
        return {"start": iso(start), "end": iso(now), "pages": len(seen),
                "records_seen": count, "new_receipts": added, "duplicates": count - added,
                "archived_receipts": archived,
                "new_rejected": connection.execute(
                    "SELECT COUNT(rejection) FROM receipts"
                ).fetchone()[0] - rejected_before}


def status(config: Config) -> dict:
    path = config.directory / "telemetry.sqlite3"
    if not path.exists():
        return {"last_success": None, "receipts": 0, "accepted": 0, "rejected": 0}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        result = dict(connection.execute("SELECT key, value FROM state"))
        result.update(zip(("receipts", "accepted", "rejected"), connection.execute(
            "SELECT COUNT(*), COUNT(decoded_json), COUNT(rejection) FROM receipts"
        ).fetchone()))
        result["archive_enabled"] = config.archive is not None
        if config.archive:
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='archived_receipts'").fetchone()
            archived = connection.execute(
                "SELECT COUNT(*) FROM archived_receipts WHERE bucket=? AND prefix=?",
                (config.archive.bucket, config.archive.prefix)).fetchone()[0] if exists else 0
            result.update(archived_receipts=archived, archive_pending=result["receipts"] - archived)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "status"))
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/suncet/satnogs.ini")
    parser.add_argument("--since", type=utc, help="One-off historical pull; does not move daily checkpoint")
    parser.add_argument("--full", action="store_true", help="Reconcile all data since initial_start")
    args = parser.parse_args(argv)
    # Raw DB frames may contain opaque/private beacon bytes: never publish this archive.
    os.umask(0o077)
    try:
        config = Config.read(args.config)
        if args.command == "status":
            result = status(config)
        else:
            client = Client(read_token(config.token_file))
            result = collect(config, client, since=args.since, full=args.full)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (CollectionError, ArchiveError, OSError, ValueError, KeyError, configparser.Error, sqlite3.Error) as error:
        # Only our deliberate diagnostics are safe for the systemd journal.
        message = str(error) if isinstance(error, (CollectionError, ArchiveError)) else type(error).__name__
        print("SatNOGS collection failed: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
