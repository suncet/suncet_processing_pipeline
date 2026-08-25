"""Safe, manual publication of finalized SunCET products over OpenSSH SFTP.

The publisher deliberately uses the operator's SSH configuration rather than
storing hostnames, usernames, or private-key locations in the repository. Files
are staged remotely, downloaded again for SHA-256 verification, and only then
renamed into their final locations. Existing content is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol, TypeVar

from suncet_processing_pipeline.data_paths import get_data_root


DEFAULT_TARGET = "lasp-sfs"
DEFAULT_REMOTE_ROOT = "/suncet_data"
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 3600
TRANSFER_LOG_DIRECTORY = "transfer_logs/lasp_publication"
TRANSFER_STAGING_DIRECTORY = "transfer_staging/lasp_publication"
TEMPORARY_SUFFIXES = (".download", ".partial", ".tmp")
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9_.@-]+$")
_T = TypeVar("_T")


class PublicationError(RuntimeError):
    """Raised when a product cannot be published safely."""


class PublicationConflictError(PublicationError):
    """Raised rather than overwriting different remote content."""


@dataclass(frozen=True)
class PublicationItem:
    local_path: str
    remote_path: str
    size_bytes: int
    sha256: str
    status: str


class SftpTransportProtocol(Protocol):
    def exists(self, remote_path: PurePosixPath) -> bool: ...

    def mkdirs(self, remote_directory: PurePosixPath) -> None: ...

    def put(self, local_path: Path, remote_path: PurePosixPath) -> None: ...

    def get(self, remote_path: PurePosixPath, local_path: Path) -> None: ...

    def rename(self, source: PurePosixPath, destination: PurePosixPath) -> None: ...

    def remove(self, remote_path: PurePosixPath) -> None: ...


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quote_sftp(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if "\n" in text or "\r" in text or "\x00" in text:
        raise PublicationError(f"SFTP path contains a control character: {text!r}")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _validated_remote_root(value: str) -> PurePosixPath:
    root = PurePosixPath(value)
    if not root.is_absolute() or ".." in root.parts:
        raise PublicationError(f"Remote root must be an absolute safe path: {value!r}")
    return root


def _remote_path(root: PurePosixPath, relative_path: Path) -> PurePosixPath:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PublicationError(f"Unsafe relative publication path: {relative_path}")
    remote = root.joinpath(*relative_path.parts)
    if not remote.is_relative_to(root):
        raise PublicationError(f"Remote path escapes publication root {root}: {remote}")
    return remote


class OpenSftpTransport:
    """Small OpenSSH ``sftp`` batch-mode adapter."""

    def __init__(
        self,
        target: str,
        *,
        connect_timeout_seconds: int = 10,
        command_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not _SAFE_TARGET.fullmatch(target):
            raise PublicationError(
                "SFTP target must be a simple SSH alias or user@host value"
            )
        if connect_timeout_seconds <= 0 or command_timeout_seconds <= 0:
            raise PublicationError("SFTP timeouts must be positive seconds")
        self.target = target
        self.connect_timeout_seconds = connect_timeout_seconds
        self.command_timeout_seconds = command_timeout_seconds

    def _run(self, commands: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        batch = "\n".join(commands + ["quit", ""])
        try:
            result = subprocess.run(
                [
                    "sftp",
                    "-q",
                    "-b",
                    "-",
                    "-oBatchMode=yes",
                    f"-oConnectTimeout={self.connect_timeout_seconds}",
                    self.target,
                ],
                input=batch,
                text=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise PublicationError("OpenSSH sftp is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise PublicationError(
                f"SFTP command timed out after {self.command_timeout_seconds}s"
            ) from exc
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise PublicationError(
                f"SFTP command failed with exit code {result.returncode}: {detail}"
            )
        return result

    def exists(self, remote_path: PurePosixPath) -> bool:
        result = self._run([f"ls -l {_quote_sftp(str(remote_path))}"], check=False)
        if result.returncode == 0:
            return True
        output = f"{result.stdout}\n{result.stderr}".lower()
        if "no such file" in output or "not found" in output:
            return False
        raise PublicationError(f"Could not inspect remote path {remote_path}: {output.strip()}")

    def mkdirs(self, remote_directory: PurePosixPath) -> None:
        commands = []
        current = PurePosixPath("/")
        for part in remote_directory.parts[1:]:
            current /= part
            commands.append(f"-mkdir {_quote_sftp(str(current))}")
        if commands:
            self._run(commands)

    def put(self, local_path: Path, remote_path: PurePosixPath) -> None:
        self._run(
            [f"put {_quote_sftp(local_path)} {_quote_sftp(str(remote_path))}"]
        )

    def get(self, remote_path: PurePosixPath, local_path: Path) -> None:
        self._run(
            [f"get {_quote_sftp(str(remote_path))} {_quote_sftp(local_path)}"]
        )

    def rename(self, source: PurePosixPath, destination: PurePosixPath) -> None:
        self._run(
            [f"rename {_quote_sftp(str(source))} {_quote_sftp(str(destination))}"]
        )

    def remove(self, remote_path: PurePosixPath) -> None:
        self._run([f"-rm {_quote_sftp(str(remote_path))}"])


class SftpPublisher:
    """Publish immutable files using staging and full read-back verification."""

    def __init__(
        self,
        transport: SftpTransportProtocol,
        *,
        remote_root: str = DEFAULT_REMOTE_ROOT,
        retries: int = DEFAULT_RETRIES,
        retry_delay_seconds: float = 1.0,
        verification_root: Path | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        self.transport = transport
        self.remote_root = _validated_remote_root(remote_root)
        self.retries = retries
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.verification_root = verification_root
        if self.verification_root is not None:
            self.verification_root.mkdir(parents=True, exist_ok=True)

    def _retry(self, operation: str, function: Callable[[], _T]) -> _T:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return function()
            except PublicationConflictError:
                raise
            except Exception as exc:  # transport errors are normalized here
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(self.retry_delay_seconds * (2**attempt))
        raise PublicationError(
            f"{operation} failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error

    def _exists(self, path: PurePosixPath) -> bool:
        return self._retry(f"inspect {path}", lambda: self.transport.exists(path))

    def _download_sha256(self, remote_path: PurePosixPath, directory: Path) -> str:
        destination = directory / "remote-readback"
        destination.unlink(missing_ok=True)
        self._retry(
            f"download {remote_path}",
            lambda: self.transport.get(remote_path, destination),
        )
        return sha256_file(destination)

    def _read_checksum_sidecar(
        self, remote_path: PurePosixPath, directory: Path
    ) -> str:
        destination = directory / "remote-checksum"
        destination.unlink(missing_ok=True)
        self._retry(
            f"download {remote_path}",
            lambda: self.transport.get(remote_path, destination),
        )
        token = destination.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise PublicationConflictError(
                f"Remote checksum sidecar is malformed: {remote_path}"
            )
        return token

    def _write_checksum_sidecar(
        self,
        checksum: str,
        final_path: PurePosixPath,
        checksum_path: PurePosixPath,
        directory: Path,
    ) -> None:
        local_sidecar = directory / "checksum.sha256"
        local_sidecar.write_text(
            f"{checksum}  {final_path.name}\n", encoding="utf-8"
        )
        staged_sidecar = PurePosixPath(f"{checksum_path}.partial.{checksum[:12]}")
        self._retry(
            f"upload checksum for {final_path}",
            lambda: self.transport.put(local_sidecar, staged_sidecar),
        )
        if self._download_sha256(staged_sidecar, directory) != sha256_file(local_sidecar):
            self.transport.remove(staged_sidecar)
            raise PublicationError(f"Checksum-sidecar read-back failed: {final_path}")
        try:
            self.transport.rename(staged_sidecar, checksum_path)
        except Exception as exc:
            if not self._exists(checksum_path):
                raise PublicationError(
                    f"Could not finalize checksum sidecar {checksum_path}: {exc}"
                ) from exc
            self.transport.remove(staged_sidecar)

    def publish_file(self, local_path: Path, relative_path: Path) -> PublicationItem:
        """Publish one file, refusing to replace any different remote content."""

        local_path = local_path.resolve(strict=True)
        checksum = sha256_file(local_path)
        size = local_path.stat().st_size
        final_path = _remote_path(self.remote_root, relative_path)
        checksum_path = PurePosixPath(f"{final_path}.sha256")
        staged_path = PurePosixPath(f"{final_path}.partial.{checksum[:12]}")

        with tempfile.TemporaryDirectory(
            prefix="suncet-sftp-verify-",
            dir=self.verification_root,
        ) as temporary:
            temporary_path = Path(temporary)
            final_exists = self._exists(final_path)
            checksum_exists = self._exists(checksum_path)

            if checksum_exists:
                remote_checksum = self._read_checksum_sidecar(
                    checksum_path, temporary_path
                )
                if remote_checksum != checksum:
                    raise PublicationConflictError(
                        f"Remote checksum differs; refusing to overwrite {final_path}"
                    )
                if final_exists:
                    actual_remote_checksum = self._download_sha256(
                        final_path, temporary_path
                    )
                    if actual_remote_checksum != checksum:
                        raise PublicationConflictError(
                            "Remote file does not match its checksum sidecar; "
                            f"refusing {final_path}"
                        )
                    return PublicationItem(
                        str(local_path), str(final_path), size, checksum, "unchanged"
                    )

            if final_exists:
                actual_remote_checksum = self._download_sha256(
                    final_path, temporary_path
                )
                if actual_remote_checksum != checksum:
                    raise PublicationConflictError(
                        f"Remote file differs; refusing to overwrite {final_path}"
                    )
                if not checksum_exists:
                    self._write_checksum_sidecar(
                        checksum,
                        final_path,
                        checksum_path,
                        temporary_path,
                    )
                return PublicationItem(
                    str(local_path), str(final_path), size, checksum, "adopted"
                )

            self._retry(
                f"create remote directory {final_path.parent}",
                lambda: self.transport.mkdirs(final_path.parent),
            )

            verified = False
            for attempt in range(self.retries + 1):
                self._retry(
                    f"upload staged file {final_path}",
                    lambda: self.transport.put(local_path, staged_path),
                )
                remote_checksum = self._download_sha256(staged_path, temporary_path)
                local_checksum_after_upload = sha256_file(local_path)
                if local_checksum_after_upload != checksum:
                    self.transport.remove(staged_path)
                    raise PublicationError(
                        f"Local source changed during publication: {local_path}"
                    )
                if remote_checksum == checksum:
                    verified = True
                    break
                self.transport.remove(staged_path)
                if attempt < self.retries:
                    time.sleep(self.retry_delay_seconds * (2**attempt))
            if not verified:
                raise PublicationError(
                    f"Remote read-back checksum failed for {final_path}"
                )

            try:
                self.transport.rename(staged_path, final_path)
            except Exception as exc:
                if not self._exists(final_path):
                    raise PublicationError(
                        f"Could not finalize staged file {final_path}: {exc}"
                    ) from exc
                self.transport.remove(staged_path)

            if not checksum_exists:
                self._write_checksum_sidecar(
                    checksum,
                    final_path,
                    checksum_path,
                    temporary_path,
                )

        return PublicationItem(
            str(local_path), str(final_path), size, checksum, "published"
        )


def collect_publication_files(
    sources: list[str], *, data_root: Path, local_base: Path
) -> list[tuple[Path, Path]]:
    """Resolve explicit sources and map them to safe remote-relative paths."""

    data_root = data_root.resolve(strict=True)
    local_base = local_base.resolve(strict=True)
    if not local_base.is_relative_to(data_root):
        raise PublicationError(f"Local base must be inside suncet_data: {local_base}")

    selected: dict[Path, Path] = {}
    for raw_source in sources:
        source = Path(raw_source).expanduser()
        if not source.is_absolute():
            source = data_root / source
        if source.is_symlink():
            raise PublicationError(f"Publication source may not be a symlink: {source}")
        source = source.resolve(strict=True)
        if not source.is_relative_to(data_root) or not source.is_relative_to(local_base):
            raise PublicationError(f"Publication source is outside its allowed root: {source}")

        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(local_base)
            if "transfer_logs" in relative.parts or "transfer_staging" in relative.parts:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            if candidate.name.endswith(TEMPORARY_SUFFIXES):
                continue
            selected[candidate] = relative
    if not selected:
        raise PublicationError("No finalized files were selected for publication")
    return sorted(selected.items(), key=lambda item: str(item[1]))


def _write_transfer_log(data_root: Path, payload: dict[str, object]) -> Path:
    log_root = data_root / TRANSFER_LOG_DIRECTORY
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = log_root / f"publish-{timestamp}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish finalized suncet_data files through an SSH SFTP alias."
    )
    parser.add_argument("sources", nargs="+", help="Files or directories below suncet_data")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="SSH config alias")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--local-base",
        help="Local directory mapped to remote-root (default: suncet_data)",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum seconds for one SFTP batch operation",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform publication; without this flag only print and log the plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    data_root = get_data_root()
    local_base = (
        Path(args.local_base).expanduser() if args.local_base else data_root
    )
    files = collect_publication_files(
        args.sources, data_root=data_root, local_base=local_base
    )
    remote_root = _validated_remote_root(args.remote_root)
    started = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": None,
        "mode": "execute" if args.execute else "dry_run",
        "target_alias": args.target,
        "remote_root": str(remote_root),
        "local_base": str(local_base.resolve()),
        "status": "running",
        "items": [],
        "error": None,
    }

    try:
        if not args.execute:
            planned = []
            for local_path, relative in files:
                item = PublicationItem(
                    str(local_path),
                    str(_remote_path(remote_root, relative)),
                    local_path.stat().st_size,
                    sha256_file(local_path),
                    "planned",
                )
                planned.append(item)
                print(f"PLAN {item.local_path} -> {item.remote_path} {item.sha256}")
            payload["items"] = [asdict(item) for item in planned]
        else:
            transport = OpenSftpTransport(
                args.target, command_timeout_seconds=args.command_timeout
            )
            publisher = SftpPublisher(
                transport,
                remote_root=args.remote_root,
                retries=args.retries,
                verification_root=data_root / TRANSFER_STAGING_DIRECTORY,
            )
            completed = []
            for local_path, relative in files:
                item = publisher.publish_file(local_path, relative)
                completed.append(item)
                payload["items"] = [asdict(record) for record in completed]
                print(f"{item.status.upper()} {item.remote_path} {item.sha256}")
        payload["status"] = "succeeded"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        log_path = _write_transfer_log(data_root, payload)
        print(f"Transfer log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
