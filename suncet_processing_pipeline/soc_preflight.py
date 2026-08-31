"""Storage and mount guardrails for manual SunCET SOC operations."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NamedTuple

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "suncet" / "soc_operations.ini"
GIB = 1024**3


class SOCPreflightError(RuntimeError):
    """Raised when the host-local operations policy is invalid."""


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


class InodeUsage(NamedTuple):
    total: int
    free: int


class MountInfo(NamedTuple):
    source: str
    fstype: str
    uuid: str | None
    options: frozenset[str]


@dataclass(frozen=True)
class StorageTarget:
    name: str
    path: Path
    expected_mountpoint: Path | None
    warning_free_bytes: int
    critical_free_bytes: int
    warning_free_percent: float
    critical_free_percent: float
    work_multiplier: float
    accepts_workload: bool
    expected_source: str | None = None
    expected_fstype: str | None = None
    expected_uuid: str | None = None
    require_writable: bool = False
    write_probe: bool = False
    warning_free_inodes: int = 0
    critical_free_inodes: int = 0
    warning_free_inode_percent: float = 0.0
    critical_free_inode_percent: float = 0.0


@dataclass(frozen=True)
class StorageResult:
    name: str
    path: str
    status: str
    messages: tuple[str, ...]
    total_bytes: int | None
    free_bytes: int | None
    free_percent: float | None
    planned_input_bytes: int
    reserved_work_bytes: int
    projected_free_bytes: int | None
    total_inodes: int | None = None
    free_inodes: int | None = None
    free_inode_percent: float | None = None

    @property
    def exit_code(self) -> int:
        return {"OK": 0, "WARNING": 1, "CRITICAL": 2}[self.status]


def _absolute_expanded_path(raw_value: str, *, field: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_value.strip()))
    if "$" in expanded:
        raise SOCPreflightError(f"{field} contains an undefined environment variable")
    path = Path(expanded)
    if not path.is_absolute():
        raise SOCPreflightError(f"{field} must be an absolute path: {raw_value!r}")
    return path.resolve(strict=False)


def _nonnegative_float(
    parser: configparser.ConfigParser, section: str, option: str
) -> float:
    try:
        value = parser.getfloat(section, option)
    except (configparser.Error, ValueError) as exc:
        raise SOCPreflightError(f"Invalid {section}.{option}") from exc
    if not math.isfinite(value) or value < 0:
        raise SOCPreflightError(f"{section}.{option} must be nonnegative")
    return value


def _nonnegative_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: int = 0,
) -> int:
    try:
        value = parser.getint(section, option, fallback=fallback)
    except (configparser.Error, ValueError) as exc:
        raise SOCPreflightError(f"Invalid {section}.{option}") from exc
    if value < 0:
        raise SOCPreflightError(f"{section}.{option} must be nonnegative")
    return value


def _inode_usage(path: Path) -> InodeUsage:
    stats = os.statvfs(path)
    return InodeUsage(total=stats.f_files, free=stats.f_favail)


def _mount_info(path: Path) -> MountInfo:
    """Read the mounted source identity using util-linux ``findmnt``."""

    try:
        result = subprocess.run(
            [
                "findmnt",
                "--json",
                "--target",
                str(path),
                "--output",
                "SOURCE,FSTYPE,OPTIONS,UUID",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SOCPreflightError(f"Could not inspect mount identity for {path}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SOCPreflightError(f"findmnt failed for {path}: {detail}")
    try:
        payload = json.loads(result.stdout)
        filesystems = payload["filesystems"]
        item = filesystems[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SOCPreflightError(f"findmnt returned invalid mount data for {path}") from exc
    source = str(item.get("source") or "")
    fstype = str(item.get("fstype") or "")
    uuid_value = item.get("uuid")
    raw_options = str(item.get("options") or "")
    if not source or not fstype:
        raise SOCPreflightError(f"findmnt omitted source or filesystem type for {path}")
    return MountInfo(
        source=source,
        fstype=fstype,
        uuid=str(uuid_value) if uuid_value else None,
        options=frozenset(value for value in raw_options.split(",") if value),
    )


def _write_probe(path: Path) -> None:
    """Create, flush, and remove a private sentinel on a workload filesystem."""

    probe = path / f".suncet-write-probe-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(descriptor, b"SunCET SOC preflight\n")
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        probe.unlink(missing_ok=True)


def _sources_match(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if actual.startswith("/") and expected.startswith("/"):
        try:
            return Path(actual).resolve(strict=True) == Path(expected).resolve(strict=True)
        except OSError:
            return False
    return False


def load_storage_targets(path: str | os.PathLike[str]) -> tuple[StorageTarget, ...]:
    """Load explicit host-local storage policy sections.

    Sections are named ``[storage:NAME]``.  Keeping the thresholds outside the
    repository lets operators tune them after measuring real pass expansion
    without silently changing pipeline behavior.
    """

    config_path = Path(path).expanduser()
    parser = configparser.ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise SOCPreflightError(
            f"Could not read SOC operations config {config_path}: {exc}"
        ) from exc

    targets: list[StorageTarget] = []
    for section in parser.sections():
        if not section.startswith("storage:"):
            continue
        name = section.partition(":")[2].strip()
        if not name:
            raise SOCPreflightError("Storage policy section has an empty name")
        raw_path = parser.get(section, "path", fallback="").strip()
        if not raw_path:
            raise SOCPreflightError(f"{section}.path is required")
        target_path = _absolute_expanded_path(raw_path, field=f"{section}.path")
        raw_mountpoint = parser.get(
            section, "expected_mountpoint", fallback=""
        ).strip()
        expected_mountpoint = (
            _absolute_expanded_path(
                raw_mountpoint, field=f"{section}.expected_mountpoint"
            )
            if raw_mountpoint
            else None
        )
        warning_gib = _nonnegative_float(parser, section, "warning_free_gib")
        critical_gib = _nonnegative_float(parser, section, "critical_free_gib")
        warning_percent = _nonnegative_float(
            parser, section, "warning_free_percent"
        )
        critical_percent = _nonnegative_float(
            parser, section, "critical_free_percent"
        )
        multiplier = _nonnegative_float(parser, section, "work_multiplier")
        warning_inodes = _nonnegative_int(
            parser, section, "warning_free_inodes", fallback=0
        )
        critical_inodes = _nonnegative_int(
            parser, section, "critical_free_inodes", fallback=0
        )
        warning_inode_percent = _nonnegative_float(
            parser, section, "warning_free_inode_percent"
        ) if parser.has_option(section, "warning_free_inode_percent") else 0.0
        critical_inode_percent = _nonnegative_float(
            parser, section, "critical_free_inode_percent"
        ) if parser.has_option(section, "critical_free_inode_percent") else 0.0
        if warning_percent > 100 or critical_percent > 100:
            raise SOCPreflightError(f"{section} percentages must not exceed 100")
        if warning_inode_percent > 100 or critical_inode_percent > 100:
            raise SOCPreflightError(
                f"{section} inode percentages must not exceed 100"
            )
        if warning_gib < critical_gib or warning_percent < critical_percent:
            raise SOCPreflightError(
                f"{section} warning thresholds must be at least critical thresholds"
            )
        if (
            warning_inodes < critical_inodes
            or warning_inode_percent < critical_inode_percent
        ):
            raise SOCPreflightError(
                f"{section} inode warning thresholds must be at least critical thresholds"
            )
        try:
            accepts_workload = parser.getboolean(
                section, "accepts_workload", fallback=False
            )
            require_writable = parser.getboolean(
                section, "require_writable", fallback=accepts_workload
            )
            write_probe = parser.getboolean(
                section, "write_probe", fallback=accepts_workload
            )
        except ValueError as exc:
            raise SOCPreflightError(f"Invalid boolean in {section}") from exc
        if accepts_workload and multiplier < 1:
            raise SOCPreflightError(
                f"{section}.work_multiplier must be at least 1 for a workload target"
            )
        targets.append(
            StorageTarget(
                name=name,
                path=target_path,
                expected_mountpoint=expected_mountpoint,
                warning_free_bytes=math.ceil(warning_gib * GIB),
                critical_free_bytes=math.ceil(critical_gib * GIB),
                warning_free_percent=warning_percent,
                critical_free_percent=critical_percent,
                work_multiplier=multiplier,
                accepts_workload=accepts_workload,
                expected_source=parser.get(
                    section, "expected_source", fallback=""
                ).strip() or None,
                expected_fstype=parser.get(
                    section, "expected_fstype", fallback=""
                ).strip().lower() or None,
                expected_uuid=parser.get(
                    section, "expected_uuid", fallback=""
                ).strip().lower() or None,
                require_writable=require_writable,
                write_probe=write_probe,
                warning_free_inodes=warning_inodes,
                critical_free_inodes=critical_inodes,
                warning_free_inode_percent=warning_inode_percent,
                critical_free_inode_percent=critical_inode_percent,
            )
        )
    if not targets:
        raise SOCPreflightError(
            f"No [storage:NAME] sections were found in {config_path}"
        )
    if sum(target.accepts_workload for target in targets) > 1:
        raise SOCPreflightError("Only one storage target may accept planned workload")
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise SOCPreflightError("Storage target names must be unique")
    return tuple(targets)


def check_storage_target(
    target: StorageTarget,
    *,
    planned_input_bytes: int = 0,
    usage_provider: Callable[[Path], DiskUsage] = shutil.disk_usage,
    mount_checker: Callable[[Path], bool] = os.path.ismount,
    mount_info_provider: Callable[[Path], MountInfo] = _mount_info,
    inode_provider: Callable[[Path], InodeUsage] = _inode_usage,
    write_probe_provider: Callable[[Path], None] = _write_probe,
) -> StorageResult:
    """Evaluate one target and return a machine-readable backpressure result."""

    if planned_input_bytes < 0:
        raise SOCPreflightError("planned_input_bytes must be nonnegative")
    messages: list[str] = []
    if not target.path.is_dir():
        return StorageResult(
            name=target.name,
            path=str(target.path),
            status="CRITICAL",
            messages=("storage path is not an existing directory",),
            total_bytes=None,
            free_bytes=None,
            free_percent=None,
            planned_input_bytes=planned_input_bytes,
            reserved_work_bytes=0,
            projected_free_bytes=None,
        )
    if target.expected_mountpoint is not None:
        mountpoint = target.expected_mountpoint
        if not (
            target.path == mountpoint or target.path.is_relative_to(mountpoint)
        ):
            messages.append(
                f"storage path is not below expected mountpoint: {mountpoint}"
            )
        elif not mountpoint.is_dir() or not mount_checker(mountpoint):
            messages.append(f"expected mountpoint is not mounted: {mountpoint}")
        else:
            try:
                if target.path.stat().st_dev != mountpoint.stat().st_dev:
                    messages.append(
                        f"storage path is not on expected mountpoint: {mountpoint}"
                    )
            except OSError as exc:
                messages.append(f"could not verify expected mountpoint: {exc}")
            if (
                target.expected_source
                or target.expected_fstype
                or target.expected_uuid
                or target.require_writable
            ):
                try:
                    mount_info = mount_info_provider(mountpoint)
                except (OSError, SOCPreflightError) as exc:
                    messages.append(f"could not verify mount identity: {exc}")
                else:
                    if target.expected_source and not _sources_match(
                        mount_info.source, target.expected_source
                    ):
                        messages.append(
                            "mounted source differs from policy: "
                            f"expected {target.expected_source}, got {mount_info.source}"
                        )
                    if (
                        target.expected_fstype
                        and mount_info.fstype.lower() != target.expected_fstype
                    ):
                        messages.append(
                            "filesystem type differs from policy: "
                            f"expected {target.expected_fstype}, got {mount_info.fstype}"
                        )
                    if (
                        target.expected_uuid
                        and (mount_info.uuid or "").lower() != target.expected_uuid
                    ):
                        messages.append(
                            "filesystem UUID differs from policy: "
                            f"expected {target.expected_uuid}, got {mount_info.uuid or 'unknown'}"
                        )
                    if target.require_writable and "ro" in mount_info.options:
                        messages.append("filesystem is mounted read-only")

    if target.require_writable and not os.access(target.path, os.W_OK):
        messages.append("storage path is not writable by the current operator")
    if target.write_probe and not messages:
        try:
            write_probe_provider(target.path)
        except OSError as exc:
            messages.append(f"write probe failed: {exc}")

    try:
        usage = usage_provider(target.path)
    except OSError as exc:
        messages.append(f"could not read filesystem usage: {exc}")
        usage = None
    if usage is None or usage.total <= 0:
        return StorageResult(
            name=target.name,
            path=str(target.path),
            status="CRITICAL",
            messages=tuple(messages or ("filesystem reported no capacity",)),
            total_bytes=None,
            free_bytes=None,
            free_percent=None,
            planned_input_bytes=planned_input_bytes,
            reserved_work_bytes=0,
            projected_free_bytes=None,
        )

    try:
        inode_usage = inode_provider(target.path)
    except OSError as exc:
        messages.append(f"could not read inode usage: {exc}")
        inode_usage = None
    if inode_usage is not None and inode_usage.total > 0:
        free_inode_percent = inode_usage.free / inode_usage.total * 100.0
        inode_critical = (
            inode_usage.free < target.critical_free_inodes
            or free_inode_percent < target.critical_free_inode_percent
        )
        inode_warning = (
            inode_usage.free < target.warning_free_inodes
            or free_inode_percent < target.warning_free_inode_percent
        )
    else:
        free_inode_percent = None
        inode_critical = bool(
            target.critical_free_inodes or target.critical_free_inode_percent
        )
        inode_warning = bool(
            target.warning_free_inodes or target.warning_free_inode_percent
        )
        if inode_warning:
            messages.append("filesystem reported no usable inode capacity")

    workload = planned_input_bytes if target.accepts_workload else 0
    reserved_work_bytes = math.ceil(workload * target.work_multiplier)
    projected_free = usage.free - reserved_work_bytes
    free_percent = usage.free / usage.total * 100.0
    projected_percent = projected_free / usage.total * 100.0
    critical = (
        projected_free < target.critical_free_bytes
        or projected_percent < target.critical_free_percent
        or inode_critical
    )
    warning = (
        projected_free < target.warning_free_bytes
        or projected_percent < target.warning_free_percent
        or inode_warning
    )
    if messages or critical:
        status = "CRITICAL"
    elif warning:
        status = "WARNING"
    else:
        status = "OK"

    if reserved_work_bytes:
        messages.append(
            f"reserved {reserved_work_bytes} bytes for planned input expansion"
        )
    if critical:
        messages.append("projected capacity violates the critical stop threshold")
    elif warning:
        messages.append("projected capacity is below the warning threshold")
    else:
        messages.append("free space satisfies configured thresholds")

    return StorageResult(
        name=target.name,
        path=str(target.path),
        status=status,
        messages=tuple(messages),
        total_bytes=usage.total,
        free_bytes=usage.free,
        free_percent=free_percent,
        planned_input_bytes=workload,
        reserved_work_bytes=reserved_work_bytes,
        projected_free_bytes=projected_free,
        total_inodes=inode_usage.total if inode_usage is not None else None,
        free_inodes=inode_usage.free if inode_usage is not None else None,
        free_inode_percent=free_inode_percent,
    )


def check_storage(
    targets: tuple[StorageTarget, ...], *, planned_input_bytes: int = 0
) -> tuple[StorageResult, ...]:
    """Check all configured filesystems before accepting a manual workload."""

    if planned_input_bytes > 0:
        workload_targets = sum(target.accepts_workload for target in targets)
        if workload_targets != 1:
            raise SOCPreflightError(
                "Exactly one storage target must accept planned workload when "
                "--planned-input-bytes is nonzero"
            )
    return tuple(
        check_storage_target(target, planned_input_bytes=planned_input_bytes)
        for target in targets
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check SunCET SOC mounts, disk thresholds, and backpressure."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--planned-input-bytes",
        type=int,
        default=0,
        help="Known compressed input bytes; the data policy applies its expansion factor.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    targets = load_storage_targets(args.config)
    results = check_storage(targets, planned_input_bytes=args.planned_input_bytes)
    if args.as_json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            free = "unknown" if result.free_bytes is None else str(result.free_bytes)
            projected = (
                "unknown"
                if result.projected_free_bytes is None
                else str(result.projected_free_bytes)
            )
            print(
                f"{result.status} {result.name}: free={free} "
                f"projected_free={projected} free_inodes="
                f"{result.free_inodes if result.free_inodes is not None else 'unknown'} "
                f"path={result.path}"
            )
            for message in result.messages:
                print(f"  {message}")
    return max(result.exit_code for result in results)


if __name__ == "__main__":
    raise SystemExit(run())
