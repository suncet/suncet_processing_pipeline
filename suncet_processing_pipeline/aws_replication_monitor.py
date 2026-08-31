"""Read-only health checks for S3 delivery-object replication."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from suncet_processing_pipeline.ingest_s3 import (
    DEFAULT_CONFIG_PATH,
    AwsCli,
    S3IngestError,
    SourceConfig,
    load_source_config,
)


HEALTHY_STATUSES = {"COMPLETE", "COMPLETED"}


@dataclass(frozen=True)
class ReplicationFinding:
    source: str
    key: str
    version_id: str | None
    is_latest: bool
    last_modified: str
    size_bytes: int | None
    status: str
    age_hours: float | None
    severity: str
    message: str

    @property
    def exit_code(self) -> int:
        return {"OK": 0, "WARNING": 1, "CRITICAL": 2}[self.severity]


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(value: object, now: datetime) -> float | None:
    timestamp = _parse_timestamp(value)
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds() / 3600.0)


def _classify(status: str, age_hours: float | None, pending_hours: float) -> tuple[str, str]:
    if status in HEALTHY_STATUSES:
        return "OK", "replication completed"
    if status == "FAILED":
        return "CRITICAL", "S3 reports replication failure"
    if status == "PENDING":
        if age_hours is None or age_hours >= pending_hours:
            return "CRITICAL", f"replication is still pending after {pending_hours:g} hours"
        return "OK", "replication is pending within the allowed window"
    if status == "REPLICA":
        return "CRITICAL", "source object unexpectedly reports replica status"
    if age_hours is None or age_hours >= pending_hours:
        return "CRITICAL", "stale object has no recognized replication status"
    return "WARNING", "recent object has no recognized replication status"


def _list_versions(client: AwsCli, source: SourceConfig) -> list[dict[str, object]]:
    arguments = ["s3api", "list-object-versions", "--bucket", source.bucket]
    if source.prefix:
        arguments.extend(["--prefix", source.prefix])
    payload = client.run_json(arguments)
    if payload.get("IsTruncated") is True:
        raise S3IngestError(
            "AWS CLI returned a truncated version listing; refusing partial "
            "replication coverage"
        )
    versions = payload.get("Versions", [])
    if not isinstance(versions, list):
        raise S3IngestError("S3 version listing had an unexpected Versions value")
    return [item for item in versions if isinstance(item, dict)]


def inspect_source(
    client: AwsCli,
    source: SourceConfig,
    *,
    pending_hours: float = 24.0,
    retention_days: float = 37.0,
    now: datetime | None = None,
) -> tuple[ReplicationFinding, ...]:
    """Inspect every source version still inside the lifecycle-risk window."""

    if pending_hours < 0:
        raise ValueError("pending_hours must be nonnegative")
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    current_time = now or datetime.now(timezone.utc)
    retention_hours = retention_days * 24.0
    objects = []
    for item in _list_versions(client, source):
        age = _age_hours(item.get("LastModified"), current_time)
        if age is None or age <= retention_hours:
            objects.append(item)
    objects.sort(
        key=lambda item: _parse_timestamp(item.get("LastModified"))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    findings: list[ReplicationFinding] = []
    for item in objects:
        key = item.get("Key")
        if not isinstance(key, str) or not key:
            continue
        version_id = item.get("VersionId")
        normalized_version_id = (
            str(version_id) if isinstance(version_id, str) and version_id else None
        )
        head_arguments = [
            "s3api",
            "head-object",
            "--bucket",
            source.bucket,
            "--key",
            key,
        ]
        if normalized_version_id is not None:
            head_arguments.extend(["--version-id", normalized_version_id])
        head = client.run_json(head_arguments)
        raw_status = head.get("ReplicationStatus", item.get("ReplicationStatus"))
        status = str(raw_status).strip().upper() if raw_status else "UNKNOWN"
        last_modified = head.get("LastModified", item.get("LastModified", ""))
        age = _age_hours(last_modified, current_time)
        severity, message = _classify(status, age, pending_hours)
        raw_size = head.get("ContentLength", item.get("Size"))
        size = int(raw_size) if isinstance(raw_size, (int, float)) else None
        findings.append(
            ReplicationFinding(
                source=source.name,
                key=key,
                version_id=normalized_version_id,
                is_latest=bool(item.get("IsLatest", False)),
                last_modified=str(last_modified),
                size_bytes=size,
                status=status,
                age_hours=age,
                severity=severity,
                message=message,
            )
        )
    return tuple(findings)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read recent S3 source-object replication states."
    )
    parser.add_argument("sources", nargs="+", help="Configured sources, e.g. xband uhf")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--aws-cli", default=None)
    parser.add_argument("--pending-hours", type=float, default=24.0)
    parser.add_argument(
        "--retention-days",
        type=float,
        default=37.0,
        help="Inspect every version within this current-plus-noncurrent lifecycle window.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    executable = args.aws_cli or shutil.which("aws")
    if not executable:
        raise SystemExit("AWS CLI was not found; pass --aws-cli explicitly")

    all_findings: list[ReplicationFinding] = []
    empty_sources: list[str] = []
    for source_name in args.sources:
        source = load_source_config(args.config, source_name)
        client = AwsCli(executable, profile=source.profile, region=source.region)
        findings = inspect_source(
            client,
            source,
            pending_hours=args.pending_hours,
            retention_days=args.retention_days,
        )
        if not findings:
            empty_sources.append(source_name)
        all_findings.extend(findings)

    if args.as_json:
        print(
            json.dumps(
                {
                    "findings": [asdict(finding) for finding in all_findings],
                    "empty_sources": empty_sources,
                },
                indent=2,
            )
        )
    else:
        for source_name in empty_sources:
            print(f"WARNING {source_name}: no objects found under configured prefix")
        for finding in all_findings:
            age = "unknown" if finding.age_hours is None else f"{finding.age_hours:.2f}h"
            print(
                f"{finding.severity} {finding.source} {finding.status} "
                f"age={age} key={finding.key!r} version={finding.version_id!r}: "
                f"{finding.message}"
            )

    codes = [finding.exit_code for finding in all_findings]
    if empty_sources:
        codes.append(1)
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(run())
