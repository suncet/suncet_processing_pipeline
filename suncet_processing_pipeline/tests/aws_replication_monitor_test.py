from datetime import datetime, timezone
from pathlib import Path

import pytest

from suncet_processing_pipeline.aws_replication_monitor import inspect_source
from suncet_processing_pipeline.ingest_s3 import S3IngestError, SourceConfig


class FakeAwsCli:
    def __init__(self, objects, heads, *, truncated=False):
        self.objects = objects
        self.heads = heads
        self.truncated = truncated
        self.calls = []

    def run_json(self, arguments):
        self.calls.append(list(arguments))
        if "list-object-versions" in arguments:
            return {"Versions": self.objects, "IsTruncated": self.truncated}
        key = arguments[arguments.index("--key") + 1]
        version = arguments[arguments.index("--version-id") + 1]
        return self.heads[(key, version)]


def _source():
    return SourceConfig(
        name="xband",
        bucket="private-source",
        prefix="passes/",
        destination=Path("telemetry/incoming/xband"),
        profile="soc",
        region="example-region",
    )


def test_replication_monitor_classifies_complete_failed_and_stale_pending():
    objects = [
        {
            "Key": "passes/complete.bin",
            "VersionId": "v1",
            "IsLatest": True,
            "LastModified": "2026-08-31T10:00:00Z",
        },
        {
            "Key": "passes/pending.bin",
            "VersionId": "v2",
            "IsLatest": False,
            "LastModified": "2026-08-29T10:00:00Z",
        },
        {
            "Key": "passes/failed.bin",
            "VersionId": "v3",
            "IsLatest": True,
            "LastModified": "2026-08-31T09:00:00Z",
        },
    ]
    heads = {
        ("passes/complete.bin", "v1"): {"ReplicationStatus": "COMPLETED", "ContentLength": 1},
        ("passes/pending.bin", "v2"): {"ReplicationStatus": "PENDING", "ContentLength": 2},
        ("passes/failed.bin", "v3"): {"ReplicationStatus": "FAILED", "ContentLength": 3},
    }
    findings_client = FakeAwsCli(objects, heads)
    findings = inspect_source(
        findings_client,
        _source(),
        pending_hours=24,
        now=datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
    )
    by_key = {finding.key: finding for finding in findings}

    assert by_key["passes/complete.bin"].severity == "OK"
    assert by_key["passes/pending.bin"].severity == "CRITICAL"
    assert by_key["passes/failed.bin"].severity == "CRITICAL"
    assert any("--version-id" in call for call in findings_client.calls)


def test_replication_monitor_covers_all_versions_in_retention_window_and_flags_stale_unknown():
    objects = [
        {"Key": "passes/too-old.bin", "VersionId": "old", "LastModified": "2026-06-01T00:00:00Z"},
        {
            "Key": "passes/old-version.bin",
            "VersionId": "v1",
            "IsLatest": False,
            "LastModified": "2026-08-29T00:00:00Z",
        },
        {
            "Key": "passes/new-version.bin",
            "VersionId": "v2",
            "IsLatest": True,
            "LastModified": "2026-08-31T11:00:00Z",
        },
    ]
    client = FakeAwsCli(
        objects,
        {
            ("passes/old-version.bin", "v1"): {},
            ("passes/new-version.bin", "v2"): {},
        },
    )
    findings = inspect_source(
        client,
        _source(),
        retention_days=37,
        now=datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
    )
    assert [finding.version_id for finding in findings] == ["v2", "v1"]
    assert findings[0].severity == "WARNING"
    assert findings[1].severity == "CRITICAL"
    assert all("too-old" not in call for call in map(str, client.calls))


def test_replication_monitor_refuses_truncated_version_coverage():
    with pytest.raises(S3IngestError, match="truncated"):
        inspect_source(FakeAwsCli([], {}, truncated=True), _source())
