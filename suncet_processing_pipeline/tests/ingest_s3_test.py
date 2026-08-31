import json
from pathlib import Path

import pytest

from suncet_processing_pipeline.ingest_s3 import (
    S3IngestConflictError,
    S3IngestError,
    SourceConfig,
    ingest_object,
    load_source_config,
)


class FakeAwsCli:
    def __init__(self, content: bytes, *, version_id: str = "version-1") -> None:
        self.content = content
        self.version_id = version_id
        self.calls: list[list[str]] = []

    def run_json(self, arguments):
        self.calls.append(list(arguments))
        destination = Path(arguments[-1])
        destination.write_bytes(self.content)
        return {
            "ContentLength": len(self.content),
            "ETag": '"example-etag"',
            "LastModified": "2026-08-26T00:00:00+00:00",
            "VersionId": self.version_id,
            "ChecksumSHA256": "example-s3-checksum",
        }


class RacingFakeAwsCli(FakeAwsCli):
    def __init__(self, content: bytes, destination: Path, competing_content: bytes):
        super().__init__(content)
        self.destination = destination
        self.competing_content = competing_content

    def run_json(self, arguments):
        metadata = super().run_json(arguments)
        self.destination.write_bytes(self.competing_content)
        return metadata


def source_config() -> SourceConfig:
    return SourceConfig(
        name="xband",
        bucket="private-delivery-name",
        prefix="passes/",
        destination=Path("telemetry/incoming/xband"),
        profile="suncet-ingest",
        region="us-east-2",
    )


def test_load_source_config_keeps_resource_names_host_local(tmp_path):
    config = tmp_path / "aws_ingest.ini"
    config.write_text(
        "[aws]\nprofile = suncet-ingest\nregion = us-east-2\n\n"
        "[xband]\nbucket = private-name\nprefix = passes/\n"
        "destination = telemetry/incoming/xband\n",
        encoding="utf-8",
    )

    loaded = load_source_config(config, "xband")

    assert loaded.bucket == "private-name"
    assert loaded.destination == Path("telemetry/incoming/xband")


def test_load_source_config_rejects_destination_escape(tmp_path):
    config = tmp_path / "aws_ingest.ini"
    config.write_text(
        "[aws]\nprofile = p\nregion = r\n"
        "[xband]\nbucket = b\ndestination = ../private\n",
        encoding="utf-8",
    )

    with pytest.raises(S3IngestError, match="safe path"):
        load_source_config(config, "xband")


def test_ingest_is_atomic_hashed_receipted_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))
    client = FakeAwsCli(b"SunCET pass data")
    state_directory = tmp_path.parent / f"{tmp_path.name}-private-state"

    destination, receipt, status = ingest_object(
        client,
        source_config(),
        "passes/pass-001.tm",
        state_directory=state_directory,
    )
    second_destination, second_receipt, second_status = ingest_object(
        client,
        source_config(),
        "passes/pass-001.tm",
        state_directory=state_directory,
    )

    assert destination.read_bytes() == b"SunCET pass data"
    assert status == "downloaded"
    assert second_destination == destination
    assert second_status == "already_present"
    assert receipt != second_receipt
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["bucket"] == "private-delivery-name"
    assert payload["version_id"] == "version-1"
    assert payload["size_bytes"] == 16
    assert len(payload["sha256"]) == 64
    assert payload["local_path"] == "telemetry/incoming/xband/pass-001.tm"
    assert not list(tmp_path.rglob("*.partial"))
    assert receipt.is_relative_to(state_directory)
    assert receipt.stat().st_mode & 0o077 == 0


def test_ingest_refuses_private_receipts_below_public_data(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))
    with pytest.raises(S3IngestError, match="overlap suncet_data"):
        ingest_object(
            FakeAwsCli(b"data"),
            source_config(),
            "passes/pass-001.tm",
            state_directory=tmp_path / "transfer_logs",
        )


def test_ingest_refuses_state_directory_that_contains_public_data(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("suncet_data", str(data_root))
    tmp_path.chmod(0o755)
    original_mode = tmp_path.stat().st_mode & 0o777

    with pytest.raises(S3IngestError, match="must not overlap suncet_data"):
        ingest_object(
            FakeAwsCli(b"data"),
            source_config(),
            "passes/pass-001.tm",
            state_directory=tmp_path,
        )

    assert tmp_path.stat().st_mode & 0o777 == original_mode


def test_ingest_refuses_same_name_with_different_content(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))
    destination = tmp_path / "telemetry/incoming/xband/pass-001.tm"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"older different data")

    with pytest.raises(S3IngestConflictError, match="Refusing to overwrite"):
        ingest_object(FakeAwsCli(b"new data"), source_config(), "passes/pass-001.tm")

    assert destination.read_bytes() == b"older different data"
    assert not list(tmp_path.rglob("*.partial"))


def test_ingest_concurrent_destination_creation_never_overwrites(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("suncet_data", str(data_root))
    destination = data_root / "telemetry/incoming/xband/pass-001.tm"
    client = RacingFakeAwsCli(b"downloaded data", destination, b"competing data")

    with pytest.raises(S3IngestConflictError, match="Refusing to overwrite"):
        ingest_object(
            client,
            source_config(),
            "passes/pass-001.tm",
            state_directory=tmp_path / "private-state",
        )

    assert destination.read_bytes() == b"competing data"
    assert not list(data_root.rglob("*.partial"))


def test_ingest_rejects_key_outside_configured_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))

    with pytest.raises(S3IngestError, match="outside configured prefix"):
        ingest_object(FakeAwsCli(b"data"), source_config(), "other/pass.tm")
