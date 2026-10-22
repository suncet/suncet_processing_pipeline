"""No-network regression tests for unattended UHF S3 collection."""

import hashlib
import json
from pathlib import Path

import pytest

from suncet_processing_pipeline import sync_s3 as s
from suncet_processing_pipeline.ingest_s3 import SourceConfig, S3IngestError


def item(key="uhf/pass.bin", version="v1", content=b"beacon"):
    return {"Key": key, "VersionId": version, "Size": len(content),
            "ETag": '"' + hashlib.md5(content).hexdigest() + '"',
            "LastModified": "2026-10-22T00:00:00Z", "IsLatest": True}


def page(rows=(), truncated=False, **markers):
    return {"Versions": list(rows), "IsTruncated": truncated, **markers}


class Client:
    def __init__(self, pages, content=b"beacon"):
        self.pages = iter(pages)
        self.content = content
        self.calls = []
        self.fail = False

    def run_json(self, args):
        self.calls.append(args)
        if args[1] == "list-object-versions":
            result = next(self.pages)
            if isinstance(result, Exception):
                raise result
            return result
        assert args[1] == "get-object"
        if self.fail:
            raise S3IngestError("interrupted")
        Path(args[-1]).write_bytes(self.content)
        return {"ContentLength": len(self.content), "VersionId": args[args.index("--version-id")+1],
                "ETag": args[args.index("--if-match")+1]}


@pytest.fixture
def context(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("suncet_data", str(data))
    monkeypatch.delenv("suncet_ctdb", raising=False)
    source = SourceConfig("uhf", "private-example", "uhf/", Path("telemetry/incoming/uhf"), "test", "test")
    return source, tmp_path / "state"


def run(context, client, **kwargs):
    return s.sync(client, *context, lambda size: None, **kwargs)


def test_download_receipt_and_repeat_without_get(context):
    result = run(context, Client([page([item()])]))
    assert result["downloaded"] == 1
    second = Client([page([item()])])
    assert run(context, second)["already_present"] == 1
    assert [call[1] for call in second.calls] == ["list-object-versions"]
    assert s.status(context[1])["retained_versions"] == 1
    receipt = next((context[1]/"receipts").glob("*.json"))
    assert json.loads(receipt.read_text())["sha256"] == hashlib.sha256(b"beacon").hexdigest()


def test_two_versions_and_same_basename_have_separate_paths(context):
    run(context, Client([page([item(), item(version="v2"), item(key="uhf/other/pass.bin")])]))
    receipts = [json.loads(p.read_text()) for p in (context[1]/"receipts").glob("*.json")]
    assert len({r["local_path"] for r in receipts}) == 3
    assert s.status(context[1])["retained_versions"] == 3


def test_null_version_replacement_is_new_identity(context):
    run(context, Client([page([item(version="null")])]))
    changed = b"new-beacon"
    result = run(context, Client([page([item(version="null", content=changed)])], changed))
    assert result["downloaded"] == 1
    assert s.status(context[1])["retained_versions"] == 2


def test_both_s3_pagination_markers_are_used(context):
    client = Client([page([item()], True, NextKeyMarker="uhf/pass.bin", NextVersionIdMarker="v1"), page()])
    run(context, client)
    last = client.calls[-1]
    assert last[last.index("--key-marker")+1] == "uhf/pass.bin"
    assert last[last.index("--version-id-marker")+1] == "v1"
    assert "--no-paginate" in last


def test_partial_listing_failure_keeps_completed_objects_not_success(context):
    run(context, Client([page()]))
    before = s.status(context[1])["last_success"]
    with pytest.raises(S3IngestError):
        run(context, Client([page([item()], True, NextKeyMarker="uhf/pass.bin"), S3IngestError("offline")]))
    assert s.status(context[1])["last_success"] == before
    assert s.status(context[1])["retained_versions"] == 1
    assert run(context, Client([page([item()])]))["already_present"] == 1


def test_missing_local_file_is_downloaded_again(context):
    run(context, Client([page([item()])]))
    payload = json.loads(next((context[1]/"receipts").glob("*.json")).read_text())
    (s.get_data_root()/payload["local_path"]).unlink()
    assert run(context, Client([page([item()])]))["downloaded"] == 1


def test_corrupt_local_file_is_not_silently_overwritten(context):
    run(context, Client([page([item()])]))
    payload = json.loads(next((context[1]/"receipts").glob("*.json")).read_text())
    target = s.get_data_root()/payload["local_path"]
    target.write_bytes(b"damage")
    with pytest.raises(S3IngestError, match="overwrite"):
        run(context, Client([page([item()])]))
    assert target.read_bytes() == b"damage"


def test_budget_stops_incomplete_run_and_preserves_prior_download(context):
    with pytest.raises(S3IngestError, match="budget"):
        run(context, Client([page([item(), item(version="v2")])]), max_download_bytes=6)
    assert s.status(context[1])["last_success"] is None
    assert s.status(context[1])["retained_versions"] == 1
    assert run(context, Client([page([item(), item(version="v2")])]), max_download_bytes=6)["downloaded"] == 1


def test_preflight_failure_prevents_requests(context):
    client = Client([])
    def stop(size):
        raise S3IngestError("storage")
    with pytest.raises(S3IngestError, match="storage"):
        s.sync(client, *context, stop)
    assert client.calls == []


@pytest.mark.parametrize("bad", [page([item(key="other/file")]), {}, page([{"Key":"uhf/x"}]),
                                 page([], True), page([], True, NextKeyMarker=42)])
def test_malformed_listing_never_marks_success(context, bad):
    with pytest.raises(S3IngestError):
        run(context, Client([bad]))
    assert s.status(context[1])["last_success"] is None


def test_delete_markers_and_folder_markers_do_not_delete_local_files(context):
    run(context, Client([page([item()])]))
    result = run(context, Client([page([item(key="uhf/folder/", content=b"")], DeleteMarkers=[{"Key":"uhf/pass.bin"}])]))
    assert result["versions_seen"] == 0
    assert s.status(context[1])["retained_versions"] == 1


def test_download_changed_between_listing_and_get_is_not_finalized(context):
    with pytest.raises(S3IngestError, match="listed"):
        run(context, Client([page([item()])], content=b"changed-and-longer"))
    assert not list(s.get_data_root().rglob("pass.bin"))


def test_concurrent_collector_is_refused(context):
    with s.database(context[1]):
        with pytest.raises(S3IngestError, match="already running"):
            run(context, Client([page()]))


@pytest.mark.parametrize("field,value", [("ETag", '"wrong"'), ("VersionId", "wrong")])
def test_response_identity_mismatch_is_not_finalized(context, field, value):
    class ChangedResponse(Client):
        def run_json(self, args):
            response = super().run_json(args)
            if args[1] == "get-object":
                response[field] = value
            return response
    with pytest.raises(S3IngestError, match="differs"):
        run(context, ChangedResponse([page([item()])]))
    assert not list(s.get_data_root().rglob("pass.bin"))


def test_repeated_pagination_markers_and_page_limit_fail(context):
    repeat = page([], True, NextKeyMarker="uhf/pass.bin", NextVersionIdMarker="v1")
    with pytest.raises(S3IngestError, match="Repeated"):
        run(context, Client([repeat, repeat]))
    with pytest.raises(S3IngestError, match="budget"):
        run(context, Client([repeat]), max_pages=1)
    assert s.status(context[1])["last_success"] is None
