import hashlib
import json

import pytest

from .. import run_provenance


def _stub_host_details(monkeypatch):
    monkeypatch.setattr(
        run_provenance,
        "_git_snapshot",
        lambda _path: {
            "available": True,
            "commit": "a" * 40,
            "branch": "main",
            "dirty": False,
            "status": [],
        },
    )
    monkeypatch.setattr(
        run_provenance,
        "_system_snapshot",
        lambda: {"hostname": "test-host", "architecture": "test-arch"},
    )
    monkeypatch.setattr(
        run_provenance,
        "_installed_packages",
        lambda: {"numpy": "2.0.0", "sunpy": "8.0.0"},
    )


def test_successful_processing_manifest_records_inputs_outputs_and_redaction(
    tmp_path, monkeypatch
):
    _stub_host_details(monkeypatch)
    data_root = tmp_path / "data"
    data_root.mkdir()
    input_path = data_root / "input.bin"
    input_path.write_bytes(b"SunCET input")
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[paths]\ndata_to_process_path = science\n"
        "[auth]\napi_token = do-not-record\n",
        encoding="utf-8",
    )

    with run_provenance.ProcessingRunProvenance(
        data_root=data_root,
        run_kind="unit_test",
        config_path=config_path,
        resolved_config={"data_root": data_root, "password": "do-not-record"},
        arguments={"input_mode": "xband", "secret": "do-not-record"},
        argv=["processor", "--token", "do-not-record", "--input-mode=xband"],
    ) as provenance:
        provenance.record_inputs([input_path])
        output_path = data_root / "level0_5" / "output.bin"
        output_path.parent.mkdir()
        output_path.write_bytes(b"SunCET output")

    payload = json.loads(provenance.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "succeeded"
    assert payload["error"] is None
    assert payload["git"]["commit"] == "a" * 40
    assert payload["packages"] == {"numpy": "2.0.0", "sunpy": "8.0.0"}
    assert payload["inputs"] == [
        {
            "modified_utc": payload["inputs"][0]["modified_utc"],
            "path": "input.bin",
            "relative_to_data_root": True,
            "sha256": hashlib.sha256(b"SunCET input").hexdigest(),
            "size_bytes": len(b"SunCET input"),
        }
    ]
    outputs = payload["outputs"]["created_or_modified"]
    assert [record["path"] for record in outputs] == ["level0_5/output.bin"]
    assert outputs[0]["sha256"] == hashlib.sha256(b"SunCET output").hexdigest()
    assert payload["configuration"]["values"]["auth"]["api_token"] == "<redacted>"
    assert payload["resolved_configuration"]["password"] == "<redacted>"
    assert payload["invocation"]["arguments"]["secret"] == "<redacted>"
    assert payload["invocation"]["argv"][2] == "<redacted>"


def test_failed_processing_manifest_preserves_error_and_partial_outputs(
    tmp_path, monkeypatch
):
    _stub_host_details(monkeypatch)
    data_root = tmp_path / "data"

    with pytest.raises(RuntimeError, match="decoder failed"):
        with run_provenance.ProcessingRunProvenance(
            data_root=data_root,
            run_kind="unit_test_failure",
        ) as provenance:
            (data_root / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("decoder failed")

    payload = json.loads(provenance.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["error"]["message"] == "decoder failed"
    assert "raise RuntimeError" in payload["error"]["traceback"]
    assert [
        record["path"] for record in payload["outputs"]["created_or_modified"]
    ] == ["partial.bin"]


def test_resolved_config_hides_private_ctdb_root(tmp_path):
    private_root = tmp_path / "private" / "ctdb"

    class ConfigStub:
        ctdb_base = str(private_root)
        bus_ctdb_path = str(private_root / "suncet_v2-0-1")
        csie_ctdb_path = str(private_root / "suncet_csie_v1-1-2")
        packet_definitions_path = str(
            private_root / "suncet_v2-0-1" / "decoders"
        )
        version_bus = "2.0.1"
        version_csie = "1.1.2"

    snapshot = run_provenance.resolved_config_snapshot(
        ConfigStub(), tmp_path / "public-data"
    )

    assert snapshot["ctdb_base"] == "$suncet_ctdb"
    assert snapshot["bus_ctdb_path"] == "$suncet_ctdb/suncet_v2-0-1"
    assert snapshot["csie_ctdb_path"] == "$suncet_ctdb/suncet_csie_v1-1-2"
    assert snapshot["packet_definitions_path"] == (
        "$suncet_ctdb/suncet_v2-0-1/decoders"
    )
    assert str(private_root) not in str(snapshot)
