"""Validate the upstream decoder and DB point construction without uploading data.

Replays private IQ through the pinned SatNOGS receiver, then executes the actual
DB ingestion functions with in-memory file and database substitutes. Reports
aggregate checks only; raw frames and recorded field values stay in memory.
"""
from __future__ import annotations

import argparse
import ast
import binascii
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/satnogs_receiver_replay"))

from replay import load_receiver, receive
from suncet_processing_pipeline.satnogs.public_schema import load_public_beacon_schema


def git_revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def mission_reference(frames):
    """Use the mission's 0.11 runtime separately from upstream's 0.10 runtime.

    Private frames/values cross only local in-memory pipes, never command-line
    arguments or files. The active conda suncet environment supplies 0.11.
    """
    script = '''
import json, sys
import kaitaistruct
from suncet_processing_pipeline.satnogs.ax25 import decode_public_ax25_beacon
from suncet_processing_pipeline.satnogs.beacon_contract import BeaconValidationError
from suncet_processing_pipeline.satnogs.build_upstream import upstream_expected
assert kaitaistruct.__version__ == '0.11'
values = []
for text in json.load(sys.stdin):
    try:
        values.append(upstream_expected(decode_public_ax25_beacon(bytes.fromhex(text), fcs_present=False)))
    except BeaconValidationError:
        values.append(None)
json.dump(values, sys.stdout)
'''
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment,
                            input=json.dumps([frame.hex() for frame in frames]),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def load_db_functions(path, decoder):
    """Keep the source functions unchanged; substitute only external effects."""
    names = {"remove_nan_fields", "flatten_data", "create_point", "decode_demoddata"}
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = {"math": math, "binascii": binascii, "decoder": decoder,
                 "decoders_version": "local-suncet-integration",
                 "settings": SimpleNamespace(USE_INFLUX=True)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def ingest(namespace, frame, sequence):
    """Exercise DB dispatch, decoder, point construction, and decode outcome."""
    points, updates = [], []
    payload = SimpleNamespace(path="in-memory-private-frame")
    timestamp = datetime(2026, 10, 5, tzinfo=timezone.utc) + timedelta(
        seconds=sequence, microseconds=123000)
    demod = SimpleNamespace(payload_frame=payload, timestamp=timestamp, id=sequence,
                            station="local-test", observer="local-test", app_source="local-test")
    satellite = SimpleNamespace(satellite_entry=SimpleNamespace(
        norad_cat_id=98244, name="SunCET", satellite_identifier=SimpleNamespace(
            sat_id="MNRC-9829-4319-5529-8975")))

    def open_frame(path, encoding):
        assert path == payload.path and encoding == "utf-8"
        return io.StringIO(frame.hex())

    namespace.update(open=open_frame, write_influx=points.extend,
                     DemodData=SimpleNamespace(objects=SimpleNamespace(
                         filter=lambda **kwargs: SimpleNamespace(update=lambda **values: updates.append(values)))))
    namespace["decode_demoddata"](demod, satellite, SimpleNamespace(decoder="suncet"))
    assert len(updates) == 1
    assert updates[0]["is_decoded"] == bool(points)
    if points:
        assert len(points) == 1
        point = points[0]
        assert point["time"] == timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert point["measurement"] == 98244
        assert point["tags"] == {
            "satellite": "SunCET", "sat_id": "MNRC-9829-4319-5529-8975", "decoder": "suncet",
            "station": "local-test", "observer": "local-test", "source": "local-test",
            "version": "local-suncet-integration"}
        json.dumps(point, allow_nan=False)
        assert all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in point["fields"].values())
    return points


def repair_checksum(frame):
    result = bytearray(frame)
    words = struct.unpack("<124H", result[16:264])
    low = sum(words) % 65535
    high = sum((124 - i) * word for i, word in enumerate(words)) % 65535
    result[-4:] = struct.pack(">I", high * 65536 + low)
    return bytes(result)


def validate_dashboard(path, aliases, point_fields):
    dashboard = json.loads(path.read_text())
    queried = set()
    panels = 0
    state_fields = {"CDH mode": "mode_system_mode", "ADCS mode": "adcs_mode",
                    "Sun-pointing state": "adcs_sun_point_state", "UHF health": "uhf_alive"}
    schema = {field.public_name: field for field in load_public_beacon_schema()}
    for panel in dashboard["panels"]:
        if panel.get("targets"):
            panels += 1
        for target in panel.get("targets", []):
            assert target["hide"] is True
            assert 'FROM "98244"' in target["query"]
            assert target["datasource"]["uid"] == "000000001"
            queried.update(re.findall(r'(?:SELECT |, )"([a-z][a-z0-9_]*)"', target["query"]))
        if panel["title"] in state_fields:
            name = state_fields[panel["title"]]
            choices = re.findall(r"(\d+)/([A-Z][A-Z0-9_]*)", schema[name].conversion_or_status_map)
            overrides = panel["fieldConfig"]["overrides"]
            for override in overrides:
                for prop in override["properties"]:
                    if prop["id"] == "mappings":
                        options = prop["value"][0]["options"]
                        assert all(code in options and label in options for code, label in choices)
                        assert all(options[code] == options[label] for code, label in choices)
    assert panels == 38 and len(queried) == 82
    assert queried <= aliases and queried <= point_fields
    return {"dashboard_uid": dashboard["uid"], "saved_version": dashboard["version"],
            "telemetry_panels": panels, "queried_public_aliases": len(queried),
            "all_aliases_present_in_decoded_points": True,
            "four_canvas_state_panels_support_numeric_codes": True,
            "all_hosted_targets_still_disabled": True}


def validate(args):
    sys.path.insert(0, str(args.decoders.resolve()))
    from satnogsdecoders import decoder
    from satnogsdecoders.decode_frame import decode_frame_to_dict, decode_frame_to_fields
    import kaitaistruct
    assert kaitaistruct.__version__ == "0.10"
    aliases = {field.public_name for field in load_public_beacon_schema()}
    assert len(aliases) == 111
    assert set(decoder.get_fields(decoder.Suncet, empty=True)) == aliases
    source = args.db / "db/base/utils.py"
    namespace = load_db_functions(source, decoder)
    raw = args.iq.read_bytes()
    assert len(raw) % 8 == 0
    iq = np.frombuffer(raw, dtype="<c8")
    assert len(iq) and np.isfinite(iq).all()
    frames = receive(iq, load_receiver(args.receiver.resolve()))
    expected_frames = mission_reference(frames)
    assert len(expected_frames) == len(frames)
    accepted = rejected = 0
    recorded_points = []
    for sequence, (frame, expected) in enumerate(zip(frames, expected_frames)):
        points = ingest(namespace, frame, sequence)
        if expected is None:
            assert not points
            rejected += 1
            continue
        assert len(points) == 1
        values = decode_frame_to_fields("suncet", frame)
        assert set(values) == aliases
        for name, value in expected.items():
            if isinstance(value, float) and not math.isfinite(value):
                assert math.isnan(values[name]) and name not in points[0]["fields"]
            else:
                assert math.isclose(values[name], value, rel_tol=1e-12, abs_tol=1e-12), name
                assert points[0]["fields"][name] == values[name]
        assert set(decode_frame_to_dict("suncet", frame)) == {"beacon"}
        recorded_points.extend(points)
        accepted += 1
    assert len(frames) == 20 and accepted == 10 and rejected == 10
    fixture = bytes.fromhex((args.decoders / "contrib/suncet/synthetic.hex").read_text())
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        frame = bytearray(fixture)
        frame[84:88] = struct.pack(">f", nonfinite)
        points = ingest(namespace, repair_checksum(frame), 100)
        assert len(points) == 1 and len(points[0]["fields"]) == 110
        assert "dsps_flare_level" not in points[0]["fields"]
    unknown = bytearray(fixture)
    unknown[16 + 227] = 255
    point = ingest(namespace, repair_checksum(unknown), 101)[0]
    assert point["fields"]["mode_system_mode"] == 255
    assert isinstance(point["fields"]["mode_system_mode"], int)
    broken = bytearray(fixture)
    broken[16 + 124] ^= 1
    assert not ingest(namespace, broken, 102)
    dashboard = validate_dashboard(args.dashboard, aliases,
                                   set.intersection(*(set(p["fields"]) for p in recorded_points)))
    return {
        "validated_on": "2026-10-05", "decoder_base_revision": git_revision(args.decoders),
        "upstream_python_runtime": kaitaistruct.__version__, "mission_reference_runtime": "0.11",
        "db_revision": git_revision(args.db), "db_utils_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "receiver": json.loads((args.receiver / "build-manifest.json").read_text()),
        "results": {"receiver_frames": len(frames), "beacons_ingested_in_memory": accepted,
                    "other_frames_rejected": rejected, "approved_fields_per_beacon": 111,
                    "all_public_values_match_mission_decoder": True,
                    "ingested_field_counts": sorted({len(p["fields"]) for p in recorded_points}),
                    "raw_structure_top_level_keys": ["beacon"],
                    "nan_and_both_infinities_omitted_without_losing_other_fields": True,
                    "unknown_state_preserves_integer_field_type": True,
                    "corrupt_excluded_region_rejected": True},
        "db_contract": {"measurement": "satellite NORAD number (98244 currently proposed)",
                        "field_names": "111 flat public aliases; engineering values and integer state codes",
                        "timestamp": "demoddata reception timestamp, formatted to whole UTC seconds",
                        "tested_timestamp_metadata": "synthetic local metadata; not recording reception times",
                        "tags": ["satellite", "sat_id", "decoder", "station", "observer", "source", "version"]},
        "dashboard": dashboard,
        "scope": "Local released-receiver replay, compiled upstream decoder, and unchanged DB ingestion functions with in-memory I/O substitutes. No live DB, Influx server, Network upload, or Grafana query/render test.",
        "privacy": "No IQ samples, packet bytes, recorded telemetry values, or other packet identifiers exported."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iq", type=Path)
    parser.add_argument("--decoders", type=Path, default=Path("tmp/satnogs-decoders"))
    parser.add_argument("--db", type=Path, default=Path("tmp/satnogs-db"))
    parser.add_argument("--receiver", type=Path, default=Path("tmp/satnogs-receiver"))
    parser.add_argument("--dashboard", type=Path, default=Path(
        "suncet_processing_pipeline/satnogs/dashboard/suncet-dev.saved.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"results": report["results"], "dashboard": report["dashboard"]}, indent=2))


if __name__ == "__main__":
    main()
