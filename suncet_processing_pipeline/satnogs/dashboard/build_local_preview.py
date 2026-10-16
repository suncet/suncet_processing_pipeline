"""Build a local Grafana preview using only validated public beacon values.

Run in the suncet conda environment. This reads captures without modifying
them; it never uploads data. Outputs contain recorded public telemetry and
belong in an ignored local directory, not the public source repository.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from suncet_processing_pipeline.satnogs.beacon_contract import (
    BeaconValidationError,
    parse_beacon_packet,
)
from suncet_processing_pipeline.satnogs.decoder import decode_public_beacon
from suncet_processing_pipeline.satnogs.dashboard.build_dashboard import build, FIELDS
from suncet_processing_pipeline.satnogs.dashboard.preview_panels import enhance_preview


EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
DATASOURCE = {"type": "grafana-testdata-datasource", "uid": "suncet-local-csv"}
UID = "suncet-recorded-local"


def capture_rows(root: Path, start: datetime, end: datetime):
    """Return unique valid public rows in the declared capture-day window."""

    counts = Counter()
    seen = set()
    rows = []
    sources = []
    files = sorted(root.glob("*/ccsds_*"))
    if not files:
        raise ValueError(f"No */ccsds_* input files below {root}")
    for path in files:
        data = path.read_bytes()
        sources.append({"file": str(path.relative_to(root)), "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
        for match in re.finditer(rb"\x08\x01..\x00\xf5", data, re.DOTALL):
            packet = data[match.start():match.start() + 252]
            counts["candidates"] += 1
            try:
                metadata = parse_beacon_packet(packet)
            except BeaconValidationError:
                counts["invalid_candidates"] += 1
                continue
            counts["valid_occurrences"] += 1
            digest = hashlib.sha256(packet).digest()
            if digest in seen:
                counts["duplicate_occurrences_removed"] += 1
                continue
            seen.add(digest)
            timestamp = EPOCH + timedelta(seconds=metadata.timestamp_seconds)
            if not start <= timestamp < end:
                counts["unique_packets_outside_capture_day"] += 1
                continue
            rows.append((timestamp, decode_public_beacon(packet)))
    rows.sort(key=lambda row: row[0])
    counts["unique_valid_packets"] = len(seen)
    counts["preview_packets"] = len(rows)
    if not rows:
        raise ValueError("No valid packets with a spacecraft clock in the requested day")
    return rows, dict(counts), sources


def frame_values(names, rows):
    """Serialize typed Grafana fields, not raw spacecraft packet bytes.

    Explicit types prevent Grafana's CSV importer from interpreting numeric
    fields containing 'time' in their names as Unix timestamps.
    """

    if not set(names) <= FIELDS.keys():
        raise ValueError("Preview includes an unapproved field")
    fields = [{"name": "Time", "type": "time", "config": {}}]
    columns = [[round(timestamp.timestamp() * 1000) for timestamp, _ in rows]]
    for name in names:
        values = [row[name] for _, row in rows]
        field_type = "string" if any(isinstance(v, str) for v in values) else "number"
        fields.append({"name": name, "type": field_type, "config": {}})
        columns.append([None if isinstance(v, float) and not math.isfinite(v)
                        else str(v) if field_type == "string" else v for v in values])
    return json.dumps([{"schema": {"name": "SunCET recorded public beacon", "refId": "A",
                                   "fields": fields}, "data": {"values": columns}}],
                      allow_nan=False)


def build_preview(rows, counts, capture_date, image_uri=None):
    dashboard = build()
    dashboard.update({
        "uid": UID, "title": "suncet-dev · recorded preview", "id": None, "version": 0,
        "tags": ["SunCET", "local-preview", "recorded-test-data", "CTDB-2.0.5"],
        "description": "Local recorded test-data preview; no SatNOGS observations or live ingestion. "
                       "Only checksum-valid public CTDB 2.0.5 fields. Time is the reported spacecraft "
                       "clock against its declared epoch, not reception UTC.",
        "time": {"from": (rows[0][0] - timedelta(minutes=1)).isoformat(),
                 "to": (rows[-1][0] + timedelta(minutes=1)).isoformat()},
    })
    used = set()
    for panel in dashboard["panels"]:
        if not panel.get("targets"):
            continue
        names = [override["matcher"]["options"] for override in panel["fieldConfig"]["overrides"]]
        used.update(names)
        for override in panel["fieldConfig"]["overrides"]:
            if override["matcher"]["options"] == "spacecraft_time_seconds_since_2000":
                for prop in override["properties"]:
                    if prop["id"] == "unit":
                        prop["value"] = "suffix:s"
                override["properties"].append({"id": "decimals", "value": 0})
        latest = panel["type"] in {"stat", "table", "bargauge"}
        panel["datasource"] = DATASOURCE.copy()
        panel["targets"] = [{"refId": "A", "datasource": DATASOURCE.copy(), "hide": False,
                             "scenarioId": "raw_frame",
                             "rawFrameContent": frame_values(names, rows[-1:] if latest else rows)}]
        panel["description"] = panel["description"].replace(
            "Staged CTDB 2.0.5 public engineering values; queries disabled pending ingestion verification.",
            f"Recorded {capture_date} CPT public engineering values. Local preview; no live reception.")
        if latest:
            panel["description"] += " Fixed final recorded packet; this summary does not follow the time picker."
        else:
            panel["description"] += " Time axis is the reported spacecraft clock, not ground reception time."
            panel["fieldConfig"]["defaults"]["custom"]["axisLabel"] = ""
    assert len(used) == 81
    dashboard["panels"][0]["title"] = "SunCET • recorded test-data preview"
    dashboard["panels"][0]["options"]["content"] = (
        "## Sun Coronal Ejection Tracker\n**LOCAL PREVIEW · RECORDED TEST DATA · " + capture_date + " CPT**\n\n"
        f"{len(rows):,} distinct, checksum-valid APID 1 beacons · 81 approved public fields · CTDB 2.0.5. "
        "Use **Edit** to try panel layouts, units and styling. Changes stay in this local Grafana. "
        "This is a recorded snapshot, not a live SatNOGS observation."
    )
    reception = next(p for p in dashboard["panels"] if p["title"] == "Reception information pending")
    reception["title"] = "Capture & time reference"
    reception["options"]["content"] = (
        "**Time axis:** reported spacecraft clock against 2000-01-01; provisional UTC display, not reception time. "
        f"{counts['unique_packets_outside_capture_day']:,} distinct packets outside the capture day "
        "(including unset startup clocks) are omitted. Duplicate packets removed.\n\n"
        "**Summary cards/tables:** final recorded packet, independent of the time picker. "
        "No reception-age, station/observation or flight-alarm claims."
    )
    histogram_examples = [(t, r) for t, r in rows if any(r[f"csie_img_hist_{i}"] for i in range(6))]
    example = ((histogram_examples[-1][0].isoformat(), histogram_examples[-1][1])
               if histogram_examples else None)
    return enhance_preview(dashboard, rows[-1][1], rows[-1][0].isoformat(), image_uri, example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--date", required=True, help="Capture day YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path, help="Ignored local output directory")
    parser.add_argument("--image", type=Path, help="Optional local PNG spacecraft render to embed")
    args = parser.parse_args()
    start = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows, counts, sources = capture_rows(args.input_root, start, start + timedelta(days=1))
    image_uri = None
    if args.image:
        image_uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
    dashboard = build_preview(rows, counts, args.date, image_uri)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dashboard.json").write_text(json.dumps(dashboard, allow_nan=False) + "\n")
    report = {"capture_date": args.date, "counts": counts, "sources": sources,
              "public_fields": 82, "telemetry_panels": sum(bool(p.get("targets")) for p in dashboard["panels"]),
              "time_semantics": "spacecraft clock against declared epoch; not reception UTC",
              "first_time": rows[0][0].isoformat(), "last_time": rows[-1][0].isoformat()}
    (args.output / "preview_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
