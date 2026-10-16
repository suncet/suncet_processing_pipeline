"""Build the online development layout without recorded telemetry or local sources.

All InfluxQL targets remain disabled until ingestion fields are verified.
An optional mission-owner supplied spacecraft PNG can be embedded for upload.
"""

import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import re

from .build_dashboard import build, DATASOURCE, FIELDS, mappings
from .preview_panels import enhance_preview, GRAY, INACTIVE_TEXT, STATE_COLORS, STATE_RULES


STAGED = "Staged CTDB 2.0.5 public engineering values; queries disabled pending ingestion verification. "
STATE_PANELS = {"CDH mode": "mode_system_mode", "ADCS mode": "adcs_mode",
                "Sun-pointing state": "adcs_sun_point_state", "UHF health": "uhf_alive"}
BIN_LABELS = [f"{i * 32}–{(i + 1) * 32 - 1}" for i in range(6)]


def target(columns, latest=True):
    """Select one coherent packet for summaries, allowing presentation aliases."""
    assert all(source in FIELDS for source, _ in columns)
    select = ", ".join(f'"{source}"' + (f' AS "{alias}"' if alias != source else "")
                       for source, alias in columns)
    return {"refId": "A", "datasource": DATASOURCE.copy(), "rawQuery": True,
            "query": f'SELECT {select} FROM "98244" WHERE $timeFilter ORDER BY time '
                     + ("DESC LIMIT 1" if latest else "ASC"),
            "resultFormat": "table" if latest else "time_series", "hide": True}


def wire_state(p, name):
    choices = re.findall(r"(\d+)/([A-Z][A-Z0-9_]*)", FIELDS[name]["conversion_or_status_map"])
    columns = [(name, name)]
    overrides = [{"matcher": {"id": "byName", "options": name},
                  "properties": [{"id": "mappings", "value": mappings(name)}]}]
    for element in p["options"]["root"]["elements"]:
        label = element["name"]
        color_name = STATE_RULES[name][label]
        for text_color in (False, True):
            alias = label + ("_text_color" if text_color else "")
            columns.append((name, alias))
            inactive = INACTIVE_TEXT if text_color else GRAY
            active = ("#17191D" if color_name == "yellow" else "#FFFFFF") if text_color else STATE_COLORS[color_name]
            options = {}
            for code, text in choices:
                display = {"text": text.replace("_", " "), "color": active if text == label else inactive}
                options[code] = display
                options[text] = display
            overrides.append({"matcher": {"id": "byName", "options": alias}, "properties": [
                {"id": "color", "value": {"mode": "fixed", "fixedColor": inactive}},
                {"id": "mappings", "value": [{"type": "value", "options": options},
                    {"type": "special", "options": {"match": "null", "result": {"text": "No data", "color": inactive}}}]},
            ]})
        # Fixed colors are fallbacks when no frame/field exists, never an active value.
        element["background"]["color"]["fixed"] = GRAY
        element["config"]["color"]["fixed"] = INACTIVE_TEXT
        pos = element["placement"]
        pos["top"] *= 0.80
        pos["bottom"] = 20 + pos["bottom"] * 0.80
    p["options"]["root"]["elements"].append({
        "name": "Reported state", "type": "text",
        "config": {"text": {"mode": "field", "field": name, "fixed": "No data"},
                   "size": 12, "align": "center", "valign": "middle", "color": {"fixed": INACTIVE_TEXT}},
        "background": {"color": {"fixed": "transparent"}}, "border": {"width": 0},
        "constraint": {"horizontal": "scale", "vertical": "scale"},
        "placement": {"left": 0, "right": 0, "top": 82, "bottom": 0},
    })
    p["fieldConfig"]["defaults"]["color"] = {"mode": "fixed", "fixedColor": GRAY}
    p["fieldConfig"]["overrides"] = overrides
    p["targets"] = [target(columns)]
    p["description"] = STAGED + f"Source: {name}. Highlights only a recognized reported state. " \
        "Unknown codes remain visible in the reported-value line; no state is selected for missing or unknown data. " \
        "State colors follow mission-owner display preferences."


def build_online(image_uri=None):
    base = deepcopy(build())
    original = {p["title"]: deepcopy(p) for p in base["panels"]}
    header = base["panels"][0]
    header["title"] = "SunCET • recorded test-data preview"
    next(p for p in base["panels"] if p["title"] == "Reception information pending")["title"] = "Capture & time reference"
    # Only schema-derived placeholders are used to construct the shared layout.
    # Every temporary data frame is replaced below; no capture is read.
    result = enhance_preview(base, dict.fromkeys(FIELDS, 0), "Awaiting data", image_uri)
    used = set()
    for p in result["panels"]:
        title = p["title"]
        if title == "SunCET • recorded test-data preview":
            p["title"] = "SunCET • public beacon development"
            p["options"] = original[p["title"]]["options"]
            p["options"]["content"] += "\n\n**82 public fields · development layout.** State highlights and gauges await valid received telemetry."
        elif title == "Capture & time reference":
            p["title"] = "Reception information pending"
            p["options"] = original[p["title"]]["options"]
            p["options"]["content"] = p["options"]["content"].replace(
                "No freshness or flight alarm thresholds have been assigned.",
                "Voltage color bands are visual guides; no alert rules or freshness thresholds are configured.")
        elif title == "SunCET • spacecraft":
            p["description"] = "Mission-owner supplied SunCET as-built spacecraft render, 2026-06-11."
        elif p.get("targets"):
            p["datasource"] = DATASOURCE.copy()
            if title in STATE_PANELS:
                wire_state(p, STATE_PANELS[title])
            elif p["type"] == "barchart":
                p["targets"] = [target([(f"csie_img_hist_{i}", label) for i, label in enumerate(BIN_LABELS)])]
                p["transformations"] = [
                    {"id": "filterFieldsByName", "options": {"include": {"names": BIN_LABELS}}},
                    {"id": "reduce", "options": {"mode": "seriesToRows", "reducers": ["last"], "labelsToFields": False}},
                ]
                p["options"]["xField"] = "Field"
                p["fieldConfig"]["defaults"].pop("max", None)
                p["fieldConfig"]["overrides"] = [
                    {"matcher": {"id": "byName", "options": "Field"}, "properties": [
                        {"id": "custom.axisLabel", "value": "EUV image intensity [DN]"}]},
                    {"matcher": {"id": "byName", "options": "Last"}, "properties": [
                        {"id": "displayName", "value": "Pixels"}]},
                ]
                p["description"] = STAGED + "Pixel counts from one packet, plotted directly in six transmitted bins. " \
                    "DN ranges assume offset 0 and width 32; update labels if configuration changes. " \
                    "Full image: 1504 × 2000 = 3,008,000 pixels; these six bins need not sum to that total."
            elif title in original:
                p["targets"] = original[title]["targets"]
            else:
                local_frame = json.loads(p["targets"][0]["rawFrameContent"])[0]
                names = [f["name"] for f in local_frame["schema"]["fields"]]
                p["targets"] = [target([(name, name) for name in names])]
                p["description"] = STAGED + p["description"].replace(
                    "Fixed final recorded packet; independent of the time picker. ",
                    "Latest single packet in the selected range. ")
            for t in p["targets"]:
                used.update(re.findall(r'(?:SELECT |, )"([a-z][a-z0-9_]*)"', t["query"]))
    result["description"] = base["description"] + " Updated mission-owner reviewed layout, state colors and voltage bands."
    result["time"] = {"from": "now-2d", "to": "now"}
    validate_online(result)
    assert len(used) == 82, len(used)
    return result


def validate_online(dashboard):
    text = json.dumps(dashboard)
    for forbidden in ("rawFrameContent", "csvContent", "suncet-local", "127.0.0.1", "LOCAL PREVIEW", "RECORDED TEST DATA", "Fixed final recorded packet"):
        assert forbidden not in text, forbidden
    for p in dashboard["panels"]:
        for t in p.get("targets", []):
            assert t["hide"] is True
            assert t["datasource"] == DATASOURCE and p["datasource"] == DATASOURCE
            sources = re.findall(r'(?:SELECT |, )"([a-z][a-z0-9_]*)"', t["query"])
            assert sources and set(sources) <= FIELDS.keys()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing", type=Path, help="Verified export of the online dashboard to update")
    args = parser.parse_args()
    image_uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode() if args.image else None
    d = build_online(image_uri)
    if args.existing:
        old = json.loads(args.existing.read_text())
        assert old["uid"] == "ffzqpb9arppfkb" and old["id"] == 496 and old["title"] == "suncet-dev"
        d.update(id=old["id"], uid=old["uid"], version=old["version"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d, indent=2) + "\n")
    print(f"Prepared {len(d['panels'])} panels/rows; 82 public fields; all queries disabled; no recorded samples.")


if __name__ == "__main__":
    main()
