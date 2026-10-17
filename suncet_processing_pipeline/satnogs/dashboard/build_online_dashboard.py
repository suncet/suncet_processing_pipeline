"""Build the hosted SunCET dashboard without recorded telemetry or local sources.

The dashboard uses the production SatNOGS datasource and public APID 1 aliases.
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


STAGED = "CTDB 2.0.5 public engineering values. "
STATE_PANELS = {"CDH mode": "mode_system_mode", "ADCS mode": "adcs_mode",
                "Sun-pointing state": "adcs_sun_point_state", "UHF health": "uhf_alive"}
BIN_LABELS = [f"{i * 32}–{(i + 1) * 32 - 1}" for i in range(6)]
GREEN = "#378343"
RED = "#C43B49"
BLUE = "#5794F2"
RECEPTION_TITLES = {
    "Ground stations leaderboard by observed frames",
    "Last frame received by",
    "Last frame (UTC)",
    "Selected frames count",
}


def target(columns, latest=True):
    """Select one coherent packet for summaries, allowing presentation aliases."""
    assert all(source in FIELDS for source, _ in columns)
    select = ", ".join(f'"{source}"' + (f' AS "{alias}"' if alias != source else "")
                       for source, alias in columns)
    return {"refId": "A", "datasource": DATASOURCE.copy(), "rawQuery": True,
            "query": f'SELECT {select} FROM "98244" WHERE $timeFilter ORDER BY time '
                     + ("DESC LIMIT 1" if latest else "ASC"),
            "resultFormat": "table" if latest else "time_series", "hide": False}


def raw_target(query, ref_id="A", result_format="time_series"):
    return {"refId": ref_id, "datasource": DATASOURCE.copy(), "rawQuery": True,
            "query": query, "resultFormat": result_format, "hide": False}


def reception_panels(start_y, first_id):
    """Mirror the useful SPRITE/AEPEX reception summary for SunCET."""
    leaderboard = {
        "id": first_id,
        "title": "Ground stations leaderboard by observed frames",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": start_y, "w": 16, "h": 9},
        "pluginVersion": "11.6.0",
        "description": "Cumulative received SunCET frames in the selected range, grouped by SatNOGS ground station.",
        "datasource": DATASOURCE.copy(),
        "fieldConfig": {"defaults": {
            "unit": "short", "mappings": [], "color": {"mode": "palette-classic"},
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            "custom": {"axisBorderShow": False, "axisCenteredZero": False,
                       "axisColorMode": "text", "axisLabel": "", "axisPlacement": "auto",
                       "barAlignment": 0, "barWidthFactor": 0.6, "drawStyle": "line",
                       "fillOpacity": 10, "gradientMode": "none",
                       "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                       "insertNulls": False, "lineInterpolation": "linear", "lineWidth": 1,
                       "pointSize": 5, "scaleDistribution": {"type": "linear"},
                       "showPoints": "never", "spanNulls": False,
                       "stacking": {"group": "A", "mode": "none"},
                       "thresholdsStyle": {"mode": "off"}}}, "overrides": []},
        "options": {"legend": {"calcs": [], "displayMode": "list", "placement": "bottom",
                                "showLegend": True},
                    "tooltip": {"hideZeros": False, "mode": "multi", "sort": "desc"}},
        "targets": [{"alias": "$tag_station", "datasource": DATASOURCE.copy(),
                     "groupBy": [{"params": ["$__interval"], "type": "time"},
                                 {"params": ["station"], "type": "tag"}],
                     "measurement": "98244", "orderByTime": "ASC", "policy": "default",
                     "refId": "A", "resultFormat": "time_series",
                     "select": [[{"params": ["pid"], "type": "field"},
                                 {"params": [], "type": "count"},
                                 {"params": [], "type": "cumulative_sum"}]], "tags": [],
                     "hide": False}],
    }
    stat_defaults = {"mappings": [{"type": "special", "options": {
        "match": "null", "result": {"text": "N/A"}}}],
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}
    receiver = {
        "id": first_id + 1, "title": "Last frame received by", "type": "stat",
        "gridPos": {"x": 16, "y": start_y, "w": 8, "h": 3}, "pluginVersion": "11.6.0",
        "description": "Ground station credited on the latest selected SunCET frame.",
        "datasource": DATASOURCE.copy(),
        "fieldConfig": {"defaults": deepcopy(stat_defaults), "overrides": []},
        "options": {"colorMode": "value", "graphMode": "none", "justifyMode": "auto",
                    "orientation": "horizontal", "percentChangeColorMode": "standard",
                    "reduceOptions": {"calcs": ["lastNotNull"],
                                      "fields": "/^station \\(uniqueValues\\)$/", "values": False},
                    "showPercentChange": False, "textMode": "auto", "wideLayout": True},
        "targets": [raw_target('SELECT last("pid"), "observer" AS "station" FROM "98244" WHERE $timeFilter',
                               result_format="table")],
        "transformations": [{"id": "groupBy", "options": {"fields": {
            "Time": {"aggregations": [], "operation": "groupby"},
            "last": {"aggregations": [], "operation": "aggregate"},
            "station": {"aggregations": ["uniqueValues"], "operation": "aggregate"}}}}],
    }
    last_frame = {
        "id": first_id + 2, "title": "Last frame (UTC)", "type": "stat",
        "gridPos": {"x": 16, "y": start_y + 3, "w": 8, "h": 3}, "pluginVersion": "11.6.0",
        "description": "SatNOGS reception time of the latest selected SunCET frame.",
        "datasource": DATASOURCE.copy(),
        "fieldConfig": {"defaults": {**deepcopy(stat_defaults), "unit": "dateTimeAsIso"},
                        "overrides": []},
        "options": {"colorMode": "none", "graphMode": "none", "justifyMode": "auto",
                    "orientation": "horizontal", "percentChangeColorMode": "standard",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "/^Time$/", "values": False},
                    "showPercentChange": False, "textMode": "auto", "wideLayout": True},
        "targets": [{"datasource": DATASOURCE.copy(), "groupBy": [], "measurement": "98244",
                     "orderByTime": "ASC", "policy": "default", "refId": "A",
                     "resultFormat": "time_series", "select": [[
                         {"params": ["pid"], "type": "field"},
                         {"params": [], "type": "last"}]], "tags": [], "hide": False}],
    }
    count = {
        "id": first_id + 3, "title": "Selected frames count", "type": "stat",
        "gridPos": {"x": 16, "y": start_y + 6, "w": 8, "h": 3}, "pluginVersion": "11.6.0",
        "description": "Number of received SunCET frames in the selected time range.",
        "datasource": DATASOURCE.copy(),
        "fieldConfig": {"defaults": {**deepcopy(stat_defaults), "unit": "none",
                                     "color": {"fixedColor": BLUE, "mode": "fixed"}},
                        "overrides": []},
        "options": {"colorMode": "none", "graphMode": "area", "justifyMode": "auto",
                    "orientation": "horizontal", "percentChangeColorMode": "standard",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "showPercentChange": False, "textMode": "auto", "wideLayout": True},
        "targets": [{"datasource": DATASOURCE.copy(),
                     "groupBy": [{"params": ["$__interval"], "type": "time"}],
                     "measurement": "98244", "orderByTime": "ASC", "policy": "default",
                     "refId": "A", "resultFormat": "table", "select": [[
                         {"params": ["pid"], "type": "field"},
                         {"params": [], "type": "count"},
                         {"params": [], "type": "cumulative_sum"}]], "tags": [], "hide": False}],
    }
    return [leaderboard, receiver, last_frame, count]


def styled_series_panel(template, panel_id, title, grid_pos, targets, aliases, description):
    panel = deepcopy(template)
    panel.update(id=panel_id, title=title, gridPos=grid_pos, targets=targets,
                 description=description)
    panel["fieldConfig"]["defaults"].update(unit="watt", noValue="No data")
    panel["fieldConfig"]["defaults"]["custom"].update(
        drawStyle="line", lineWidth=2, pointSize=3, showPoints="auto")
    panel["fieldConfig"]["overrides"] = []
    for alias, display, color in aliases:
        properties = [{"id": "unit", "value": "watt"},
                      {"id": "displayName", "value": display}]
        if color:
            properties.append({"id": "color", "value": {"mode": "fixed", "fixedColor": color}})
        panel["fieldConfig"]["overrides"].append(
            {"matcher": {"id": "byName", "options": alias}, "properties": properties})
    return panel


def configure_battery_currents(panel):
    panel["description"] = ("Battery charge-current magnitude, colored by the corresponding charging-state bit. "
                            "Green means charging; red means discharging. The state is evaluated from the same packet.")
    panel["targets"] = [
        raw_target('SELECT "battery_1_charge_current" AS "Battery 1 charging" FROM "98244" '
                   'WHERE $timeFilter AND "battery_1_charging_state" = 1 ORDER BY time ASC', "A"),
        raw_target('SELECT "battery_1_charge_current" AS "Battery 1 discharging" FROM "98244" '
                   'WHERE $timeFilter AND "battery_1_charging_state" = 0 ORDER BY time ASC', "B"),
        raw_target('SELECT "battery_2_charge_current" AS "Battery 2 charging" FROM "98244" '
                   'WHERE $timeFilter AND "battery_2_charging_state" = 1 ORDER BY time ASC', "C"),
        raw_target('SELECT "battery_2_charge_current" AS "Battery 2 discharging" FROM "98244" '
                   'WHERE $timeFilter AND "battery_2_charging_state" = 0 ORDER BY time ASC', "D"),
    ]
    panel["fieldConfig"]["defaults"]["custom"].update(
        drawStyle="line", lineWidth=2, pointSize=3, showPoints="auto")
    panel["fieldConfig"]["overrides"] = []
    for alias, color in (("Battery 1 charging", GREEN), ("Battery 1 discharging", RED),
                         ("Battery 2 charging", GREEN), ("Battery 2 discharging", RED)):
        panel["fieldConfig"]["overrides"].append({
            "matcher": {"id": "byName", "options": alias},
            "properties": [{"id": "unit", "value": "amp"},
                           {"id": "displayName", "value": alias},
                           {"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
        })


def arrange_power_voltage_current_rows(panels):
    """Put derived power beside its source voltage and current panels."""
    by_title = {panel["title"]: panel for panel in panels}

    # The three EPS power panels initially occupy their own row. Reclaim that
    # row, then lead each related row with power so the derivation is visually
    # adjacent to its voltage and current inputs.
    eps_power_floor = by_title["Battery power"]["gridPos"]["y"]
    eps_power_height = by_title["Battery power"]["gridPos"]["h"]
    for panel in panels:
        if panel["gridPos"]["y"] >= eps_power_floor + eps_power_height:
            panel["gridPos"]["y"] -= eps_power_height
    eps_rows = (
        ("Battery power", "Battery voltages", "Battery charge currents"),
        ("EPS bus power", "Bus & 3.3 V rail voltages", "Bus & 3.3 V rail currents"),
        ("Solar array power", "Solar array voltages", "Solar array currents"),
    )
    first_eps_y = by_title["Battery voltages"]["gridPos"]["y"]
    for row_index, titles in enumerate(eps_rows):
        for column, title in enumerate(titles):
            by_title[title]["gridPos"].update(
                x=column * 8, y=first_eps_y + row_index * 8, w=8, h=8)

    # Payload and radio power were also appended as full-width rows. Collapse
    # each into the first column of its voltage/current row.
    for power_title, voltage_title, current_title in (
        ("Payload power", "Payload voltages", "Payload currents"),
        ("Radio power", "Radio voltages", "Radio currents"),
    ):
        power = by_title[power_title]
        voltage = by_title[voltage_title]
        current = by_title[current_title]
        old_power_y = power["gridPos"]["y"]
        old_power_height = power["gridPos"]["h"]
        for panel in panels:
            if panel["gridPos"]["y"] >= old_power_y + old_power_height:
                panel["gridPos"]["y"] -= old_power_height
        row_y = voltage["gridPos"]["y"]
        power["gridPos"].update(x=0, y=row_y, w=8, h=8)
        voltage["gridPos"].update(x=8, y=row_y, w=8, h=8)
        current["gridPos"].update(x=16, y=row_y, w=8, h=8)


def apply_operational_layout(result):
    """Enable hosted queries and add live reception and derived-power panels."""
    panels = result["panels"]
    by_title = {p["title"]: p for p in panels}
    header = by_title["SunCET • public beacon development"]
    header["title"] = "SunCET • public beacon"
    header["options"]["content"] = (
        "## Sun Coronal Ejection Tracker\n\n"
        "[Mission](https://suncet.jhuapl.edu/) · "
        "[SatNOGS DB](https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975) · "
        "[Beacon specification](https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md)")

    pending = by_title["Reception information pending"]
    pending_y = pending["gridPos"]["y"]
    pending_id = pending["id"]
    panels.remove(pending)
    for panel in panels:
        if panel["gridPos"]["y"] >= pending_y + 5:
            panel["gridPos"]["y"] += 4
    next_id = max(p["id"] for p in panels) + 1
    reception = reception_panels(pending_y, next_id)
    reception[0]["id"] = pending_id
    panels.extend(reception)
    next_id = max(p["id"] for p in panels) + 1

    by_title = {p["title"]: p for p in panels}
    configure_battery_currents(by_title["Battery charge currents"])
    charge_state = by_title["Battery charging states"]
    power_y = charge_state["gridPos"]["y"]
    panels.remove(charge_state)
    for panel in panels:
        if panel["gridPos"]["y"] >= power_y + charge_state["gridPos"]["h"]:
            panel["gridPos"]["y"] += 4
    template = by_title["Battery charge currents"]
    battery_power = styled_series_panel(
        template, next_id, "Battery power", {"x": 0, "y": power_y, "w": 8, "h": 8}, [
            raw_target('SELECT "battery_1_voltage" * "battery_1_charge_current" AS "Battery 1 charging power" '
                       'FROM "98244" WHERE $timeFilter AND "battery_1_charging_state" = 1 ORDER BY time ASC', "A"),
            raw_target('SELECT -1 * "battery_1_voltage" * "battery_1_charge_current" AS "Battery 1 discharging power" '
                       'FROM "98244" WHERE $timeFilter AND "battery_1_charging_state" = 0 ORDER BY time ASC', "B"),
            raw_target('SELECT "battery_2_voltage" * "battery_2_charge_current" AS "Battery 2 charging power" '
                       'FROM "98244" WHERE $timeFilter AND "battery_2_charging_state" = 1 ORDER BY time ASC', "C"),
            raw_target('SELECT -1 * "battery_2_voltage" * "battery_2_charge_current" AS "Battery 2 discharging power" '
                       'FROM "98244" WHERE $timeFilter AND "battery_2_charging_state" = 0 ORDER BY time ASC', "D"),
        ], [("Battery 1 charging power", "Battery 1 charging", GREEN),
            ("Battery 1 discharging power", "Battery 1 discharging", RED),
            ("Battery 2 charging power", "Battery 2 charging", GREEN),
            ("Battery 2 discharging power", "Battery 2 discharging", RED)],
        "Derived electrical power. Positive/green is charging; negative/red is discharging. "
        "Voltage, current, and charging state come from the same beacon packet.")
    next_id += 1
    bus_power = styled_series_panel(
        template, next_id, "EPS bus power", {"x": 8, "y": power_y, "w": 8, "h": 8},
        [raw_target('SELECT "eps_bus_voltage" * "eps_bus_current" AS "EPS bus power" FROM "98244" '
                    'WHERE $timeFilter ORDER BY time ASC')],
        [("EPS bus power", "EPS bus", None)],
        "Derived from EPS bus voltage × EPS bus current in each beacon packet.")
    next_id += 1
    solar_power = styled_series_panel(
        template, next_id, "Solar array power", {"x": 16, "y": power_y, "w": 8, "h": 8},
        [raw_target('SELECT "solar_array_8_cell_string_voltage" * "solar_array_8_cell_string_current" '
                    'AS "8-cell array power", "solar_array_9_cell_string_voltage" * '
                    '"solar_array_9_cell_string_current" AS "9-cell array power" FROM "98244" '
                    'WHERE $timeFilter ORDER BY time ASC')],
        [("8-cell array power", "8-cell string", None),
         ("9-cell array power", "9-cell string", None)],
        "Derived from the voltage × current pair for each solar-array string in each beacon packet.")
    next_id += 1
    panels.extend([battery_power, bus_power, solar_power])

    by_title = {p["title"]: p for p in panels}
    payload_current = by_title["Payload currents"]
    payload_power_y = payload_current["gridPos"]["y"] + payload_current["gridPos"]["h"]
    for panel in panels:
        if panel["gridPos"]["y"] >= payload_power_y:
            panel["gridPos"]["y"] += 8
    payload_power = styled_series_panel(
        template, next_id, "Payload power", {"x": 0, "y": payload_power_y, "w": 24, "h": 8},
        [raw_target('SELECT "csie_voltage" * "csie_current" AS "CSIE power", '
                    '"dsps_voltage" * "dsps_current" AS "Dual-SPS power" FROM "98244" '
                    'WHERE $timeFilter ORDER BY time ASC')],
        [("CSIE power", "CSIE", None), ("Dual-SPS power", "Dual-SPS", None)],
        "Derived from the voltage × current pair for each payload in each beacon packet.")
    next_id += 1
    panels.append(payload_power)

    by_title = {p["title"]: p for p in panels}
    radio_current = by_title["Radio currents"]
    radio_power_y = radio_current["gridPos"]["y"] + radio_current["gridPos"]["h"]
    for panel in panels:
        if panel["gridPos"]["y"] >= radio_power_y:
            panel["gridPos"]["y"] += 8
    radio_power = styled_series_panel(
        template, next_id, "Radio power", {"x": 0, "y": radio_power_y, "w": 24, "h": 8},
        [raw_target('SELECT "uhf_voltage" * "uhf_current" AS "UHF power", '
                    '"xband_voltage" * "xband_current" AS "X-band electronics power", '
                    '"xband_voltage" * "xband_pa_current" AS "X-band PA power" FROM "98244" '
                    'WHERE $timeFilter ORDER BY time ASC')],
        [("UHF power", "UHF", None),
         ("X-band electronics power", "X-band electronics", None),
         ("X-band PA power", "X-band PA", None)],
        "Derived electrical input power. The X-band PA series uses the reported X-band voltage × PA current.")
    panels.append(radio_power)
    arrange_power_voltage_current_rows(panels)
    panels.sort(key=lambda p: (p["gridPos"]["y"], p["gridPos"]["x"]))

    for panel in panels:
        if isinstance(panel.get("description"), str):
            panel["description"] = panel["description"].replace(
                "Staged CTDB 2.0.5 public engineering values; queries disabled pending ingestion verification. ",
                STAGED)
        for query in panel.get("targets", []):
            query["hide"] = False
    result.update(title="SunCET", tags=["SunCET", "CTDB-2.0.5", "public-beacon"],
                  description="SunCET public APID 1 telemetry dashboard.", refresh="1m")
    return result


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
    apply_operational_layout(result)
    validate_online(result)
    assert len(used) == 82, len(used)
    return result


def validate_online(dashboard):
    text = json.dumps(dashboard)
    for forbidden in ("rawFrameContent", "csvContent", "suncet-local", "127.0.0.1", "LOCAL PREVIEW",
                      "RECORDED TEST DATA", "Fixed final recorded packet", "queries disabled",
                      "No received SunCET data is connected yet", "82 public fields · development layout"):
        assert forbidden not in text, forbidden
    for p in dashboard["panels"]:
        for t in p.get("targets", []):
            assert t["hide"] is False
            assert t["datasource"] == DATASOURCE and p["datasource"] == DATASOURCE
            if p["title"] in RECEPTION_TITLES:
                continue
            sources = re.findall(r'"([a-z][a-z0-9_]*)"', t["query"])
            assert sources and set(sources) <= FIELDS.keys()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing", type=Path, help="Verified export of the online dashboard to update")
    args = parser.parse_args()
    image_uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode() if args.image else None
    old = None
    if args.existing:
        old = json.loads(args.existing.read_text())
        assert old["uid"] == "ffzqpb9arppfkb" and old["id"] == 496
        assert old["title"] in {"suncet-dev", "SunCET"}
        if image_uri is None:
            spacecraft = next((p for p in old["panels"] if p["title"] == "SunCET • spacecraft"), None)
            if spacecraft:
                match = re.search(r'<img src="(data:image/png;base64,[^"]+)"',
                                  spacecraft.get("options", {}).get("content", ""))
                image_uri = match.group(1) if match else None
    d = build_online(image_uri)
    if old:
        d.update(id=old["id"], uid=old["uid"], version=old["version"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d, indent=2) + "\n")
    print(f"Prepared {len(d['panels'])} panels/rows; 82 public fields; hosted queries enabled; no recorded samples.")


if __name__ == "__main__":
    main()
