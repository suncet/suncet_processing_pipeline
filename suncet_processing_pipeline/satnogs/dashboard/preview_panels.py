"""Native Grafana panel styling, initially built for the recorded preview.

Derived state indicators are presentation values, not new telemetry fields.
The online builder reuses the layout and replaces all recorded data sources.
"""

from copy import deepcopy
import json
import math
import re

from .build_dashboard import FIELDS


SOURCE = {"type": "grafana-testdata-datasource", "uid": "suncet-local-csv"}
BLUE = "#5794F2"
GRAY = "#282D35"
INACTIVE_TEXT = "#969DA8"
STATE_COLORS = {"red": "#C43B49", "yellow": "#E6C84F", "green": "#378343"}
STATE_RULES = {
    "mode_system_mode": {"PHOENIX": "red", "SAFE": "yellow", "DOWNLINK": "green", "SCIENCE": "green"},
    "adcs_mode": {"SUN_POINT": "yellow", "FINE_REF_POINT": "green"},
    "adcs_sun_point_state": {
        "FINE_REF_POINT": "green", "ON_SUN": "green", "SUN_POINT": "yellow",
        "CONVERGING": "yellow", "SEARCHING": "yellow", "SEARCH_INIT": "red",
        "NOT_ACTV": "red", "WAITING": "red",
    },
    "uhf_alive": {"OFF": "red", "ALIVE": "green", "DEAD": "red"},
}
LATEST = "Fixed final recorded packet; independent of the time picker. "


def frame(columns):
    """Build typed presentation columns: (name, type, values)."""
    return json.dumps([{
        "schema": {"name": "SunCET recorded preview", "refId": "A", "fields": [
            {"name": name, "type": kind, "config": {}} for name, kind, _ in columns]},
        "data": {"values": [values for _, _, values in columns]},
    }], allow_nan=False)


def defaults():
    return {"unit": "none", "noValue": "No data", "mappings": [],
            "color": {"mode": "fixed", "fixedColor": BLUE},
            "thresholds": {"mode": "absolute", "steps": [{"color": BLUE, "value": None}]}}


def panel(title, kind, pos, columns, note):
    return {"title": title, "type": kind, "gridPos": dict(zip(("x", "y", "w", "h"), pos)),
            "pluginVersion": "11.6.0", "description": LATEST + note,
            "datasource": SOURCE.copy(), "fieldConfig": {"defaults": defaults(), "overrides": []},
            "targets": [{"refId": "A", "datasource": SOURCE.copy(), "hide": False,
                         "scenarioId": "raw_frame", "rawFrameContent": frame(columns)}]}


def stat_options():
    return {"colorMode": "background_solid", "graphMode": "none", "textMode": "value",
            "orientation": "auto", "justifyMode": "center", "wideLayout": True,
            "reduceOptions": {"calcs": ["last"], "fields": "/.*/", "values": False}}


def states(title, name, value, pos):
    """Show every approved state, highlighting only an exact code/name match."""
    choices = re.findall(r"(\d+)/([A-Z][A-Z0-9_]*)", FIELDS[name]["conversion_or_status_map"])
    rules = STATE_RULES[name]
    order = list(rules)
    choices.sort(key=lambda choice: order.index(choice[1]) if choice[1] in order else len(order))
    known = any(value in (text, int(code), code) for code, text in choices)
    columns = [(text, "number", [int(value in (text, int(code), code))]) for code, text in choices]
    if not known:
        columns.append(("UNKNOWN", "number", [1]))
    # Separate color fields let native Canvas bind both background and text
    # to the reported state while keeping inactive text at lower contrast.
    color_columns = [(label + "_text_color", kind, selected) for label, kind, selected in columns]
    result = panel(title, "canvas", pos, columns + color_columns,
                   f"Source: {name}. Active colors follow mission-owner display preferences. "
                   "Inactive choices have dimmed text. " + ("" if known else f"Unrecognized value: {value!r}."))
    elements = []
    ncols = min(4, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    for i, (label, _, _) in enumerate(columns):
        display = label.replace("_", " ") if label != "UNKNOWN" else f"UNKNOWN ({value})"
        color_name = rules.get(label, "red")
        selected_color = STATE_COLORS[color_name]
        result["fieldConfig"]["overrides"].append({
            "matcher": {"id": "byName", "options": label}, "properties": [{
                "id": "mappings", "value": [{"type": "value", "options": {
                    "0": {"text": display, "color": GRAY},
                    "1": {"text": display, "color": selected_color},
                }}]}]})
        result["fieldConfig"]["overrides"].append({
            "matcher": {"id": "byName", "options": label + "_text_color"}, "properties": [{
                "id": "mappings", "value": [{"type": "value", "options": {
                    "0": {"color": INACTIVE_TEXT},
                    "1": {"color": "#17191D" if color_name == "yellow" else "#FFFFFF"},
                }}]}]})
        row_index, col_index = divmod(i, ncols)
        elements.append({
            "name": label, "type": "text",
            "config": {"text": {"mode": "fixed", "fixed": display}, "align": "center", "valign": "middle",
                       "size": 16, "color": {"field": label + "_text_color"}},
            "background": {"color": {"field": label}},
            "border": {"width": 0, "radius": 3},
            "constraint": {"horizontal": "scale", "vertical": "scale"},
            "placement": {"left": col_index * 100 / ncols + 0.4,
                          "right": 100 - (col_index + 1) * 100 / ncols + 0.4,
                          "top": row_index * 100 / nrows + 2,
                          "bottom": 100 - (row_index + 1) * 100 / nrows + 2},
        })
    result["options"] = {"inlineEditing": False, "panZoom": False, "infinitePan": False,
                         "showAdvancedTypes": False,
                         "root": {"name": "State choices", "type": "frame", "elements": elements}}
    return result


def power(title, name, value, pos):
    result = panel(title, "stat", pos, [(name, "string", [str(value)])],
                   f"Source: {name}. EPS power-enable flag, not subsystem health.")
    result["options"] = stat_options()
    result["fieldConfig"]["defaults"]["mappings"] = [{"type": "value", "options": {
        "ON": {"text": "ENABLED", "color": "green"}, "1": {"text": "ENABLED", "color": "green"},
        "OFF": {"text": "DISABLED", "color": "dark-red"}, "0": {"text": "DISABLED", "color": "dark-red"},
    }}]
    return result


def gauge(title, name, value, pos):
    result = panel(title, "gauge", pos, [(name, "number", [value])],
                   f"Source: {name}. Display scale 12–16 V. Mission-owner display bands: "
                   "red below 13.2 V, yellow from 13.2 to 14 V, green at 14 V and above. "
                   "Unit inferred in CTDB public schema; engineering conversion applied once.")
    result["fieldConfig"]["defaults"].update({"unit": "volt", "min": 12, "max": 16, "decimals": 2})
    result["fieldConfig"]["defaults"].update({"color": {"mode": "thresholds"},
        "thresholds": {"mode": "absolute", "steps": [
            {"color": "red", "value": None}, {"color": "yellow", "value": 13.2},
            {"color": "green", "value": 14}]}})
    result["options"] = {"orientation": "auto", "showThresholdLabels": True,
                         "showThresholdMarkers": True, "sizing": "auto",
                         "reduceOptions": {"calcs": ["last"], "fields": "", "values": False}}
    return result


def histogram(values, pos):
    ranges = [f"{i * 32}–{(i + 1) * 32 - 1}" for i in range(6)]
    counts = [values[f"csie_img_hist_{i}"] for i in range(6)]
    result = panel("CSIE image histogram • default DN bins", "barchart", pos,
                   [("Pixel DN (offset 0, width 32)", "string", ranges), ("Pixels", "number", counts)],
                   "Transmitted pixel counts plotted directly, not re-binned. Only bins 0–5 are transmitted. "
                   "DN labels assume the documented defaults: offset 0, width 32. "
                   "This configuration is not carried in the beacon; update labels if it changes. "
                   "Full image: 1504 × 2000 = 3,008,000 pixels; six transmitted bins need not sum to that total.")
    if not any(counts):
        result["description"] += " All six counts are zero in this recorded packet."
    result["fieldConfig"]["defaults"].update({"min": 0, "decimals": 0, "custom": {
        "axisPlacement": "left", "axisLabel": "Pixel count (3,008,000 total)", "axisCenteredZero": False,
        "fillOpacity": 90, "lineWidth": 1, "gradientMode": "none", "hideFrom": {
            "legend": False, "tooltip": False, "viz": False}, "scaleDistribution": {"type": "linear"}}})
    result["fieldConfig"]["overrides"] = [{
        "matcher": {"id": "byName", "options": "Pixel DN (offset 0, width 32)"},
        "properties": [{"id": "custom.axisLabel", "value": "EUV image intensity [DN]"}],
    }]
    # A zero-valued capture still needs a visible, stable axis.
    if not any(counts):
        result["fieldConfig"]["defaults"]["max"] = 1
    result["options"] = {"orientation": "vertical", "xField": "Pixel DN (offset 0, width 32)",
                         "xTickLabelRotation": 0, "xTickLabelSpacing": 0, "showValue": "always",
                         "stacking": "none", "groupWidth": 1, "barWidth": 1, "barRadius": 0,
                         "legend": {"showLegend": False, "displayMode": "list", "placement": "bottom"},
                         "tooltip": {"mode": "single", "sort": "none"}}
    return result


def enhance_preview(dashboard, values, timestamp, image_uri=None, histogram_example=None):
    """Extend a saved or freshly generated local dashboard; retain other settings."""
    result = deepcopy(dashboard)
    panels = result["panels"]
    assert not any(p["title"] == "Subsystem power states" for p in panels), "Already enhanced"
    by_title = {p["title"]: p for p in panels}
    pointing = by_title["Sun pointing error"]
    pointing["fieldConfig"]["defaults"].update(min=0, max=10)
    pointing["options"]["legend"].update(displayMode="table", calcs=["max"])
    pointing["description"] += " Display axis: 0–10°. The legend shows the actual maximum in the selected range, including values above 10°; samples are not clipped or discarded."
    next_id = max(p["id"] for p in panels) + 1

    def append(p):
        nonlocal next_id
        p["id"] = next_id
        next_id += 1
        panels.append(p)

    # Make room while retaining the existing time-series styling and queries.
    for p in panels:
        y = p["gridPos"]["y"]
        if y >= 23:
            p["gridPos"]["y"] += 14
        elif y >= 22:
            p["gridPos"]["y"] = 29
        elif y >= 14:
            p["gridPos"].update(y=20, h=5)
        elif y >= 5:
            p["gridPos"]["y"] += 3
    intro = by_title["SunCET • recorded test-data preview"]
    intro["gridPos"].update(w=16 if image_uri else 24, h=8)
    intro["options"]["content"] = intro["options"]["content"].replace("81 approved", "82 approved")
    intro["options"]["content"] += f"\n\n**Summary packet:** {timestamp}. Gauges, states and histogram use this same packet."
    if image_uri:
        append({"title": "SunCET • spacecraft", "type": "text", "gridPos": {"x": 16, "y": 0, "w": 8, "h": 8},
                "description": "User-supplied as-built spacecraft render, 2026-06-11. Embedded locally.",
                "options": {"mode": "html", "content":
                    f'<img src="{image_uri}" alt="SunCET spacecraft with deployed solar arrays" '
                    'style="max-width:100%;max-height:220px;width:auto;height:auto;display:block;margin:0 auto" />'}})
    old_state = by_title["Spacecraft state"]
    mode = states("CDH mode", "mode_system_mode", values["mode_system_mode"], (0, 13, 12, 3))
    mode["id"] = old_state["id"]
    panels[panels.index(old_state)] = mode
    append(states("ADCS mode", "adcs_mode", values["adcs_mode"], (12, 13, 12, 3)))
    append(states("Sun-pointing state", "adcs_sun_point_state", values["adcs_sun_point_state"], (0, 16, 18, 4)))
    append(states("UHF health", "uhf_alive", values["uhf_alive"], (18, 16, 6, 4)))
    append({"title": "Subsystem power states", "type": "row", "gridPos": {"x": 0, "y": 25, "w": 24, "h": 1},
            "collapsed": False, "panels": []})
    for i, (title, name) in enumerate([
        ("ADCS / XACT", "eps_pwr_state_adcs"), ("CSIE • EUV imager", "eps_pwr_state_csie"),
        ("Dual-SPS", "eps_pwr_state_dsps"), ("UHF radio", "eps_pwr_state_uhf"), ("X-band radio", "eps_pwr_state_xband"),
    ]):
        append(power(title, name, values[name], (i * 5, 26, 4 if i == 4 else 5, 3)))
    for i, (title, name) in enumerate([
        ("Battery 1 voltage", "battery_1_voltage"), ("Battery 2 voltage", "battery_2_voltage"),
        ("EPS bus voltage", "eps_bus_voltage"),
    ]):
        append(gauge(title, name, values[name], (i * 8, 30, 8, 7)))
    old_hist = by_title["CSIE image histogram • six transmitted bins"]
    new_hist = histogram(values, tuple(old_hist["gridPos"][k] for k in ("x", "y", "w", "h")))
    new_hist["id"] = old_hist["id"]
    panels[panels.index(old_hist)] = new_hist
    if histogram_example:
        sample_time, sample_values = histogram_example
        insert_y = by_title["Radio health"]["gridPos"]["y"]
        for p in panels:
            if p["gridPos"]["y"] >= insert_y:
                p["gridPos"]["y"] += 8
        example = histogram(sample_values, (0, insert_y, 24, 8))
        example["title"] = f"CSIE histogram • earlier recorded packet {sample_time}"
        example["description"] = example["description"].replace(
            LATEST, "Most recent packet with nonzero histogram counts, selected as a recorded example. "
            f"Reported spacecraft time: {sample_time}. Independent of the time picker. "
            "The final packet's zero counts are shown separately above. ")
        append(example)
        intro["options"]["content"] += " The additional CSIE histogram example is explicitly dated to an earlier packet."
    # Leave room for readable labels and notes when the browser pane is narrow.
    for p in panels:
        if p["title"] in {"Sun-pointing state", "UHF health"}:
            p["gridPos"]["h"] = 6
        elif p["gridPos"]["y"] >= 20:
            p["gridPos"]["y"] += 2
    for p in panels:
        if p["title"] == "Capture & time reference":
            p["gridPos"]["h"] = 5
        elif p["gridPos"]["y"] >= 12:
            p["gridPos"]["y"] += 2
    panels.sort(key=lambda p: (p["gridPos"]["y"], p["gridPos"]["x"]))
    return result
