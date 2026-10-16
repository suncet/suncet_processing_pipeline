"""Build the staged SunCET Grafana dashboard from the approved public schema.

Run with the suncet conda environment. No network access or telemetry is used.
All query targets stay disabled until the SatNOGS ingestion contract is verified.
"""

import csv
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHEMA = HERE.parent / "public_beacon_schema.csv"
FIELDS = {row["public_name"]: row for row in csv.DictReader(SCHEMA.open())}
DATASOURCE = {"type": "influxdb", "uid": "000000001"}
SPEC = "https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md"
DB = "https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975"
PANELS = []
USED = set()


def unit(name):
    value = FIELDS[name]["unit"].replace(" (inferred)", "")
    return {"V": "volt", "A": "amp", "degC": "celsius", "deg": "degree",
            "rad/s": "suffix:rad/s", "rpm": "rotrpm", "s": "s", "ms": "ms",
            "log10(XRS-B flux)": "suffix:log10(XRS-B flux)"}.get(value, "none")


def label(name):
    return name.replace("_", " ").capitalize()


def mappings(name):
    options = {}
    for code, text in re.findall(r"(\d+)/([A-Z][A-Z0-9_]*)", FIELDS[name]["conversion_or_status_map"]):
        # The public decoder emits names for known states; support either representation.
        display = {"text": f"{text} ({code})"}
        options[code] = display
        options[text] = display
    return ([{"type": "value", "options": options}] if options else []) + [
        {"type": "special", "options": {"match": "null", "result": {"text": "No data"}}},
        {"type": "special", "options": {"match": "nan", "result": {"text": "Non-finite"}}},
    ]


def add(title, kind, pos, **extra):
    panel = {"id": len(PANELS) + 1, "title": title, "type": kind,
             "gridPos": dict(zip(("x", "y", "w", "h"), pos)), **extra}
    PANELS.append(panel)
    return panel


def row(title, y):
    add(title, "row", (0, y, 24, 1), collapsed=False, panels=[])


def telemetry(title, names, pos, kind="timeseries", note=""):
    names = names.split()
    assert all(name in FIELDS for name in names), names
    USED.update(names)
    latest = kind in {"stat", "table", "bargauge"}
    columns = ", ".join(f'"{name}"' for name in names)
    # One row for each latest-value summary: no independent LAST(field) calls.
    query = f'SELECT {columns} FROM "98244" WHERE $timeFilter ORDER BY time ' + (
        "DESC LIMIT 1" if latest else "ASC")
    overrides = []
    for name in names:
        overrides.append({"matcher": {"id": "byName", "options": name}, "properties": [
            {"id": "unit", "value": unit(name)},
            {"id": "displayName", "value": label(name)},
            {"id": "mappings", "value": mappings(name)},
        ]})
    inferred = [name for name in names if "inferred" in FIELDS[name]["unit"]]
    description = "Staged CTDB 2.0.5 public engineering values; queries disabled pending ingestion verification. "
    description += note + " Unknown state codes retain their numeric value. "
    description += "Fields: " + ", ".join(names) + ". "
    if inferred:
        description += "Schema units are inferred for: " + ", ".join(inferred) + ". "
    defaults = {"unit": "none", "noValue": "No data", "mappings": [],
                "color": {"mode": "palette-classic"},
                "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}]}}
    if kind == "timeseries":
        defaults["custom"] = {"drawStyle": "points", "lineWidth": 0, "pointSize": 3,
                              "showPoints": "always", "spanNulls": False, "insertNulls": False,
                              "fillOpacity": 0, "axisPlacement": "auto", "axisLabel": "",
                              "scaleDistribution": {"type": "linear"},
                              "stacking": {"mode": "none", "group": "A"},
                              "hideFrom": {"legend": False, "tooltip": False, "viz": False}}
        options = {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True, "calcs": []},
                   "tooltip": {"mode": "multi", "sort": "none"}}
    elif kind == "table":
        options = {"showHeader": True, "cellHeight": "sm"}
    elif kind == "bargauge":
        options = {"orientation": "horizontal", "displayMode": "basic", "showUnfilled": False,
                   "reduceOptions": {"calcs": ["last"], "fields": "/^(?!Time$).*/", "values": False}}
    else:
        options = {"colorMode": "none", "graphMode": "none", "textMode": "value_and_name",
                   "orientation": "auto", "justifyMode": "auto",
                   "reduceOptions": {"calcs": ["last"], "fields": "/^(?!Time$).*/", "values": False}}
    return add(title, kind, pos, description=description.strip(), datasource=DATASOURCE,
               fieldConfig={"defaults": defaults, "overrides": overrides}, options=options,
               targets=[{"refId": "A", "datasource": DATASOURCE, "rawQuery": True,
                         "query": query, "resultFormat": "table" if latest else "time_series", "hide": True}],
               pluginVersion="11.6.0")


def pair(y, left, right):
    telemetry(left[0], left[1], (0, y, 12, 8), **(left[2] if len(left) > 2 else {}))
    telemetry(right[0], right[1], (12, y, 12, 8), **(right[2] if len(right) > 2 else {}))


def build():
    PANELS.clear()
    USED.clear()
    add("SunCET • public beacon development", "text", (0, 0, 24, 5), options={"mode": "markdown", "content":
        "## Sun Coronal Ejection Tracker\n**Development · awaiting telemetry integration**\n\n"
        "No received SunCET data is connected yet. Empty panels are expected; no synthetic telemetry is displayed. "
        "This dashboard covers the public **CCSDS APID 1 / CTDB 2.0.5** beacon.\n\n"
        f"[Mission](https://suncet.jhuapl.edu/) · [SatNOGS DB]({DB}) · [Beacon specification]({SPEC})"})
    row("Reception & coverage", 5)
    add("Reception information pending", "text", (0, 6, 24, 3), options={"mode": "markdown", "content":
        "**Last valid reception:** unavailable · **Sample age:** unavailable · **Station / observation:** unavailable\n\n"
        "Reception statistics will be connected after the SunCET source and timestamp semantics are verified. "
        "No freshness or flight alarm thresholds have been assigned."})
    row("Health & status", 9)
    telemetry("Spacecraft state", "mode_system_mode adcs_mode adcs_sun_point_state uhf_alive", (0, 10, 24, 4), "stat")
    pair(14, ("Packet identity & reset count", "ccsds_apid ccsds_sequence_count num_sc_resets", {"kind": "table", "note": "One packet per summary. A sequence gap alone does not establish packet loss or reboot."}),
         ("Spacecraft clock & elapsed time", "spacecraft_time_seconds_since_2000 spacecraft_time_milliseconds time_since_boot time_alive time_mission_elapsed_time mode_seconds_since_mode_change", {"kind": "table", "note": "Raw spacecraft clock: coarse seconds since 2000-01-01T00:00:00Z plus fine milliseconds / 1000. UTC/leap-second display policy remains unvalidated."}))
    row("EPS • batteries, buses & solar arrays", 22)
    pair(23, ("Battery voltages", "battery_1_voltage battery_2_voltage"), ("Battery charge currents", "battery_1_charge_current battery_2_charge_current"))
    pair(31, ("Bus & 3.3 V rail voltages", "eps_bus_voltage rail_3v3_voltage"), ("Bus & 3.3 V rail currents", "eps_bus_current rail_3v3_current"))
    pair(39, ("Solar array voltages", "solar_array_8_cell_string_voltage solar_array_9_cell_string_voltage"), ("Solar array currents", "solar_array_8_cell_string_current solar_array_9_cell_string_current"))
    telemetry("Battery charging states", "battery_1_charging_state battery_2_charging_state", (0, 47, 24, 4), "table", "Latest single packet; no inferred state-of-charge percentage.")
    row("ADCS • validity, pointing & motion", 51)
    pair(52, ("ADCS validity", "adcs_att_valid adcs_ref_valid adcs_time_valid", {"kind": "table", "note": "Inspect these validity flags before interpreting numerical pointing measurements."}),
         ("Sun pointing error", "adcs_sun_point_angle_error", {"note": "Interpret with ADCS validity flags; a number alone does not establish a valid pointing estimate."}))
    pair(60, ("Spacecraft body rates", "adcs_body_rate_1 adcs_body_rate_2 adcs_body_rate_3"), ("Reaction wheel speeds", "adcs_wheel_speed_1 adcs_wheel_speed_2 adcs_wheel_speed_3"))
    row("Thermal", 68)
    pair(69, ("Bus, batteries & solar arrays", "cdh_temp eps_temp ifb_therm1 battery_1_temp batt_board_temp sa_minus_y_temp sa_plus_y_temp"),
         ("Payload, radio & reaction wheel", "csie_temp dsps_sensor_board_temp uhf_temp xband_pa_temp adcs_ana_motor1_temp"))
    row("Payload • EUV imager & Dual Sun Position Sensor", 77)
    telemetry("Instrument activity & power", "csie_capture_state eps_pwr_state_csie eps_pwr_state_dsps", (0, 78, 24, 5), "table", "CSIE = Compact Spectral Imager Electronics, the primary EUV imager. Dual-SPS = Dual Sun Position Sensor, the secondary instrument. Capture-state codes remain numeric.")
    pair(83, ("Payload voltages", "csie_voltage dsps_voltage"), ("Payload currents", "csie_current dsps_current"))
    pair(91, ("Dual-SPS flare threshold", "dsps_flare_level"), ("Dual-SPS flare magnitude", "dsps_flare_magnitude", {"note": "Unsigned raw magnitude; no approved CTDB 2.0.5 engineering conversion."}))
    pair(99, ("Dual-SPS flare phase", "dsps_flare_phase", {"kind": "table"}),
         ("CSIE image histogram • six transmitted bins", "csie_img_hist_0 csie_img_hist_1 csie_img_hist_2 csie_img_hist_3 csie_img_hist_4 csie_img_hist_5", {"kind": "bargauge", "note": "Pixel counts from one packet. Bin indices 0–5; these are not the full image histogram. DN ranges depend on confirmed histogram configuration."}))
    row("Radio health", 107)
    telemetry("Radio state & data source", "uhf_alive eps_pwr_state_uhf eps_pwr_state_xband xband_data_source", (0, 108, 24, 5), "table", "These are spacecraft radio-health values, not measurements of RF reception quality.")
    pair(113, ("Radio voltages", "uhf_voltage xband_voltage"), ("Radio currents", "uhf_current xband_current xband_pa_current"))
    row("Fault-protection status", 121)
    telemetry("Task & watchpoint states", "fault_protection_task_state " + " ".join(f"fault_protection_watchpoint_{i}_state" for i in range(7)), (0, 122, 24, 6), "table", "Watchpoint numbers retain their published identifiers. ENABLED describes configuration; it does not mean a fault occurred.")
    digest = hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
    return {"id": None, "uid": None, "title": "suncet-dev", "version": 0, "schemaVersion": 41,
            "editable": True, "tags": ["SunCET", "development", "CTDB-2.0.5"],
            "description": "SunCET public APID 1 development dashboard. AEPEX-inspired section layout. "
            "All queries disabled pending verified SatNOGS ingestion. Public schema SHA-256: " + digest,
            "timezone": "utc", "time": {"from": "now-2d", "to": "now"}, "refresh": "",
            "links": [{"title": "SunCET DB", "type": "link", "url": DB, "targetBlank": True},
                      {"title": "Public beacon spec", "type": "link", "url": SPEC, "targetBlank": True}],
            "annotations": {"list": []}, "templating": {"list": []}, "panels": PANELS}


if __name__ == "__main__":
    dashboard = build()
    plan = (HERE.parents[2] / "docs" / "SATNOGS_DASHBOARD_PLAN.md").read_text()
    planned = set(re.findall(r"`([a-z][a-z0-9_]+)`", plan)) & FIELDS.keys()
    assert planned <= USED, f"Missing planned public fields: {planned - USED}"
    assert len(USED) == 81, f"Unexpected public field coverage: {len(USED)}"
    assert all(t["hide"] for p in PANELS for t in p.get("targets", []))
    (HERE / "suncet-dev.json").write_text(json.dumps(dashboard, indent=2) + "\n")
    print(f"Built {len(PANELS)} panels/rows, {len(USED)} approved fields; all queries disabled.")
