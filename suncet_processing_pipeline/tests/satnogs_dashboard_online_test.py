"""Guard the hosted layout's public-data boundary and state semantics."""

import json
import pytest

from suncet_processing_pipeline.satnogs.dashboard.build_online_dashboard import (
    build_online, validate_online, STATE_PANELS, RECEPTION_TITLES, GREEN, RED,
)
from suncet_processing_pipeline.satnogs.dashboard.preview_panels import GRAY, STATE_COLORS, STATE_RULES


def test_online_template_has_no_recorded_data_and_preserves_missing_data():
    d = build_online()
    validate_online(d)
    assert sum(bool(p.get("targets")) for p in d["panels"]) == 46
    assert d["time"] == {"from": "now-2d", "to": "now"}
    assert d["title"] == "SunCET"
    assert d["refresh"] == "1m"
    assert "2026-09-23" not in json.dumps(d)
    assert all(target["hide"] is False for p in d["panels"] for target in p.get("targets", []))
    for p in d["panels"]:
        if p["type"] == "gauge":
            assert p["fieldConfig"]["defaults"]["max"] == 16
        if p["type"] == "canvas":
            for elem in p["options"]["root"]["elements"][:-1]:
                assert elem["background"]["color"]["fixed"] == GRAY
            assert p["options"]["root"]["elements"][-1]["config"]["text"]["fixed"] == "No data"


def test_online_reception_summary_matches_satnogs_pattern():
    d = build_online()
    by_title = {p["title"]: p for p in d["panels"]}
    assert RECEPTION_TITLES <= by_title.keys()
    leaderboard = by_title["Ground stations leaderboard by observed frames"]
    assert leaderboard["targets"][0]["measurement"] == "98244"
    assert leaderboard["targets"][0]["groupBy"][1]["params"] == ["station"]
    assert '"observer" AS "station"' in by_title["Last frame received by"]["targets"][0]["query"]


def test_battery_current_state_is_encoded_in_series_color_and_state_table_is_removed():
    d = build_online()
    by_title = {p["title"]: p for p in d["panels"]}
    assert "Battery charging states" not in by_title
    panel = by_title["Battery charge currents"]
    assert len(panel["targets"]) == 4
    assert any('"battery_1_charging_state" = 1' in t["query"] for t in panel["targets"])
    assert any('"battery_2_charging_state" = 0' in t["query"] for t in panel["targets"])
    colors = {override["matcher"]["options"]:
              next(prop["value"]["fixedColor"] for prop in override["properties"]
                   if prop["id"] == "color")
              for override in panel["fieldConfig"]["overrides"]}
    assert colors["Battery 1 charging"] == GREEN
    assert colors["Battery 1 discharging"] == RED


def test_derived_power_panels_use_packet_coherent_voltage_current_pairs():
    d = build_online()
    by_title = {p["title"]: p for p in d["panels"]}
    expected = {"Battery power", "EPS bus power", "Solar array power", "Payload power", "Radio power"}
    assert expected <= by_title.keys()
    for title in expected:
        assert by_title[title]["fieldConfig"]["defaults"]["unit"] == "watt"
        assert all('FROM "98244" WHERE $timeFilter' in t["query"] for t in by_title[title]["targets"])
    assert '"eps_bus_voltage" * "eps_bus_current"' in by_title["EPS bus power"]["targets"][0]["query"]
    assert '"xband_voltage" * "xband_pa_current"' in by_title["Radio power"]["targets"][0]["query"]


def test_power_panels_lead_their_voltage_and_current_rows():
    d = build_online()
    by_title = {p["title"]: p for p in d["panels"]}
    for titles in (
        ("Battery power", "Battery voltages", "Battery charge currents"),
        ("EPS bus power", "Bus & 3.3 V rail voltages", "Bus & 3.3 V rail currents"),
        ("Solar array power", "Solar array voltages", "Solar array currents"),
        ("Payload power", "Payload voltages", "Payload currents"),
        ("Radio power", "Radio voltages", "Radio currents"),
    ):
        panels = [by_title[title] for title in titles]
        assert [p["gridPos"]["x"] for p in panels] == [0, 8, 16]
        assert len({p["gridPos"]["y"] for p in panels}) == 1
        assert [p["gridPos"]["w"] for p in panels] == [8, 8, 8]


@pytest.mark.parametrize("title,name", STATE_PANELS.items())
def test_online_state_mapping_uses_source_enum_not_local_boolean(title, name):
    p = next(p for p in build_online()["panels"] if p["title"] == title)
    options = {}
    for override in p["fieldConfig"]["overrides"]:
        label = override["matcher"]["options"]
        if label in STATE_RULES[name]:
            options[label] = next(prop["value"][0]["options"] for prop in override["properties"] if prop["id"] == "mappings")
    for value in STATE_RULES[name]:
        for label, mapping in options.items():
            expected = STATE_COLORS[STATE_RULES[name][label]] if label == value else GRAY
            assert mapping[value]["color"] == expected
    assert all('"'+name+'" AS "'+label+'"' in p["targets"][0]["query"] for label in options)
