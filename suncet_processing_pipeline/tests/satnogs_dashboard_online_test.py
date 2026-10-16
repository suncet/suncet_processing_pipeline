"""Guard the hosted layout's public-data boundary and state semantics."""

import json
import pytest

from suncet_processing_pipeline.satnogs.dashboard.build_online_dashboard import (
    build_online, validate_online, STATE_PANELS,
)
from suncet_processing_pipeline.satnogs.dashboard.preview_panels import GRAY, STATE_COLORS, STATE_RULES


def test_online_template_has_no_recorded_data_and_preserves_missing_data():
    d = build_online()
    validate_online(d)
    assert sum(bool(p.get("targets")) for p in d["panels"]) == 38
    assert d["time"] == {"from": "now-2d", "to": "now"}
    assert "2026-09-23" not in json.dumps(d)
    for p in d["panels"]:
        if p["type"] == "gauge":
            assert p["fieldConfig"]["defaults"]["max"] == 16
        if p["type"] == "canvas":
            for elem in p["options"]["root"]["elements"][:-1]:
                assert elem["background"]["color"]["fixed"] == GRAY
            assert p["options"]["root"]["elements"][-1]["config"]["text"]["fixed"] == "No data"


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
