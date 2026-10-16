"""Verify the local preview's filtering and typed public-field boundary."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct

import pytest

from suncet_processing_pipeline.satnogs.beacon_contract import suncet_fletcher32
from suncet_processing_pipeline.satnogs.dashboard.build_local_preview import (
    capture_rows,
    frame_values,
)
from suncet_processing_pipeline.satnogs.dashboard.preview_panels import histogram, states


def test_preview_deduplicates_and_excludes_invalid_and_unset_clock_packets(tmp_path):
    packet = bytearray(bytes.fromhex(
        (Path(__file__).parents[1] / "satnogs/test_data/suncet_apid1_synthetic_252.hex")
        .read_text()
    ))
    start = datetime(2026, 9, 23, tzinfo=timezone.utc)
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    struct.pack_into(">IH", packet, 6, int((start - epoch).total_seconds()) + 100, 123)
    packet[-4:] = suncet_fletcher32(packet[:-4]).to_bytes(4, "big")
    unset = bytearray(packet)
    unset[6:10] = (10).to_bytes(4, "big")
    unset[-4:] = suncet_fletcher32(unset[:-4]).to_bytes(4, "big")
    invalid = bytearray(packet)
    invalid[100] ^= 1
    run = tmp_path / "run"
    run.mkdir()
    path = run / "ccsds_example"
    original = bytes(packet + packet + unset + invalid)
    path.write_bytes(original)
    rows, counts, _ = capture_rows(tmp_path, start, start + timedelta(days=1))
    assert len(rows) == 1
    assert counts["valid_occurrences"] == 3
    assert counts["duplicate_occurrences_removed"] == 1
    assert counts["unique_packets_outside_capture_day"] == 1
    assert counts["invalid_candidates"] == 1
    assert path.read_bytes() == original


def test_only_axis_is_time_and_nonfinite_values_remain_missing():
    timestamp = datetime(2026, 9, 23, tzinfo=timezone.utc)
    names = ["spacecraft_time_seconds_since_2000", "time_since_boot", "uhf_temp", "uhf_alive"]
    rows = [(timestamp, dict(zip(names, [843500000, 123, float("nan"), "ALIVE"])))]
    frame = json.loads(frame_values(names, rows))[0]
    assert [field["type"] for field in frame["schema"]["fields"]] == [
        "time", "number", "number", "number", "string"
    ]
    assert frame["data"]["values"][1:] == [[843500000], [123], [None], ["ALIVE"]]


def test_unapproved_fields_cannot_enter_the_preview():
    with pytest.raises(ValueError, match="unapproved"):
        frame_values(["opaque_private_bytes"], [])


def test_histogram_preserves_prebinned_counts_in_dn_order():
    counts = [700, 90, 15, 3, 0, 1]
    panel = histogram(dict(zip((f"csie_img_hist_{i}" for i in range(6)), counts)), (0, 0, 12, 8))
    data = json.loads(panel["targets"][0]["rawFrameContent"])[0]["data"]["values"]
    assert data == [["0–31", "32–63", "64–95", "96–127", "128–159", "160–191"], counts]


@pytest.mark.parametrize("reported", ["SAFE", 1, "1", 99])
def test_mode_choices_highlight_exact_state_and_preserve_unknown(reported):
    panel = states("CDH mode", "mode_system_mode", reported, (0, 0, 12, 3))
    data = json.loads(panel["targets"][0]["rawFrameContent"])[0]
    by_name = dict(zip((f["name"] for f in data["schema"]["fields"]), data["data"]["values"]))
    selected = [name for name, value in by_name.items() if value == [1] and not name.endswith("_text_color")]
    assert selected == (["UNKNOWN"] if reported == 99 else ["SAFE"])
    if reported == 99:
        assert "99" in panel["description"]
