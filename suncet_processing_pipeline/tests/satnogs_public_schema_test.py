"""Contract tests for the reviewed public APID 1 field schema."""

import csv
from pathlib import Path

import pytest

from suncet_processing_pipeline.satnogs import public_schema
from suncet_processing_pipeline.satnogs.public_schema import (
    PUBLIC_BEACON_CHECKSUM_BYTES,
    PUBLIC_BEACON_CTDB_VERSION,
    PUBLIC_BEACON_FIELD_COUNT,
    PUBLIC_BEACON_PACKET_BYTES,
    load_public_beacon_schema,
)


def _by_name():
    return {field.public_name: field for field in load_public_beacon_schema()}


def test_public_schema_is_complete_ordered_and_unique():
    fields = load_public_beacon_schema()

    assert PUBLIC_BEACON_CTDB_VERSION == "2.0.5"
    assert PUBLIC_BEACON_PACKET_BYTES == 252
    assert PUBLIC_BEACON_CHECKSUM_BYTES == 4
    assert len(fields) == PUBLIC_BEACON_FIELD_COUNT == 111
    assert fields[0].public_name == "ccsds_version"
    assert fields[-1].public_name == "xband_data_source"


def test_public_schema_excludes_uplink_and_command_status_source_fields():
    fields = load_public_beacon_schema()
    source_names = {field.source_field for field in fields}

    assert "beac_fp_resp_count" not in source_names
    assert "beacon_checksum" not in source_names
    assert not any("_cmd_" in name or name.startswith("beac_cmd") for name in source_names)
    assert not any("arm_state" in name for name in source_names)


def test_reviewed_units_and_current_dualsps_representation_are_encoded():
    fields = _by_name()

    assert fields["adcs_body_rate_1"].unit == "rad/s"
    assert fields["adcs_wheel_speed_1"].unit == "rpm"
    assert fields["adcs_sun_point_angle_error"].unit == "deg"
    assert fields["clt_hours_until_reboot"].unit == "h"

    flare = fields["dsps_flare_magnitude"]
    assert flare.data_type == "U8"
    assert flare.unit == "raw"
    assert flare.conversion_or_status_map == ""
    assert "no engineering conversion" in flare.description

    phase = fields["dsps_flare_phase"]
    assert "40/RISING_FLARE" in phase.conversion_or_status_map
    assert "24/DECLINING_FLARE" in phase.conversion_or_status_map
    assert "4/FLARE_START" in phase.conversion_or_status_map


def test_current_ctdb_layout_and_changed_engineering_values():
    fields = _by_name()

    assert "csie_meta_nand_sci_write_ptr" not in fields
    assert fields["time_since_boot"].byte_offset == 52
    assert fields["adcs_body_rate_1"].byte_offset == 72
    assert fields["adcs_wheel_speed_1"].byte_offset == 108
    assert fields["mode_system_mode"].byte_offset == 227

    pa_temperature = fields["xband_pa_temp"]
    assert pa_temperature.byte_offset == 120
    assert pa_temperature.data_type == "I32"
    assert pa_temperature.bit_length == 32

    dsps_temperature = fields["dsps_sensor_board_temp"]
    assert dsps_temperature.byte_offset == 212
    assert dsps_temperature.data_type == "I16"
    for name in ("battery_1_voltage", "battery_2_voltage"):
        assert fields[name].conversion_or_status_map == (
            "C0=0.000000e+00 C1=8.862300e-03"
        )

    assert fields["xband_data_source"].byte_offset == 246
    assert max(f.bit_offset + f.bit_length for f in fields.values()) == 247 * 8


def test_storage_pointers_and_resolved_millisecond_time_are_public():
    fields = _by_name()

    assert fields["partition_write_adcs"].unit == "raw address"
    assert fields["partition_read_sci"].unit == "raw address"
    fine_time = fields["spacecraft_time_milliseconds"]
    assert fine_time.data_type == "U16"
    assert fine_time.unit == "ms"
    assert "0 through 999" in fine_time.description


def test_csie_histogram_defaults_and_beacon_truncation_are_documented():
    fields = _by_name()
    expected_ranges = ("0-31", "32-63", "64-95", "96-127", "128-159", "160-191")

    for index, expected_range in enumerate(expected_ranges):
        field = fields[f"csie_img_hist_{index}"]
        assert field.unit == "count"
        assert expected_range in field.description
        assert "configurable DN range" in field.description

    assert "truncates the full histogram after bin 5" in fields[
        "csie_img_hist_5"
    ].description


@pytest.mark.parametrize(
    ("name", "changes", "message"),
    [
        ("xband_data_source", {"bit_offset": "1984", "byte_offset": "248"}, "outside"),
        ("ccsds_version", {"bit_offset": "-8", "byte_offset": "-1"}, "outside"),
        ("ccsds_version", {"data_type": "X3"}, "does not match"),
        ("ccsds_version", {"data_type": "U0", "bit_length": "0"}, "does not match"),
        (
            "xband_pa_temp",
            {"data_type": "F16", "bit_length": "16"},
            "floating-point width",
        ),
        ("ccsds_version", {"data_type": "I3"}, "byte aligned"),
        ("ccsds_packet_type", {"source_field": "VERSION"}, "unique source"),
    ],
)
def test_schema_rejects_invalid_payload_layout(
    tmp_path, monkeypatch, name, changes, message
):
    original = Path(public_schema.__file__).with_name("public_beacon_schema.csv")
    with original.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    row = next(row for row in rows if row["public_name"] == name)
    row.update(changes)
    with (tmp_path / original.name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(public_schema, "files", lambda _package: tmp_path)

    with pytest.raises(ValueError, match=message):
        load_public_beacon_schema()
