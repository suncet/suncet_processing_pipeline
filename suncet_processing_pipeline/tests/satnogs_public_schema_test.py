"""Contract tests for the reviewed public APID 1 field schema."""

from suncet_processing_pipeline.satnogs.public_schema import (
    PUBLIC_BEACON_FIELD_COUNT,
    load_public_beacon_schema,
)


def _by_name():
    return {field.public_name: field for field in load_public_beacon_schema()}


def test_public_schema_is_complete_ordered_and_unique():
    fields = load_public_beacon_schema()

    assert len(fields) == PUBLIC_BEACON_FIELD_COUNT == 112
    assert fields[0].public_name == "ccsds_version"
    assert fields[-1].public_name == "xband_data_source"


def test_public_schema_excludes_uplink_and_command_status_source_fields():
    fields = load_public_beacon_schema()
    source_names = {field.source_field for field in fields}

    assert "beac_fp_resp_count" not in source_names
    assert "beacon_checksum" not in source_names
    assert not any("_cmd_" in name or name.startswith("beac_cmd") for name in source_names)
    assert not any("arm_state" in name for name in source_names)


def test_reviewed_units_and_dualsps_scaling_are_encoded():
    fields = _by_name()

    assert fields["adcs_body_rate_1"].unit == "rad/s"
    assert fields["adcs_wheel_speed_1"].unit == "rpm"
    assert fields["adcs_sun_point_angle_error"].unit == "deg"
    assert fields["clt_hours_until_reboot"].unit == "h"

    flare = fields["dsps_flare_magnitude"]
    assert flare.data_type == "I8"
    assert flare.unit == "log10(XRS-B flux)"
    assert "C1=1.000000e-01" in flare.conversion_or_status_map

    phase = fields["dsps_flare_phase"]
    assert "40/IN_FLARE_RISING" in phase.conversion_or_status_map


def test_storage_pointers_are_public_but_fine_time_remains_raw():
    fields = _by_name()

    assert fields["partition_write_adcs"].unit == "raw address"
    assert fields["partition_read_sci"].unit == "raw address"
    assert fields["spacecraft_time_fine_raw"].data_type == "U16"
    assert "TBC" in fields["spacecraft_time_fine_raw"].unit


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
