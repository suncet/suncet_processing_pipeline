"""Unit tests for realtime UHF parser and telemetry state."""

from datetime import datetime, timezone
from pathlib import Path

from suncet_processing_pipeline.realtime_display.color_limits import ColorLimitEvaluator
from suncet_processing_pipeline.realtime_display.config import (
    DecodeConfig,
    FramingConfig,
    load_config,
)
from suncet_processing_pipeline.realtime_display.main import _packet_ignored
from suncet_processing_pipeline.realtime_display.packet_decoder import (
    DecodedPacket,
    RealtimePacketDecoder,
)
from suncet_processing_pipeline.realtime_display.telemetry_state import (
    TelemetrySelector,
    TelemetryStore,
)
from suncet_processing_pipeline.realtime_display.uhf_realtime_parser import (
    UHFRealtimeParser,
)


def _framing(**overrides):
    values = dict(
        allow_ccsds_asm=True,
        ccsds_asm=b"\x1a\xcf\xfc\x1d",
        allow_frame_prefix=True,
        frame_prefix=b"\x1b\xad\xca\xfe",
        frame_prefix_bytes=16,
        allow_ax25_header=True,
        allow_missing_ax25_header=True,
        ax25_destination="SUN1",
        ax25_source="LASP",
        ax25_ctrl=0x03,
        ax25_pid=0xF0,
        segmentation_enabled=True,
        segmented_packet_threshold_bytes=10,
        max_segment_payload_bytes=10,
        segment_flag_start=1,
        segment_flag_middle=0,
        segment_flag_end=2,
        max_packet_bytes=4096,
        max_buffer_bytes=4096,
    )
    values.update(overrides)
    return FramingConfig(**values)


def _ccsds_packet(apid=42, body=b"0123456789abcdef"):
    length = len(body) - 1
    first = 0x0800 | (apid & 0x07FF)
    seq = 0xC000
    return (
        first.to_bytes(2, "big")
        + seq.to_bytes(2, "big")
        + length.to_bytes(2, "big")
        + body
    )


def _ax25():
    return b"SUN1    " + b"LASP    " + bytes([0x03, 0xF0])


def _frame_prefix():
    return b"\x1b\xad\xca\xfe" + b"\x00" * 12


def test_parser_emits_asm_prefixed_ccsds_across_tcp_chunks():
    parser = UHFRealtimeParser(_framing(allow_ax25_header=False), valid_apids={42})
    packet = _ccsds_packet()
    stream = b"\x99\x88" + b"\x1a\xcf\xfc\x1d" + packet

    assert parser.feed(stream[:3]) == []
    out = parser.feed(stream[3:])

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].source == "ccsds_asm_direct"


def test_parser_emits_frame_prefixed_ccsds_across_tcp_chunks():
    parser = UHFRealtimeParser(_framing(allow_ax25_header=False), valid_apids={42})
    packet = _ccsds_packet()
    stream = b"\x99\x88" + _frame_prefix() + packet

    assert parser.feed(stream[:8]) == []
    out = parser.feed(stream[8:])

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].source == "frame_prefix_direct"
    assert parser.stats.frame_prefix_frames_seen == 1


def test_frame_prefix_parses_unknown_apid_without_losing_sync():
    parser = UHFRealtimeParser(_framing(allow_ax25_header=False), valid_apids={42})
    unknown = _ccsds_packet(apid=999)
    known = _ccsds_packet(apid=42)

    out = parser.feed(_frame_prefix() + unknown + _frame_prefix() + known)

    assert [envelope.packet for envelope in out] == [unknown, known]
    assert parser.stats.frame_prefix_frames_seen == 2


def test_sync_prefix_beats_false_direct_candidate_before_sync():
    parser = UHFRealtimeParser(_framing(allow_ax25_header=False), valid_apids={42})
    packet = _ccsds_packet()
    false_direct_header = (
        (0x0800 | 42).to_bytes(2, "big")
        + (0xC000).to_bytes(2, "big")
        + (4000).to_bytes(2, "big")
    )

    out = parser.feed(false_direct_header + _frame_prefix() + packet)

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].source == "frame_prefix_direct"


def _segment_header(packet, segment_count, flag):
    return (
        (0x1234).to_bytes(2, "big")
        + (0x002A).to_bytes(2, "big")
        + len(packet).to_bytes(2, "big")
        + (0x0042).to_bytes(2, "big")
        + (0x0064).to_bytes(2, "big")
        + bytes([segment_count, flag])
    )


def test_parser_emits_direct_ccsds_across_tcp_chunks():
    parser = UHFRealtimeParser(_framing(), valid_apids={42})
    packet = _ccsds_packet()

    assert parser.feed(packet[:5]) == []
    out = parser.feed(packet[5:])

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].source == "direct_ccsds"


def test_parser_emits_ax25_wrapped_direct_packet():
    parser = UHFRealtimeParser(_framing(), valid_apids={42})
    packet = _ccsds_packet()

    out = parser.feed(_ax25() + packet)

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].ax25_present is True
    assert out[0].source == "ax25_direct"


def test_decoder_injects_raw_secondary_header_when_generated_fields_are_absent():
    decoder = RealtimePacketDecoder(None)
    packet = _ccsds_packet(
        apid=537,
        body=(833326475).to_bytes(4, "big") + (1234).to_bytes(2, "big") + b"data",
    )

    decoded = decoder.decode(packet)

    assert decoded is not None
    assert decoded.fields["ccsdsSecHeader2_sec"] == 833326475
    assert decoded.fields["ccsdsSecHeader2_sub"] == 1234


def test_parser_reassembles_ax25_segmented_packet():
    parser = UHFRealtimeParser(_framing(), valid_apids={42})
    packet = _ccsds_packet(body=b"abcdefghijklmnopqrstuvwxyz")
    chunks = [packet[:10], packet[10:20], packet[20:30], packet[30:]]
    frames = [
        _ax25() + _segment_header(packet, 0, 1) + chunks[0],
        _ax25() + _segment_header(packet, 1, 0) + chunks[1],
        _ax25() + _segment_header(packet, 2, 0) + chunks[2],
        _ax25() + _segment_header(packet, 3, 2) + chunks[3],
    ]

    out = []
    for frame in frames:
        out.extend(parser.feed(frame))

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].segmented is True
    assert out[0].source == "ax25_segmented"


def test_parser_reassembles_segmented_packet_without_ax25():
    parser = UHFRealtimeParser(_framing(allow_ax25_header=False), valid_apids={42})
    packet = _ccsds_packet(body=b"abcdefghijklmnopqrstuvwxyz")
    chunks = [packet[:10], packet[10:20], packet[20:30], packet[30:]]

    out = []
    out.extend(parser.feed(_segment_header(packet, 0, 1) + chunks[0]))
    out.extend(parser.feed(_segment_header(packet, 1, 0) + chunks[1]))
    out.extend(parser.feed(_segment_header(packet, 2, 0) + chunks[2]))
    out.extend(parser.feed(_segment_header(packet, 3, 2) + chunks[3]))

    assert len(out) == 1
    assert out[0].packet == packet
    assert out[0].source == "segmented"


def test_telemetry_store_selects_temperatures_and_voltages():
    store = TelemetryStore(
        selector=TelemetrySelector(
            field_patterns=["*temp*", "*bus*v*", "*bat*v*", "*curr*"],
            excluded_field_patterns=[
                "*sa_minus_y_temp*",
                "*sa_plus_y_temp*",
                "*1p0*",
                "*1p8*",
                "*3p3*",
            ],
        ),
        history_points=5,
    )
    packet = DecodedPacket(
        apid=1,
        packet_name="BEACON",
        fields={
            "ccsdsSecHeader2_sec_beacon": 0,
            "ccsdsSecHeader2_sub_beacon": 0,
            "beac_time_mission_elapsed_time": 123.0,
            "beac_ana_cdh_temp": 22.5,
            "beac_ana_sa_minus_y_temp": 19.0,
            "beac_ana_eps_bus_v": 12.1,
            "beac_ana_eps_bus_i": 0.3,
            "beac_batt1_charge_current": 0.42,
            "beac_ana_bat1_v": 14.8,
            "beac_ana_3p3_v": 3.3,
        },
        header={"apid": 1},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()
    names = {field["field"] for field in snapshot["fields"]}

    assert snapshot["last_packet"]["packet_time"] is not None
    assert snapshot["last_packet"]["packet_time"] == 0.0
    assert snapshot["last_packet"]["time_source"] == "ccsds_secondary_header_coarse"
    assert snapshot["last_packet"]["onboard_utc"] == "2000-01-01T00:00:00Z"
    assert snapshot["onboard_utc"] == "2000-01-01T00:00:00Z"
    assert "beac_ana_cdh_temp" in names
    assert "beac_ana_eps_bus_v" in names
    assert "beac_ana_bat1_v" in names
    assert "beac_batt1_charge_current" in names
    assert "beac_ana_eps_bus_i" not in names
    assert "beac_ana_sa_minus_y_temp" not in names
    assert "beac_ana_3p3_v" not in names


def test_color_limits_xml_marks_numeric_ranges():
    path = (
        Path(__file__).resolve().parents[1]
        / "realtime_display"
        / "color_limits_tlm.xml"
    )
    limits = ColorLimitEvaluator.from_xml(path)

    assert limits.evaluate("beac_ana_bat1_v", 15.0)["state"] == "green"
    assert limits.evaluate("beac_ana_bat1_v", 13.5)["state"] == "yellow"
    assert limits.evaluate("beac_ana_bat1_v", 12.5)["state"] == "red"


def test_telemetry_store_attaches_limit_status():
    limits = ColorLimitEvaluator.from_xml(
        Path(__file__).resolve().parents[1]
        / "realtime_display"
        / "color_limits_tlm.xml"
    )
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*bat*v*"]),
        color_limits=limits,
        history_points=5,
    )
    packet = DecodedPacket(
        apid=1,
        packet_name="BEACON",
        fields={"beac_time_since_boot": 1.0, "beac_ana_bat1_v": 12.0},
        header={"apid": 1},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()

    assert snapshot["fields"][0]["field"] == "beac_ana_bat1_v"
    assert snapshot["fields"][0]["limit_status"]["state"] == "red"


def test_onboard_utc_uses_configurable_midnight_j2000_epoch_with_leap_seconds():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
        j2000_epoch_utc=datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    packet = DecodedPacket(
        apid=1,
        packet_name="BEACON",
        fields={
            "ccsdsSecHeader2_sec_beacon": 833237728,
            "ccsdsSecHeader2_sub_beacon": 0,
            "beac_ana_cdh_temp": 20.0,
        },
        header={"apid": 1},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()

    assert snapshot["last_packet"]["packet_time"] == 833237728.0
    assert snapshot["last_packet"]["onboard_utc"] == "2026-05-27T22:55:33Z"
    assert snapshot["onboard_utc"] == "2026-05-27T22:55:33Z"


def test_csie_apids_update_displayed_onboard_utc_from_secondary_header():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
    )
    store.add_packet(
        DecodedPacket(
            apid=1,
            packet_name="beacon",
            fields={
                "ccsdsSecHeader2_sec_beacon": 0,
                "ccsdsSecHeader2_sub_beacon": 0,
                "beac_time_since_boot": 0,
                "beac_ana_cdh_temp": 20.0,
            },
            header={"apid": 1},
            decode_status="decoded",
        )
    )
    store.add_packet(
        DecodedPacket(
            apid=537,
            packet_name="csie_hk",
            fields={
                "ccsdsSecHeader2_sec_csie_hk": 86400,
                "ccsdsSecHeader2_sub_csie_hk": 0,
                "csie_adc_core_temp": 21.0,
            },
            header={"apid": 537},
            decode_status="decoded",
        )
    )

    snapshot = store.snapshot()

    assert snapshot["last_packet"]["packet_name"] == "csie_hk"
    assert snapshot["last_packet"]["packet_time"] == 86400.0
    assert snapshot["last_packet"]["onboard_utc"] == "2000-01-02T00:00:00Z"
    assert snapshot["onboard_utc"] == "2000-01-02T00:00:00Z"


def test_packet_time_uses_secondary_header_instead_of_nonclock_time_fields():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
    )
    packet = DecodedPacket(
        apid=537,
        packet_name="csie_hk",
        fields={
            "ccsdsSecHeader2_sec_csie_hk": 100,
            "ccsdsSecHeader2_sub_csie_hk": 32768,
            "csie_exposure_time": 3000,
            "csie_adc_core_temp": 21.0,
        },
        header={"apid": 537},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()

    assert snapshot["last_packet"]["packet_time"] == 100.0
    assert snapshot["last_packet"]["time_source"] == "ccsds_secondary_header_coarse"
    csie_temp = next(
        field
        for field in snapshot["fields"]
        if field["field"] == "csie_adc_core_temp"
    )
    assert csie_temp["packet_time"] == 100.0
    assert csie_temp["packet_utc"] == "2000-01-01T00:01:40Z"


def test_secondary_header_packet_updates_plot_history_without_met_calibration():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
    )
    packet = DecodedPacket(
        apid=537,
        packet_name="csie_hk",
        fields={
            "ccsdsSecHeader2_sec_csie_hk": 100,
            "ccsdsSecHeader2_sub_csie_hk": 0,
            "csie_adc_core_temp": 21.0,
        },
        header={"apid": 537},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()

    assert snapshot["last_packet"]["time_status"] == "accepted"
    assert snapshot["last_packet"]["packet_time"] == 100.0
    assert snapshot["fields"][0]["field"] == "csie_adc_core_temp"
    assert snapshot["fields"][0]["packet_time"] == 100.0


def test_telemetry_store_rejects_implausible_packet_time_jumps():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
        max_plot_time_jump_seconds=20.0,
        max_plot_time_wall_rate=0.0,
    )
    good_packet = DecodedPacket(
        apid=1,
        packet_name="beacon",
        fields={
            "beac_time_mission_elapsed_time": 100.0,
            "beac_ana_cdh_temp": 20.0,
        },
        header={"apid": 1},
        decode_status="decoded",
    )
    bad_packet = DecodedPacket(
        apid=1,
        packet_name="beacon",
        fields={
            "beac_time_mission_elapsed_time": 1_000_000.0,
            "beac_ana_cdh_temp": 99.0,
        },
        header={"apid": 1},
        decode_status="decoded",
    )

    store.add_packet(good_packet)
    store.add_packet(bad_packet)
    snapshot = store.snapshot()
    field = next(
        item for item in snapshot["fields"] if item["field"] == "beac_ana_cdh_temp"
    )

    assert field["value"] == 20.0
    assert [point["packet_time"] for point in field["history"]] == [100.0]
    assert snapshot["stats"]["time_rejections"] == 1
    assert snapshot["last_packet"]["time_status"] == "rejected"
    assert snapshot["last_packet"]["selected_points"] == 0


def test_telemetry_store_filters_value_outliers_from_plot_history():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["sensor_temp"]),
        history_points=10,
        value_filter_enabled=True,
        value_filter_sigma_threshold=3.0,
        value_filter_min_samples=5,
        value_filter_window_points=5,
        value_filter_std_floor=0.1,
        value_filter_relative_std_floor=0.0,
    )

    for index, value in enumerate([10.0, 10.1, 9.9, 10.05, 9.95], start=1):
        store.add_packet(
            DecodedPacket(
                apid=1,
                packet_name="beacon",
                fields={
                    "beac_time_since_boot": float(index),
                    "sensor_temp": value,
                },
                header={"apid": 1},
                decode_status="decoded",
            )
        )

    store.add_packet(
        DecodedPacket(
            apid=1,
            packet_name="beacon",
            fields={
                "beac_time_since_boot": 6.0,
                "sensor_temp": 50.0,
            },
            header={"apid": 1},
            decode_status="decoded",
        )
    )
    snapshot = store.snapshot()
    field = next(item for item in snapshot["fields"] if item["field"] == "sensor_temp")

    assert field["value"] == 9.95
    assert [point["value"] for point in field["history"]] == [
        10.0,
        10.1,
        9.9,
        10.05,
        9.95,
    ]
    assert snapshot["stats"]["value_filter_rejections"] == 1
    assert snapshot["last_packet"]["selected_points"] == 0
    assert snapshot["last_packet"]["filtered_points"] == 1
    assert "sensor_temp" in snapshot["last_packet"]["value_filter_rejection_reasons"][0]


def test_low_secondary_header_time_does_not_update_plots_or_onboard_utc():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
        min_valid_j2000_seconds=700_000_000.0,
    )
    packet = DecodedPacket(
        apid=16,
        packet_name="payload_hk",
        fields={
            "ccsdsSecHeader2_sec_payload_hk": 2266,
            "ccsdsSecHeader2_sub_payload_hk": 0,
            "payload_temp": 19.0,
        },
        header={"apid": 16},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()

    assert snapshot["fields"] == []
    assert snapshot["onboard_utc"] is None
    assert snapshot["last_packet"]["onboard_utc"] is None
    assert snapshot["last_packet"]["time_status"] == "rejected"
    assert "below minimum" in snapshot["last_packet"]["time_rejection_reason"]


def test_beacon_power_states_are_exposed_for_status_summary():
    store = TelemetryStore(
        selector=TelemetrySelector(field_patterns=["*temp*"]),
        history_points=5,
    )
    packet = DecodedPacket(
        apid=1,
        packet_name="beacon",
        fields={
            "beac_mode_system_mode": "SCIENCE",
            "beac_eps_pwr_state_adcs": "ON",
            "beac_eps_pwr_state_uhf": 1,
            "beac_eps_pwr_state_xband": "OFF",
            "beac_eps_pwr_state_csie": True,
            "beac_eps_pwr_state_dsps": 0,
        },
        header={"apid": 1},
        decode_status="decoded",
    )

    store.add_packet(packet)
    snapshot = store.snapshot()
    states = {
        item["label"]: item["state"]
        for item in snapshot["beacon_power_states"]["states"]
    }

    assert states == {
        "CDH": "green",
        "ADCS": "on",
        "UHF": "on",
        "X-Band": "off",
        "CSIE": "on",
        "DSPS": "off",
    }
    modes = {
        item["label"]: item["display_value"]
        for item in snapshot["beacon_power_states"]["states"]
    }
    assert modes["CDH"] == "SCIENCE MODE"
    assert snapshot["beacon_power_states"]["age_seconds"] is not None


def test_realtime_config_loads_time_conversion_settings():
    config = load_config(
        Path(__file__).resolve().parents[1] / "realtime_display" / "config.ini"
    )

    assert config.time.j2000_epoch_utc == datetime(
        2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc
    )
    assert config.time.add_post_j2000_leap_seconds is True
    assert config.tcp.idle_reconnect_seconds == 15.0
    assert config.time.min_valid_j2000_seconds == 700000000.0
    assert config.time.max_plot_time_jump_seconds == 300.0
    assert config.time.max_plot_time_wall_rate == 5.0
    assert config.value_filter.enabled is True
    assert config.value_filter.sigma_threshold == 3.0
    assert config.value_filter.min_samples == 12
    assert config.value_filter.window_points == 50
    assert config.value_filter.std_floor == 0.05
    assert config.value_filter.relative_std_floor == 0.01
    assert config.framing.allow_frame_prefix is True
    assert config.framing.frame_prefix == b"\x1b\xad\xca\xfe"
    assert config.framing.frame_prefix_bytes == 16
    assert config.decode.ignored_apids == [
        3,
        6,
        8,
        11,
        14,
        15,
        19,
        20,
        27,
        49,
        57,
        515,
        518,
        520,
        523,
        526,
        527,
        532,
        539,
    ]
    assert config.decode.ignored_packet_name_patterns == [
        "*des*",
        "*mem*",
        "*log*",
        "*tbl*",
        "ana_hk",
    ]
    selector = TelemetrySelector(
        field_patterns=config.telemetry.field_patterns,
        excluded_field_patterns=config.telemetry.excluded_field_patterns,
    )
    assert selector.field_matches("csie_det0_therm")
    assert selector.field_matches("csie_det1_therm")
    assert selector.field_matches("beac_ana_sa_8_cell_str_v")
    assert selector.field_matches("beac_ana_sa_9_cell_str_v")
    assert selector.field_matches("beac_ana_xact_v")
    assert selector.field_matches("beac_ana_uhf_v")
    assert selector.field_matches("beac_ana_xband_v")
    assert selector.field_matches("beac_ana_sa_8_cell_str_i")
    assert not selector.field_matches("beac_ana_3p3_v")
    assert not selector.field_matches("csie_curr_det")
    assert not selector.field_matches("csie_curr_inst")
    assert not selector.field_matches("csie_curr_peak_util")
    assert not selector.field_matches("csie_curr_util")
    assert not selector.field_matches("csie_volt_1p5")
    assert not selector.field_matches("csie_volt_2p5")
    assert not selector.field_matches("csie_volt_adc_buff")
    assert not selector.field_matches("csie_volt_det")
    assert not selector.field_matches("csie_volt_util")


def test_packet_ignore_config_matches_apids_and_packet_names():
    decode_config = DecodeConfig(
        require_packet_checksums=False,
        drop_failed_checksums=False,
        ignored_apids=[99],
        ignored_packet_name_patterns=["*des*", "*mem*", "*log*", "*tbl*", "ana_hk"],
    )

    assert _packet_ignored(
        DecodedPacket(
            apid=1,
            packet_name="des_hk",
            fields={},
            header={},
            decode_status="decoded",
        ),
        decode_config,
    )
    assert _packet_ignored(
        DecodedPacket(
            apid=2,
            packet_name="event_log",
            fields={},
            header={},
            decode_status="decoded",
        ),
        decode_config,
    )
    assert _packet_ignored(
        DecodedPacket(
            apid=3,
            packet_name="tbl_hk",
            fields={},
            header={},
            decode_status="decoded",
        ),
        decode_config,
    )
    assert _packet_ignored(
        DecodedPacket(
            apid=99,
            packet_name="ana_hk",
            fields={},
            header={},
            decode_status="decoded",
        ),
        decode_config,
    )
    assert _packet_ignored(
        DecodedPacket(
            apid=27,
            packet_name="ana_hk",
            fields={},
            header={},
            decode_status="decoded",
        ),
        decode_config,
    )
