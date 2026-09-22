"""Tests for DSPS codegen and provisional APID 35 CSV decoding."""

import csv
from types import SimpleNamespace

import pytest

from ..make_level0_5 import (
    PacketRecord,
    _select_generated_decoder,
    decode_packet_records_to_csv,
    import_bus_decoder_bundle,
)


def _config(tmp_path):
    bus_decoders = tmp_path / "suncet_v2-0-4" / "decoders"
    bus_decoders.mkdir(parents=True)
    (bus_decoders / "gen_eus.py").write_text('SOURCE = "bus"\n')
    (bus_decoders / "gen_states.py").write_text('SOURCE = "bus"\n')
    (bus_decoders / "gen_pkts.py").write_text(
        "import gen_eus\nclass BUS:\n    source = gen_eus.SOURCE\n"
    )
    return SimpleNamespace(
        ctdb_base=str(tmp_path),
        version_bus="2.0.4",
        bus_ctdb_path=str(tmp_path / "suncet_v2-0-4"),
        packet_definitions_path=str(bus_decoders),
        csie_ctdb_path=str(tmp_path / "suncet_csie_v1-1-6"),
    )


def test_dsps_split_ctdb_uses_bus_helpers_and_generated_packet_fields(tmp_path):
    config = _config(tmp_path)
    split_decoders = tmp_path / "suncet_dsps_v2-0-4" / "decoders"
    split_decoders.mkdir(parents=True)
    (split_decoders / "gen_pkts.py").write_text(
        "import gen_eus\nimport gen_states\n"
        "class DSPS_PASS:\n"
        "    def __init__(self, packet, header, file_origin):\n"
        "        self.dsps_power_5V_mV = int.from_bytes(packet[:2], 'big')\n"
        "        self.dsps_pass_checksum = int.from_bytes(packet[112:116], 'big')\n"
        "        self.helpers = (gen_eus.SOURCE, gen_states.SOURCE)\n"
    )

    bundle = import_bus_decoder_bundle(config)
    packet_class, kind = _select_generated_decoder("dsps_data", bundle)
    decoded = packet_class(b"\x13\x88", b"\x00" * 6, "test.bin")

    assert kind == "generated_dsps_temp_dsps_data_alias"
    assert decoded.dsps_power_5V_mV == 5000
    assert decoded.helpers == ("bus", "bus")
    assert bundle.bus_pkts.BUS.source == "bus"
    assert not any("DSPS generated decoder not found" in warning for warning in bundle.warnings)


@pytest.mark.parametrize(
    ("unknown_tail", "expected_nonzero", "expected_suffix"),
    [
        (b"\x00\x00", "False", "unknown_tail_zero"),
        (b"\x01\x00", "True", "unknown_tail_nonzero"),
    ],
)
def test_dsps_data_csv_marks_provisional_fields_and_unknown_tail(
    tmp_path, unknown_tail, expected_nonzero, expected_suffix
):
    config = _config(tmp_path)
    split_decoders = tmp_path / "suncet_dsps_v2-0-4" / "decoders"
    split_decoders.mkdir(parents=True)
    (split_decoders / "gen_pkts.py").write_text(
        "import gen_eus\nimport gen_states\n"
        "class DSPS_PASS:\n"
        "    def __init__(self, packet, header, file_origin):\n"
        "        self.dsps_power_5V_mV = int.from_bytes(packet[:2], 'big')\n"
        "        self.dsps_pass_checksum = int.from_bytes(packet[112:116], 'big')\n"
    )
    data_field = (
        b"\x13\x88" + b"\x00" * 110 + unknown_tail + b"\x8f\xed\x3c\xa2"
    )
    packet = b"\x08\x23\xc0\x01\x00\x75" + data_field
    record = PacketRecord(
        packet_index=1,
        source_offset=0,
        apid=35,
        packet_len=124,
        source="test",
        acceptance_mode="checksum_bypassed_structural",
        checksum_validated=False,
        original_primary_header_endian="big",
        primary_header_normalized=False,
        payload_16bit_words_swapped=False,
        packet=packet,
    )

    stats = decode_packet_records_to_csv(
        [record], config, {35: "dsps_data"}, tmp_path / "decoded"
    )
    with (stats.decoded_dir / "decoded_apid_0035_dsps_data.csv").open(newline="") as f:
        row = next(csv.DictReader(f))

    assert stats.decoded_packets == 1
    assert row["dsps_power_5V_mV"] == "5000"
    assert "dsps_pass_checksum" not in row
    assert row["dsps_data_unknown_tail_2bytes_hex"] == unknown_tail.hex()
    assert row["dsps_data_unknown_tail_nonzero"] == expected_nonzero
    assert row["dsps_data_trailer_4bytes_hex"] == "8fed3ca2"
    assert row["dsps_data_trailer_validated"] == "False"
    assert row["dsps_data_schema_status"].endswith(expected_suffix)
    assert row["dsps_data_eu_conversion_status"] == (
        "raw_values_no_generated_dsps_conversions"
    )


def test_dsps_nested_ctdb_remains_supported(tmp_path):
    config = _config(tmp_path)
    nested_decoders = (
        tmp_path / "suncet_v2-0-4" / "decoders" / "dsps_decoders"
    )
    nested_decoders.mkdir(parents=True)
    (nested_decoders / "gen_eus.py").write_text('SOURCE = "nested"\n')
    (nested_decoders / "gen_states.py").write_text('SOURCE = "nested"\n')
    (nested_decoders / "gen_pkts.py").write_text(
        "import gen_eus\nimport gen_states\n"
        "class DSPS_PASS:\n"
        "    helpers = (gen_eus.SOURCE, gen_states.SOURCE)\n"
    )

    bundle = import_bus_decoder_bundle(config)

    assert bundle.dsps_decoders.DSPS_PASS.helpers == ("nested", "nested")
    assert bundle.bus_pkts.BUS.source == "bus"
