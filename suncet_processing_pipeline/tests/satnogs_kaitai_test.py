"""Tests for the generated provisional Kaitai decoder and public fixture."""

import json
from pathlib import Path

from suncet_processing_pipeline.satnogs.beacon_contract import parse_beacon_packet
from suncet_processing_pipeline.satnogs.kaitai_generator import (
    CHECKSUM_BITS,
    CTDB_PACKET_BITS,
    PUBLIC_PAYLOAD_BITS,
    generate_kaitai,
)
from suncet_processing_pipeline.satnogs.synthetic_fixture import (
    build_synthetic_fixture,
)


SATNOGS_DIR = Path(__file__).parents[1] / "satnogs"


def test_tracked_kaitai_definition_is_generated_from_public_schema():
    tracked = (SATNOGS_DIR / "suncet_apid1.ksy").read_text(encoding="utf-8")

    assert tracked == generate_kaitai()
    assert "https://github.com/suncet/suncet_processing_pipeline/" in tracked
    assert "bit-endian: be" in tracked
    assert "if: _io.size == 252" in tracked
    assert "id: spacecraft_time_milliseconds" in tracked
    assert "max: 999" in tracked
    assert "source_field" not in tracked
    assert CTDB_PACKET_BITS == 2008
    assert CHECKSUM_BITS == 32
    assert PUBLIC_PAYLOAD_BITS == 1976


def test_synthetic_fixture_is_stable_and_contract_valid():
    packet, expected = build_synthetic_fixture()
    stored_hex = (SATNOGS_DIR / "test_data/suncet_apid1_synthetic_251.hex").read_text(
        encoding="ascii"
    )
    stored_expected = json.loads(
        (SATNOGS_DIR / "test_data/suncet_apid1_synthetic_251_expected.json").read_text(
            encoding="utf-8"
        )
    )

    assert bytes.fromhex(stored_hex) == packet
    assert stored_expected == expected
    parsed = parse_beacon_packet(packet, accepted_lengths={251})
    assert parsed.sequence_count == 42
    assert parsed.coarse_seconds == 833_326_475
    assert parsed.fine_milliseconds == 234
    assert parsed.timestamp_seconds == 833_326_475.234


def test_fixture_is_explicitly_synthetic_not_flight_data():
    packet, expected = build_synthetic_fixture()

    assert len(packet) == 251
    assert expected["mode_system_mode"]["engineering"] == "SCIENCE"
    assert expected["dsps_flare_phase"]["engineering"] == "IN_FLARE_RISING"
    assert expected["uhf_alive"]["engineering"] == "ALIVE"
    assert expected["adcs_sun_point_state"]["engineering"] == "ON_SUN"
