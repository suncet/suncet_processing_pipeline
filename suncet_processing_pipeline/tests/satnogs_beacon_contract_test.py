"""Tests for the public APID 1 packet contract."""

import pytest

from suncet_processing_pipeline.satnogs.beacon_contract import (
    CURRENT_BEACON_PACKET_LENGTHS,
    BeaconValidationError,
    parse_beacon_packet,
    suncet_fletcher32,
)


def _beacon_packet(
    total_length: int = 252,
    *,
    apid: int = 1,
    coarse_seconds: int = 833_326_475,
    fine_milliseconds: int = 234,
    sequence_count: int = 42,
) -> bytes:
    opaque_length = total_length - 6 - 6 - 4
    secondary_and_data = (
        coarse_seconds.to_bytes(4, "big")
        + fine_milliseconds.to_bytes(2, "big")
        + bytes((index % 251 for index in range(opaque_length)))
    )
    first_word = 0x0800 | apid
    sequence_word = 0xC000 | sequence_count
    data_field_length = len(secondary_and_data) + 4
    header = (
        first_word.to_bytes(2, "big")
        + sequence_word.to_bytes(2, "big")
        + (data_field_length - 1).to_bytes(2, "big")
    )
    without_checksum = header + secondary_and_data
    return without_checksum + suncet_fletcher32(without_checksum).to_bytes(4, "big")


def test_fletcher32_known_word_order_vector():
    assert suncet_fletcher32(b"\x00\x01\x02\x03") == 0x05020402


def test_parses_current_ctdb_2_0_5_layout():
    packet = parse_beacon_packet(_beacon_packet())

    assert packet.packet_length == 252
    assert CURRENT_BEACON_PACKET_LENGTHS == {252}
    assert packet.sequence_flags == 3
    assert packet.sequence_count == 42
    assert packet.coarse_seconds == 833_326_475
    assert packet.fine_milliseconds == 234
    assert packet.fine_time_raw == 234
    assert packet.timestamp_seconds == 833_326_475.234
    assert packet.checksum == int.from_bytes(packet.raw[-4:], "big")


@pytest.mark.parametrize("packet_length", [16, 250, 251, 253, 256])
def test_rejects_every_noncurrent_layout_even_with_valid_checksum(packet_length):
    with pytest.raises(BeaconValidationError, match=r"CTDB 2\.0\.5 requires 252"):
        parse_beacon_packet(_beacon_packet(packet_length))


@pytest.mark.parametrize("packet_length", [0, 1, 5, 6, 11, 247, 248, 251])
def test_rejects_truncated_packets(packet_length):
    with pytest.raises(BeaconValidationError, match=r"CTDB 2\.0\.5 requires 252"):
        parse_beacon_packet(_beacon_packet()[:packet_length])


def test_rejects_non_beacon_apid():
    with pytest.raises(BeaconValidationError, match="APID is 2"):
        parse_beacon_packet(_beacon_packet(apid=2))


def test_rejects_header_length_mismatch():
    packet = bytearray(_beacon_packet())
    packet[4:6] = (244).to_bytes(2, "big")

    with pytest.raises(BeaconValidationError, match="header declares 251"):
        parse_beacon_packet(bytes(packet))


@pytest.mark.parametrize("offset", [12, 20, 247, 248, 251])
def test_rejects_corrupt_checksum(offset):
    packet = bytearray(_beacon_packet())
    packet[offset] ^= 0x01

    with pytest.raises(BeaconValidationError, match="Fletcher-32 mismatch"):
        parse_beacon_packet(bytes(packet))


@pytest.mark.parametrize("fine_milliseconds", [1000, 65535])
def test_rejects_fine_time_outside_one_second(fine_milliseconds):
    with pytest.raises(BeaconValidationError, match="expected 0-999 ms"):
        parse_beacon_packet(_beacon_packet(fine_milliseconds=fine_milliseconds))


@pytest.mark.parametrize("fine_milliseconds", [0, 999])
def test_accepts_fine_time_boundaries(fine_milliseconds):
    packet = parse_beacon_packet(_beacon_packet(fine_milliseconds=fine_milliseconds))
    assert packet.fine_milliseconds == fine_milliseconds
    assert packet.timestamp_seconds == 833_326_475 + fine_milliseconds / 1000


def test_rejects_packet_without_secondary_header():
    packet = bytearray(_beacon_packet())
    packet[0:2] = (1).to_bytes(2, "big")

    with pytest.raises(BeaconValidationError, match="secondary time header is absent"):
        parse_beacon_packet(bytes(packet))
