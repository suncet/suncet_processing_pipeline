"""Tests for the public APID 1 packet contract."""

import pytest

from suncet_processing_pipeline.satnogs.beacon_contract import (
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


@pytest.mark.parametrize("packet_length", [251, 252])
def test_parses_both_current_candidate_lengths(packet_length):
    packet = parse_beacon_packet(_beacon_packet(packet_length))

    assert packet.packet_length == packet_length
    assert packet.sequence_flags == 3
    assert packet.sequence_count == 42
    assert packet.coarse_seconds == 833_326_475
    assert packet.fine_milliseconds == 234
    assert packet.fine_time_raw == 234
    assert packet.timestamp_seconds == 833_326_475.234
    assert packet.checksum == int.from_bytes(packet.raw[-4:], "big")


def test_caller_can_enforce_one_confirmed_length():
    with pytest.raises(BeaconValidationError, match="accepted lengths are 252"):
        parse_beacon_packet(_beacon_packet(251), accepted_lengths={252})


def test_rejects_non_beacon_apid():
    with pytest.raises(BeaconValidationError, match="APID is 2"):
        parse_beacon_packet(_beacon_packet(apid=2))


def test_rejects_header_length_mismatch():
    packet = bytearray(_beacon_packet())
    packet[4:6] = (244).to_bytes(2, "big")

    with pytest.raises(BeaconValidationError, match="header declares 251"):
        parse_beacon_packet(bytes(packet))


def test_rejects_corrupt_checksum():
    packet = bytearray(_beacon_packet())
    packet[20] ^= 0x01

    with pytest.raises(BeaconValidationError, match="Fletcher-32 mismatch"):
        parse_beacon_packet(bytes(packet))


def test_rejects_fine_time_outside_one_second():
    with pytest.raises(BeaconValidationError, match="expected 0-999 ms"):
        parse_beacon_packet(_beacon_packet(fine_milliseconds=1_000))


def test_rejects_packet_without_secondary_header():
    packet = bytearray(_beacon_packet())
    packet[0:2] = (1).to_bytes(2, "big")

    with pytest.raises(BeaconValidationError, match="secondary time header is absent"):
        parse_beacon_packet(bytes(packet))
