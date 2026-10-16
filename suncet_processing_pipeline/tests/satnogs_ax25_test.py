"""Framing and independent integrity checks for the public AX.25 adapter."""

import binascii
from pathlib import Path

import pytest

from suncet_processing_pipeline.satnogs.ax25 import (
    SUNCET_AX25_HEADER,
    crc16_x25,
    decode_public_ax25_beacon,
)
from suncet_processing_pipeline.satnogs.beacon_contract import BeaconValidationError
from suncet_processing_pipeline.satnogs.decoder import decode_public_beacon


PACKET = bytes.fromhex(
    (Path(__file__).parents[1] / "satnogs/test_data/suncet_apid1_synthetic_252.hex")
    .read_text(encoding="ascii")
)
HEADER = bytes.fromhex("98 82 a6 a0 40 40 41 a6 aa 9c 62 40 40 41 03 f0")


def _with_fcs(data):
    # Independent implementation: reverse bits for the non-reflected C routine.
    reflected = bytes(int(f"{octet:08b}"[::-1], 2) for octet in data)
    crc = binascii.crc_hqx(reflected, 0xFFFF)
    result = int(f"{crc:016b}"[::-1], 2) ^ 0xFFFF
    return bytes(data) + result.to_bytes(2, "little")


def test_crc_check_vector_and_literal_header():
    assert crc16_x25(b"123456789") == 0x906E
    assert _with_fcs(b"123456789")[-2:] == b"\x6e\x90"
    assert SUNCET_AX25_HEADER == HEADER


@pytest.mark.parametrize("fcs_present", [False, True])
def test_both_boundaries_return_exactly_the_approved_public_values(fcs_present):
    frame = HEADER + PACKET
    if fcs_present:
        frame = _with_fcs(frame)
    result = decode_public_ax25_beacon(frame, fcs_present=fcs_present)
    assert result == decode_public_beacon(PACKET)
    assert len(result) == 111


@pytest.mark.parametrize("offset", range(16))
def test_every_header_octet_is_enforced_even_with_valid_fcs(offset):
    frame = bytearray(HEADER + PACKET)
    frame[offset] ^= 1
    with pytest.raises(BeaconValidationError, match="downlink header"):
        decode_public_ax25_beacon(_with_fcs(frame), fcs_present=True)


@pytest.mark.parametrize("fcs_present", [False, True])
@pytest.mark.parametrize("offset", [16 + 48, 16 + 124, 16 + 247, 16 + 251])
def test_payload_and_opaque_bytes_require_fletcher_even_with_valid_fcs(offset, fcs_present):
    frame = bytearray(HEADER + PACKET)
    frame[offset] ^= 1
    if fcs_present:
        frame = _with_fcs(frame)
    with pytest.raises(BeaconValidationError, match="Fletcher-32"):
        decode_public_ax25_beacon(frame, fcs_present=fcs_present)


@pytest.mark.parametrize("offset", [268, 269])
def test_each_fcs_byte_is_validated(offset):
    frame = bytearray(_with_fcs(HEADER + PACKET))
    frame[offset] ^= 1
    with pytest.raises(BeaconValidationError, match="CRC-16/X-25"):
        decode_public_ax25_beacon(frame, fcs_present=True)


@pytest.mark.parametrize("fcs_present", [False, True])
@pytest.mark.parametrize("frame", [
    PACKET, HEADER + PACKET[:-1], HEADER + PACKET + b"\x00",
    b"\x7e" + HEADER + PACKET + b"\x7e", (HEADER + PACKET) * 2,
])
def test_no_scanning_padding_or_flag_stripping(frame, fcs_present):
    with pytest.raises(BeaconValidationError):
        decode_public_ax25_beacon(frame, fcs_present=fcs_present)


def test_fcs_mode_must_match_the_supplied_boundary():
    with pytest.raises(BeaconValidationError, match="requires 270"):
        decode_public_ax25_beacon(HEADER + PACKET, fcs_present=True)
    with pytest.raises(BeaconValidationError, match="requires 268"):
        decode_public_ax25_beacon(_with_fcs(HEADER + PACKET), fcs_present=False)
    with pytest.raises(TypeError):
        decode_public_ax25_beacon(HEADER + PACKET)
    with pytest.raises(TypeError):
        decode_public_ax25_beacon(HEADER + PACKET, fcs_present="False")


def test_other_apids_and_command_packets_are_rejected():
    for primary in (0x0802, 0x1801):
        packet = bytearray(PACKET)
        packet[:2] = primary.to_bytes(2, "big")
        with pytest.raises(BeaconValidationError):
            decode_public_ax25_beacon(_with_fcs(HEADER + packet), fcs_present=True)
