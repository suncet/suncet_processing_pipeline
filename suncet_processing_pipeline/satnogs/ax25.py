"""Decode the accepted SunCET downlink header and one public APID 1 beacon.

The caller declares whether the receiver retained the two FCS bytes. This
adapter does not discover framing in streams, remove flags/KISS escaping, or
infer the actual SatNOGS receiver boundary.
"""

from __future__ import annotations

from .beacon_contract import BeaconValidationError
from .decoder import decode_public_beacon
from .public_schema import PUBLIC_BEACON_PACKET_BYTES


# Literal flight header, corroborated by the 2026-09-23 Hydra capture.
# Both 0x41 address octets are intentional; do not walk generic extension bits.
SUNCET_AX25_HEADER = bytes.fromhex(
    "98 82 a6 a0 40 40 41 a6 aa 9c 62 40 40 41 03 f0"
)


def crc16_x25(data: bytes) -> int:
    """Return CRC-16/X-25 (reflected 0x1021, init/xorout 0xffff)."""

    crc = 0xFFFF
    for octet in data:
        crc ^= octet
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return crc ^ 0xFFFF


def decode_public_ax25_beacon(
    frame: bytes, *, fcs_present: bool
) -> dict[str, int | float | str]:
    """Decode exactly one header-bearing CTDB 2.0.5 beacon.

    ``fcs_present=False`` accepts 268 bytes (16 header + 252 CCSDS), matching
    the saved Hydra representation and the locally replayed SatNOGS GFSK/AX.25
    receiver output. ``True`` accepts 270 bytes and validates
    the final little-endian CRC-16/X-25 over the header and CCSDS packet.
    Both paths also pass the ten flight-unit beacons recovered from the
    2026-09-30 IQ recording; independent HDLC recovery retains their FCS.

    Both paths enforce the literal header and the CCSDS Fletcher-32 contract,
    then return only the 111 approved fields. Wrong addresses, other APIDs,
    truncation, extra bytes, and checksum failures raise BeaconValidationError.
    """

    if not isinstance(fcs_present, bool):
        raise TypeError("fcs_present must explicitly be True or False")
    payload_end = len(SUNCET_AX25_HEADER) + PUBLIC_BEACON_PACKET_BYTES
    expected_length = payload_end + (2 if fcs_present else 0)
    if len(frame) != expected_length:
        raise BeaconValidationError(
            f"SunCET AX.25 input has {len(frame)} bytes; declared FCS mode "
            f"requires {expected_length}"
        )
    if frame[:len(SUNCET_AX25_HEADER)] != SUNCET_AX25_HEADER:
        raise BeaconValidationError("Unexpected SunCET AX.25 downlink header")
    if fcs_present:
        stored = int.from_bytes(frame[payload_end:], "little")
        if stored != crc16_x25(frame[:payload_end]):
            raise BeaconValidationError("AX.25 CRC-16/X-25 mismatch")
    return decode_public_beacon(frame[len(SUNCET_AX25_HEADER):payload_end])
