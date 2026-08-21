"""Strict, public-facing packet contract for the SunCET APID 1 beacon.

This module intentionally stops at the boundaries that are still awaiting
flight-software or RF confirmation. It validates the CCSDS packet and the
mission Fletcher-32 checksum. FSW has confirmed that the secondary time header
means coarse seconds since 2000-01-01T00:00:00Z plus microseconds after the
coarse second, but the current 16-bit field cannot literally cover 0-999999.
The module therefore exposes fine time raw until its serialization is resolved,
and temporarily accepts both the CTDB-declared and flight-data-observed packet
lengths.

It contains no private CTDB definitions and can serve as an independent oracle
for the future SatNOGS Kaitai decoder and RF validation fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection


BEACON_APID = 1
CCSDS_PRIMARY_HEADER_BYTES = 6
CCSDS_SECONDARY_TIME_BYTES = 6
FLETCHER32_BYTES = 4

# CTDB 2.0.1 declares 251 bytes; checksum-valid flight-model captures declare
# 252. Collapse this to one value when FSW identifies the apparent spare byte.
CURRENT_BEACON_PACKET_LENGTHS = frozenset({251, 252})


class BeaconValidationError(ValueError):
    """Raised when bytes do not satisfy the current public beacon contract."""


@dataclass(frozen=True)
class BeaconPacket:
    """Validated APID 1 packet metadata that is safe to expose publicly."""

    raw: bytes
    sequence_flags: int
    sequence_count: int
    coarse_seconds: int
    fine_time_raw: int
    checksum: int

    @property
    def packet_length(self) -> int:
        return len(self.raw)


def suncet_fletcher32(data: bytes) -> int:
    """Return SunCET's Fletcher-32 over little-endian 16-bit input words.

    Both accumulators start at ``0xffff``. An odd trailing input byte is padded
    with zero for checksum calculation. The returned integer is serialized big
    endian in APID 1 packets.
    """

    if len(data) % 2:
        data += b"\x00"

    sum1 = 0xFFFF
    sum2 = 0xFFFF
    for offset in range(0, len(data), 2):
        word = data[offset] | (data[offset + 1] << 8)
        sum1 = (sum1 + word) % 0xFFFF
        sum2 = (sum2 + sum1) % 0xFFFF
    return (sum2 << 16) | sum1


def parse_beacon_packet(
    packet: bytes,
    *,
    accepted_lengths: Collection[int] = CURRENT_BEACON_PACKET_LENGTHS,
) -> BeaconPacket:
    """Validate and expose the stable envelope fields of one APID 1 packet.

    ``accepted_lengths`` is explicit so a flight-confirmed length can be used
    immediately by a caller before the repository-wide default is updated.
    The function does not convert spacecraft time to UTC while the confirmed
    microsecond meaning conflicts with the current 16-bit field width.
    """

    minimum_length = (
        CCSDS_PRIMARY_HEADER_BYTES
        + CCSDS_SECONDARY_TIME_BYTES
        + FLETCHER32_BYTES
    )
    if len(packet) < minimum_length:
        raise BeaconValidationError(
            f"packet has {len(packet)} bytes; at least {minimum_length} are required"
        )

    first_word = int.from_bytes(packet[0:2], "big")
    version = (first_word >> 13) & 0x07
    packet_type = (first_word >> 12) & 0x01
    secondary_header_flag = (first_word >> 11) & 0x01
    apid = first_word & 0x07FF

    if version != 0:
        raise BeaconValidationError(f"CCSDS version is {version}, expected 0")
    if packet_type != 0:
        raise BeaconValidationError("CCSDS packet is a command, not telemetry")
    if secondary_header_flag != 1:
        raise BeaconValidationError("CCSDS secondary time header is absent")
    if apid != BEACON_APID:
        raise BeaconValidationError(f"APID is {apid}, expected {BEACON_APID}")

    declared_length = int.from_bytes(packet[4:6], "big") + 7
    if declared_length != len(packet):
        raise BeaconValidationError(
            f"CCSDS header declares {declared_length} bytes, received {len(packet)}"
        )

    allowed = frozenset(accepted_lengths)
    if len(packet) not in allowed:
        expected = ", ".join(str(length) for length in sorted(allowed)) or "none"
        raise BeaconValidationError(
            f"APID 1 packet has {len(packet)} bytes; accepted lengths are {expected}"
        )

    stored_checksum = int.from_bytes(packet[-FLETCHER32_BYTES:], "big")
    calculated_checksum = suncet_fletcher32(packet[:-FLETCHER32_BYTES])
    if stored_checksum != calculated_checksum:
        raise BeaconValidationError(
            "Fletcher-32 mismatch: "
            f"stored 0x{stored_checksum:08x}, calculated 0x{calculated_checksum:08x}"
        )

    sequence_word = int.from_bytes(packet[2:4], "big")
    return BeaconPacket(
        raw=bytes(packet),
        sequence_flags=(sequence_word >> 14) & 0x03,
        sequence_count=sequence_word & 0x3FFF,
        coarse_seconds=int.from_bytes(packet[6:10], "big"),
        fine_time_raw=int.from_bytes(packet[10:12], "big"),
        checksum=stored_checksum,
    )
