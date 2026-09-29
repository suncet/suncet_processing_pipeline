"""Strict, public-facing packet contract for the SunCET APID 1 beacon.

This module implements the current CTDB 2.0.5 packet boundary. It validates the
CCSDS packet and the mission Fletcher-32 checksum. The secondary time header is
coarse seconds since 2000-01-01T00:00:00Z plus an integer 0-999 milliseconds
after the coarse second. Only the current 252-byte packet is accepted.

It contains no private CTDB definitions and can serve as an independent oracle
for the SatNOGS Kaitai decoder and RF validation fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

from suncet_processing_pipeline.spacecraft_time import (
    FINE_MILLISECONDS_MAX,
    combine_spacecraft_time_seconds,
)
from .public_schema import (
    PUBLIC_BEACON_CHECKSUM_BYTES,
    PUBLIC_BEACON_CTDB_VERSION,
    PUBLIC_BEACON_PACKET_BYTES,
)


BEACON_APID = 1
CCSDS_PRIMARY_HEADER_BYTES = 6
CCSDS_SECONDARY_TIME_BYTES = 6
FLETCHER32_BYTES = PUBLIC_BEACON_CHECKSUM_BYTES

CURRENT_BEACON_PACKET_LENGTHS = frozenset({PUBLIC_BEACON_PACKET_BYTES})


class BeaconValidationError(ValueError):
    """Raised when bytes do not satisfy the current public beacon contract."""


@dataclass(frozen=True)
class BeaconPacket:
    """Validated APID 1 packet metadata that is safe to expose publicly."""

    raw: bytes
    sequence_flags: int
    sequence_count: int
    coarse_seconds: int
    fine_milliseconds: int
    checksum: int

    @property
    def packet_length(self) -> int:
        return len(self.raw)

    @property
    def timestamp_seconds(self) -> float:
        """Seconds since 2000-01-01T00:00:00Z, including milliseconds."""

        return combine_spacecraft_time_seconds(
            self.coarse_seconds, self.fine_milliseconds
        )

    @property
    def fine_time_raw(self) -> int:
        """Compatibility alias for callers predating the resolved wire unit."""

        return self.fine_milliseconds


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


def parse_beacon_packet(packet: bytes) -> BeaconPacket:
    """Validate and expose the stable envelope fields of one APID 1 packet.

    The function exposes combined epoch seconds but deliberately does not apply
    a leap-second policy or format a UTC string.
    """

    if len(packet) != PUBLIC_BEACON_PACKET_BYTES:
        raise BeaconValidationError(
            f"APID 1 packet has {len(packet)} bytes; CTDB "
            f"{PUBLIC_BEACON_CTDB_VERSION} requires {PUBLIC_BEACON_PACKET_BYTES}"
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

    stored_checksum = int.from_bytes(packet[-FLETCHER32_BYTES:], "big")
    calculated_checksum = suncet_fletcher32(packet[:-FLETCHER32_BYTES])
    if stored_checksum != calculated_checksum:
        raise BeaconValidationError(
            "Fletcher-32 mismatch: "
            f"stored 0x{stored_checksum:08x}, calculated 0x{calculated_checksum:08x}"
        )

    fine_milliseconds = int.from_bytes(packet[10:12], "big")
    if fine_milliseconds > FINE_MILLISECONDS_MAX:
        raise BeaconValidationError(
            f"fine spacecraft time is {fine_milliseconds} ms; expected 0-999 ms"
        )

    sequence_word = int.from_bytes(packet[2:4], "big")
    return BeaconPacket(
        raw=bytes(packet),
        sequence_flags=(sequence_word >> 14) & 0x03,
        sequence_count=sequence_word & 0x3FFF,
        coarse_seconds=int.from_bytes(packet[6:10], "big"),
        fine_milliseconds=fine_milliseconds,
        checksum=stored_checksum,
    )
