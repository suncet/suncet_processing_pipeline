"""Checksum-validated public decoding for the current SunCET APID 1 packet.

The generated Kaitai parser describes the public wire layout. This module is
the integration entrypoint: it first enforces the complete packet contract,
including Fletcher-32, and returns only reviewed public engineering values.
It accepts bare CCSDS packets; receiver-specific AX.25 framing is upstream.
"""

from __future__ import annotations

from enum import Enum
from io import BytesIO

from kaitaistruct import KaitaiStream

from .beacon_contract import BeaconValidationError, parse_beacon_packet
from .generated_suncet_apid1 import SuncetApid1
from .public_schema import load_public_beacon_schema


def decode_public_beacon(packet: bytes) -> dict[str, int | float | str]:
    """Decode one valid CTDB 2.0.5 packet into its approved public fields.

    Known enumerations are their public uppercase labels; unknown values are
    retained as integers rather than assigned a misleading state. Scaled
    values are in the schema's engineering units. Private/opaque regions and
    the stored checksum are never included in the result. Spacecraft time is
    exposed in its two wire fields without inventing a UTC/leap-second policy.

    Raises ``BeaconValidationError`` for invalid packet framing or checksum.
    """

    validated = parse_beacon_packet(packet)
    stream = KaitaiStream(BytesIO(validated.raw))
    parsed = SuncetApid1(stream)
    if not stream.is_eof():
        raise BeaconValidationError("Public decoder did not consume the complete packet")

    values: dict[str, int | float | str] = {}
    for field in load_public_beacon_schema():
        value = getattr(parsed, field.public_name)
        if isinstance(value, Enum):
            value = value.name.upper()
        values[field.public_name] = value
    return values
