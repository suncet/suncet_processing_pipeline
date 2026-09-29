"""Reviewed public field schema for the SunCET CTDB 2.0.5 APID 1 beacon.

Only mission-approved public fields are present. Omitted fields remain gaps in
the bit layout so the SatNOGS decoder can consume them opaquely without
publishing private command or uplink telemetry.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from importlib.resources import files


PUBLIC_BEACON_CTDB_VERSION = "2.0.5"
PUBLIC_BEACON_PACKET_BYTES = 252
PUBLIC_BEACON_CHECKSUM_BYTES = 4
PUBLIC_BEACON_FIELD_COUNT = 111
_TYPE_PATTERN = re.compile(r"^([UIDF])([1-9]\d*)$")
_PUBLIC_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class PublicBeaconField:
    """One approved public field at its authoritative APID 1 bit offset."""

    category: str
    byte_offset: int
    bit_offset: int
    bit_in_byte: int
    bit_length: int
    data_type: str
    source_field: str
    public_name: str
    description: str
    unit: str
    conversion_or_status_map: str


def load_public_beacon_schema() -> tuple[PublicBeaconField, ...]:
    """Load and validate the approved CTDB 2.0.5 fields.

    The source export supplies offsets, types, and conversions. Its Units
    column is blank; the table retains the previously reviewed public unit
    annotations and descriptions, including units explicitly marked inferred.
    No fields from older CTDB layouts or additional unreviewed fields are added.
    """

    resource = files(__package__).joinpath("public_beacon_schema.csv")
    with resource.open(newline="", encoding="utf-8") as stream:
        fields = tuple(
            PublicBeaconField(
                category=row["category"],
                byte_offset=int(row["byte_offset"]),
                bit_offset=int(row["bit_offset"]),
                bit_in_byte=int(row["bit_in_byte"]),
                bit_length=int(row["bit_length"]),
                data_type=row["data_type"],
                source_field=row["source_field"],
                public_name=row["public_name"],
                description=row["description"],
                unit=row["unit"],
                conversion_or_status_map=row["conversion_or_status_map"],
            )
            for row in csv.DictReader(stream)
        )

    if len(fields) != PUBLIC_BEACON_FIELD_COUNT:
        raise ValueError(
            f"expected {PUBLIC_BEACON_FIELD_COUNT} public beacon fields, "
            f"found {len(fields)}"
        )

    names: set[str] = set()
    source_names: set[str] = set()
    payload_bits = (PUBLIC_BEACON_PACKET_BYTES - PUBLIC_BEACON_CHECKSUM_BYTES) * 8
    previous_end = 0
    for field in fields:
        match = _TYPE_PATTERN.fullmatch(field.data_type)
        if match is None or int(match.group(2)) != field.bit_length:
            raise ValueError(
                f"{field.public_name}: {field.data_type} does not match "
                f"{field.bit_length} bits"
            )
        prefix = match.group(1)
        if field.bit_length > 64:
            raise ValueError(f"{field.public_name}: field is wider than 64 bits")
        if prefix == "F" and field.bit_length not in {32, 64}:
            raise ValueError(f"{field.public_name}: unsupported floating-point width")
        if prefix in {"I", "F"} and (
            field.bit_length not in {8, 16, 32, 64} or field.bit_in_byte != 0
        ):
            raise ValueError(
                f"{field.public_name}: signed and floating-point fields "
                "must occupy whole bytes and be byte aligned"
            )
        if field.bit_offset < 0 or field.bit_offset + field.bit_length > payload_bits:
            raise ValueError(
                f"{field.public_name}: field falls outside the CTDB "
                f"{PUBLIC_BEACON_CTDB_VERSION} payload before its checksum"
            )
        if field.byte_offset != field.bit_offset // 8:
            raise ValueError(f"{field.public_name}: byte offset is inconsistent")
        if field.bit_in_byte != field.bit_offset % 8:
            raise ValueError(f"{field.public_name}: bit-in-byte is inconsistent")
        if field.bit_offset < previous_end:
            raise ValueError(f"{field.public_name}: fields overlap or are unordered")
        if (
            not _PUBLIC_NAME_PATTERN.fullmatch(field.public_name)
            or not field.description
        ):
            raise ValueError("all public fields require a name and description")
        if field.public_name in names:
            raise ValueError(f"duplicate public field name {field.public_name!r}")
        names.add(field.public_name)
        if not field.source_field or field.source_field in source_names:
            raise ValueError("all public fields require unique source field names")
        source_names.add(field.source_field)
        previous_end = field.bit_offset + field.bit_length

    return fields
