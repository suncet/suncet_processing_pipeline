"""Reviewed public field schema for the SunCET APID 1 beacon.

Only mission-approved public fields are present. Omitted fields remain gaps in
the bit layout so a future SatNOGS decoder can consume them opaquely without
publishing private command or uplink telemetry.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from importlib.resources import files


PUBLIC_BEACON_FIELD_COUNT = 112
_TYPE_PATTERN = re.compile(r"^[A-Za-z](\d+)$")


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
    """Load and validate the reviewed, dependency-free public field table."""

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
    previous_end = 0
    for field in fields:
        match = _TYPE_PATTERN.fullmatch(field.data_type)
        if match is None or int(match.group(1)) != field.bit_length:
            raise ValueError(
                f"{field.public_name}: {field.data_type} does not match "
                f"{field.bit_length} bits"
            )
        if field.byte_offset != field.bit_offset // 8:
            raise ValueError(f"{field.public_name}: byte offset is inconsistent")
        if field.bit_in_byte != field.bit_offset % 8:
            raise ValueError(f"{field.public_name}: bit-in-byte is inconsistent")
        if field.bit_offset < previous_end:
            raise ValueError(f"{field.public_name}: fields overlap or are unordered")
        if not field.public_name or not field.description:
            raise ValueError("all public fields require a name and description")
        if field.public_name in names:
            raise ValueError(f"duplicate public field name {field.public_name!r}")
        names.add(field.public_name)
        previous_end = field.bit_offset + field.bit_length

    return fields
