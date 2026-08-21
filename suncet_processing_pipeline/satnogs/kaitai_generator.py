"""Generate the provisional public SunCET APID 1 Kaitai definition.

The reviewed CSV is the authoritative public-field list.  This generator keeps
the Kaitai skeleton synchronized with that list while replacing every excluded
region with anonymous padding.  Link-layer framing remains intentionally out of
scope until the flight AX.25 construction and SatNOGS receiver output are
confirmed.
"""

from __future__ import annotations

import re
from pathlib import Path

from .public_schema import PublicBeaconField, load_public_beacon_schema


PUBLIC_SPEC_URL = (
    "https://github.com/suncet/suncet_processing_pipeline/blob/main/"
    "docs/SUNCET_PUBLIC_BEACON_SPEC.md"
)
CTDB_PACKET_BITS = 2008
CHECKSUM_BITS = 32
PUBLIC_PAYLOAD_BITS = CTDB_PACKET_BITS - CHECKSUM_BITS

_COEFFICIENT_PATTERN = re.compile(
    r"C(?P<degree>\d+)=(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_ENUM_PATTERN = re.compile(r"(?P<value>-?\d+)/(?P<label>[A-Za-z0-9_+-]+)")


def _indent_block(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else prefix for line in lines]


def _doc_lines(field: PublicBeaconField) -> list[str]:
    lines = [field.description, f"Engineering units: {field.unit}"]
    if field.conversion_or_status_map:
        lines.append(f"Conversion/status map: {field.conversion_or_status_map}")
    return lines


def _base_type(field: PublicBeaconField) -> str:
    prefix = field.data_type[0]
    bits = field.bit_length
    if bits not in (8, 16, 32) or field.bit_in_byte:
        return f"b{bits}"
    if prefix == "F":
        return f"f{bits // 8}"
    if prefix == "I":
        return f"s{bits // 8}"
    return f"u{bits // 8}"


def _coefficients(field: PublicBeaconField) -> list[float]:
    matches = list(_COEFFICIENT_PATTERN.finditer(field.conversion_or_status_map))
    if not matches:
        return []
    by_degree = {int(match["degree"]): float(match["value"]) for match in matches}
    if set(by_degree) != set(range(max(by_degree) + 1)):
        raise ValueError(f"{field.public_name}: non-contiguous polynomial coefficients")
    return [by_degree[index] for index in range(max(by_degree) + 1)]


def _enum_values(field: PublicBeaconField) -> list[tuple[int, str]]:
    return [
        (int(match["value"]), match["label"].lower())
        for match in _ENUM_PATTERN.finditer(field.conversion_or_status_map)
    ]


def _number(value: float) -> str:
    return format(value, ".15g")


def _polynomial_expression(raw_name: str, coefficients: list[float]) -> str:
    expression = _number(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        expression = f"({_number(coefficient)} + {raw_name} * {expression})"
    return expression


def _opaque_entries(start_bit: int, bit_length: int, index: int) -> list[str]:
    """Return byte/bit entries that consume one excluded region."""

    if start_bit % 8:
        raise ValueError(f"opaque region {index} starts at non-byte boundary {start_bit}")
    entries: list[str] = []
    byte_count, remaining_bits = divmod(bit_length, 8)
    if byte_count:
        entries.extend(
            [
                f"  - id: opaque_{index}_bytes",
                f"    size: {byte_count}",
                "    doc: Excluded public-beacon fields, consumed opaquely.",
            ]
        )
    if remaining_bits:
        entries.extend(
            [
                f"  - id: opaque_{index}_bits",
                f"    type: b{remaining_bits}",
                "    doc: Excluded public-beacon bits, consumed opaquely.",
            ]
        )
    return entries


def generate_kaitai() -> str:
    """Return the generated ``suncet_apid1.ksy`` contents."""

    fields = load_public_beacon_schema()
    lines = [
        "meta:",
        "  id: suncet_apid1",
        "  title: SunCET public CCSDS APID 1 beacon (provisional)",
        "  endian: be",
        "  bit-endian: be",
        "doc-ref: |",
        f"  {PUBLIC_SPEC_URL}",
        "doc: |",
        "  Provisional bare-CCSDS decoder for the public SunCET APID 1 beacon.",
        "  The AX.25 wrapper will be added after flight framing is confirmed.",
        "",
    ]
    for field in fields:
        lines.append(f"  :field {field.public_name}: {field.public_name}")

    lines.append("seq:")
    instances: list[str] = []
    enums: list[str] = []
    cursor = 0
    opaque_index = 0

    header_instances = {
        "ccsds_version": "(ccsds_primary_word >> 13) & 7",
        "ccsds_packet_type": "(ccsds_primary_word >> 12) & 1",
        "ccsds_secondary_header_flag": "(ccsds_primary_word >> 11) & 1",
        "ccsds_apid": "ccsds_primary_word & 2047",
        "ccsds_sequence_flags": "(ccsds_sequence_word >> 14) & 3",
        "ccsds_sequence_count": "ccsds_sequence_word & 16383",
    }

    for field_index, field in enumerate(fields):
        if field_index == 0:
            lines.extend(
                [
                    "  - id: ccsds_primary_word",
                    "    type: u2",
                    "    valid: 2049",
                    "    doc: |",
                    "      Packed CCSDS version/type/secondary-header/APID word.",
                    "      The only accepted value describes telemetry APID 1 with",
                    "      the SunCET secondary header present.",
                ]
            )
        elif field_index == 4:
            lines.extend(
                [
                    "  - id: ccsds_sequence_word",
                    "    type: u2",
                    "    doc: Packed CCSDS sequence flags and sequence counter.",
                ]
            )

        if field.public_name in header_instances:
            instances.extend(
                [
                    f"  {field.public_name}:",
                    f"    value: {header_instances[field.public_name]}",
                    "    doc: |",
                    *_indent_block(_doc_lines(field), 6),
                ]
            )
            cursor = field.bit_offset + field.bit_length
            continue

        if field.bit_offset > cursor:
            opaque_index += 1
            lines.extend(
                _opaque_entries(cursor, field.bit_offset - cursor, opaque_index)
            )
            cursor = field.bit_offset

        coefficients = _coefficients(field)
        enum_values = _enum_values(field)
        raw_name = f"{field.public_name}_raw" if coefficients else field.public_name
        lines.extend([f"  - id: {raw_name}", f"    type: {_base_type(field)}"])

        if enum_values:
            enum_name = f"{field.public_name}_values"
            lines.append(f"    enum: {enum_name}")
            enums.extend([f"  {enum_name}:"])
            for value, label in enum_values:
                # Quote labels so YAML 1.1 parsers do not reinterpret values
                # such as ``on``, ``off``, ``yes``, and ``no`` as booleans.
                enums.append(f"    {value}: '{label}'")

        if field.public_name == "ccsds_packet_length_field":
            lines.extend(
                [
                    "    valid:",
                    "      expr: _ == _io.size - 7",
                ]
            )

        lines.append("    doc: |")
        lines.extend(_indent_block(_doc_lines(field), 6))

        if coefficients:
            instances.extend(
                [
                    f"  {field.public_name}:",
                    f"    value: {_polynomial_expression(raw_name, coefficients)}",
                    "    doc: |",
                    *_indent_block(_doc_lines(field), 6),
                ]
            )

        cursor = field.bit_offset + field.bit_length

    if cursor < PUBLIC_PAYLOAD_BITS:
        opaque_index += 1
        lines.extend(_opaque_entries(cursor, PUBLIC_PAYLOAD_BITS - cursor, opaque_index))

    lines.extend(
        [
            "  - id: provisional_extra_byte",
            "    type: u1",
            "    if: _io.size == 252",
            "    doc: |",
            "      Apparent additional pre-checksum byte in current 252-byte captures.",
            "      Its flight definition is unresolved and it is not public telemetry.",
            "  - id: opaque_fletcher32_checksum",
            "    size: 4",
            "    doc: Fletcher-32 bytes consumed for framing; not exposed as telemetry.",
        ]
    )

    if instances:
        lines.append("instances:")
        lines.extend(instances)
    if enums:
        lines.append("enums:")
        lines.extend(enums)
    return "\n".join(lines) + "\n"


def main() -> int:
    output = Path(__file__).with_name("suncet_apid1.ksy")
    output.write_text(generate_kaitai(), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
