"""Build a deterministic, non-flight APID 1 interoperability fixture."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from .beacon_contract import parse_beacon_packet, suncet_fletcher32
from .public_schema import PublicBeaconField, load_public_beacon_schema


_COEFFICIENT_PATTERN = re.compile(
    r"C(?P<degree>\d+)=(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_ENUM_PATTERN = re.compile(r"(?P<value>-?\d+)/(?P<label>[A-Za-z0-9_+-]+)")


def _set_bits(packet: bytearray, offset: int, length: int, value: int) -> None:
    if value < 0:
        value += 1 << length
    if not 0 <= value < (1 << length):
        raise ValueError(f"value {value} does not fit in {length} bits")
    for index in range(length):
        bit = (value >> (length - index - 1)) & 1
        absolute = offset + index
        mask = 1 << (7 - absolute % 8)
        if bit:
            packet[absolute // 8] |= mask
        else:
            packet[absolute // 8] &= ~mask


def _candidate_raw_value(field: PublicBeaconField, index: int) -> int | float:
    fixed = {
        "ccsds_version": 0,
        "ccsds_packet_type": 0,
        "ccsds_secondary_header_flag": 1,
        "ccsds_apid": 1,
        "ccsds_sequence_flags": 3,
        "ccsds_sequence_count": 42,
        "ccsds_packet_length_field": 244,
        "spacecraft_time_seconds_since_2000": 833_326_475,
        "spacecraft_time_fine_raw": 1_234,
        "mode_system_mode": 2,
        "dsps_flare_phase": 40,
        "uhf_alive": 1,
        "adcs_sun_point_state": 6,
    }
    if field.public_name in fixed:
        return fixed[field.public_name]
    if field.data_type == "F32":
        return 12.5

    enum_values = [
        int(match["value"])
        for match in _ENUM_PATTERN.finditer(field.conversion_or_status_map)
    ]
    if enum_values:
        return enum_values[-1]

    maximum = (1 << field.bit_length) - 1
    magnitude = (index * 37 + 11) & maximum
    if field.data_type.startswith("I"):
        signed_maximum = (1 << (field.bit_length - 1)) - 1
        magnitude = max(1, magnitude & signed_maximum)
        return -magnitude if index % 2 else magnitude
    return magnitude


def _raw_bits(field: PublicBeaconField, value: int | float) -> int:
    if field.data_type == "F32":
        return int.from_bytes(struct.pack(">f", float(value)), "big")
    return int(value)


def _engineering_value(field: PublicBeaconField, raw: int | float) -> object:
    coefficients = {
        int(match["degree"]): float(match["value"])
        for match in _COEFFICIENT_PATTERN.finditer(field.conversion_or_status_map)
    }
    if coefficients:
        result = 0.0
        for degree, coefficient in coefficients.items():
            result += coefficient * float(raw) ** degree
        return result

    statuses = {
        int(match["value"]): match["label"]
        for match in _ENUM_PATTERN.finditer(field.conversion_or_status_map)
    }
    if statuses and int(raw) in statuses:
        return statuses[int(raw)]
    return raw


def build_synthetic_fixture() -> tuple[bytes, dict[str, dict[str, object]]]:
    """Return a 251-byte synthetic packet and independently expected values."""

    packet = bytearray(251)
    expected: dict[str, dict[str, object]] = {}
    for index, field in enumerate(load_public_beacon_schema()):
        raw = _candidate_raw_value(field, index)
        _set_bits(packet, field.bit_offset, field.bit_length, _raw_bits(field, raw))
        expected[field.public_name] = {
            "raw": raw,
            "engineering": _engineering_value(field, raw),
        }

    checksum = suncet_fletcher32(bytes(packet[:-4]))
    packet[-4:] = checksum.to_bytes(4, "big")
    parse_beacon_packet(bytes(packet), accepted_lengths={251})
    return bytes(packet), expected


def main() -> int:
    output_dir = Path(__file__).with_name("test_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    packet, expected = build_synthetic_fixture()
    (output_dir / "suncet_apid1_synthetic_251.hex").write_text(
        packet.hex() + "\n", encoding="ascii"
    )
    (output_dir / "suncet_apid1_synthetic_251_expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote synthetic fixture files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
