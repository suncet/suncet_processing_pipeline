"""Generate the public SunCET contribution for SatNOGS's Kaitai 0.10 toolchain.

Uses only the reviewed public schema and synthetic fixture. Compiled Python/JS
are build products in the upstream checkout, never contribution source files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml

from .ax25 import SUNCET_AX25_HEADER
from .kaitai_generator import generate_kaitai, PUBLIC_SPEC_URL
from .public_schema import load_public_beacon_schema
from .synthetic_fixture import build_synthetic_fixture


class _Literal(str):
    pass


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


_Dumper.add_representer(_Literal, lambda dumper, value: dumper.represent_scalar(
    "tag:yaml.org,2002:str", value, style="|"))


def generate_upstream_kaitai():
    core = yaml.safe_load(generate_kaitai())
    for key in ("meta", "doc-ref", "doc", "enums"):
        core.pop(key, None)
    for entry in core["seq"]:
        name = entry.get("id", "")
        if name.startswith("opaque_"):
            entry.pop("id")  # Compiler-generated _unnamed attributes stay private.
            if name == "opaque_fletcher32_checksum":
                entry["doc"] = "Already validated by the outer frame's Fletcher-32 check."
        if "enum" in entry:
            entry.pop("enum")  # Stable integer fields, including unknown state codes.
            if entry["type"] == "b1":
                entry["id"] = name + "_raw"
                core["instances"][name] = {"value": f"{name}_raw ? 1 : 0"}
        if entry.get("type") == "f4":
            # SatNOGS DB drops NaN fields, but does not drop +/-infinity. Normalize
            # all nonfinite F32 values to NaN so one sensor cannot reject a point.
            entry["id"] = name + "_raw"
            raw = entry["id"]
            core["instances"][name] = {"value": (
                f"{raw} >= -3.4028234663852886e38 and {raw} <= 3.4028234663852886e38 "
                f"? {raw} : {raw} - {raw}")}
    fields = load_public_beacon_schema()
    doc = ("SunCET public APID 1 beacon, CTDB 2.0.5.\n"
           "Input retains the literal AX.25 header; the receiver has removed flags and FCS.\n"
           "Fletcher-32 and the complete packet envelope are validated on construction.\n\n")
    doc += "\n".join(f":field {field.public_name}: beacon.{field.public_name}" for field in fields)
    # Kaitai 0.10 adds punctuation at the end of prose. Keep it off the last
    # :field path, or the upstream regex will silently omit that field.
    doc += ("\n\nState fields retain integer codes, including unknown values.\n"
            "Nonfinite floating-point readings become NaN for omission by SatNOGS DB.\n")
    root = {
        "meta": {"id": "suncet", "title": "SunCET public APID 1 beacon (CTDB 2.0.5)",
                 "endian": "be", "bit-endian": "be", "ks-version": "0.10"},
        "doc": _Literal(doc), "doc-ref": PUBLIC_SPEC_URL,
        "seq": [
            {"size": 0, "valid": {"expr": "_io.size == 268"}},
            {"contents": list(SUNCET_AX25_HEADER)},
            {"type": "fletcher_check"},
            # Force the otherwise-lazy instance so from_bytes itself rejects
            # wrong APIDs, lengths, and fine times, even before get_fields.
            {"size": 0, "valid": {"expr": "beacon.ccsds_apid == 1"}},
        ],
        "instances": {"beacon": {"pos": 16, "size": 252, "type": "public_beacon"}},
        "types": {
            "public_beacon": core,
            "fletcher_check": {
                "doc": "Little-endian input words, accumulators seeded 65535, big-endian stored checksum.",
                "seq": [
                    {"id": "words", "type": (
                        "fletcher_word(_index == 0 ? 65535 : words[_index - 1].sum1, "
                        "_index == 0 ? 65535 : words[_index - 1].sum2)"),
                     "repeat": "expr", "repeat-expr": 124},
                    # Multiplication avoids JavaScript's signed 32-bit bitwise OR.
                    {"type": "u4", "valid": {"expr": "_ == words[123].sum2 * 65536 + words[123].sum1"}},
                ],
            },
            "fletcher_word": {
                "params": [{"id": "previous_sum1", "type": "u4"},
                           {"id": "previous_sum2", "type": "u4"}],
                "seq": [{"id": "word", "type": "u2le"}],
                "instances": {"sum1": {"value": "(previous_sum1 + word) % 65535"},
                              "sum2": {"value": "(previous_sum2 + sum1) % 65535"}},
            },
        },
    }
    return yaml.dump(root, Dumper=_Dumper, sort_keys=False, width=170, explicit_start=True)


def upstream_expected(expected):
    """Convert the mission fixture's enum labels to stable upstream wire codes."""
    result = dict(expected)
    for field in load_public_beacon_schema():
        states = {label: int(code) for code, label in re.findall(
            r"(\d+)/([A-Z][A-Z0-9_]*)", field.conversion_or_status_map)}
        if states and isinstance(result[field.public_name], str):
            result[field.public_name] = states[result[field.public_name]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("upstream"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    packet, expected = build_synthetic_fixture()
    artifacts = {
        "ksy/suncet.ksy": generate_upstream_kaitai(),
        "contrib/suncet/synthetic.hex": (SUNCET_AX25_HEADER + packet).hex() + "\n",
        "contrib/suncet/synthetic_expected.json": json.dumps(
            upstream_expected({name: value["engineering"] for name, value in expected.items()}), indent=2, sort_keys=True, allow_nan=False) + "\n",
    }
    for name, contents in artifacts.items():
        destination = args.output_dir / name
        if args.check:
            if not destination.is_file() or destination.read_text() != contents:
                parser.exit(1, f"Stale upstream artifact: {name}\n")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents)
    print(f"{'Verified' if args.check else 'Wrote'} {len(artifacts)} upstream artifacts")


if __name__ == "__main__":
    main()
