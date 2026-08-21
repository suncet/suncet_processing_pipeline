"""Create a private review worksheet for the public APID 1 field selection.

The generated CSV is an internal review artifact, not a public definition. No
field is automatically approved for publication: ordinary fields start as
``REVIEW`` and command/uplink-related fields start as ``OMIT``. A reviewed,
minimal public schema will be produced separately from the approved rows.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


REVIEW_COLUMNS = (
    "source_row",
    "bit_offset",
    "byte_offset",
    "bit_in_byte",
    "bit_length",
    "data_type",
    "source_name",
    "short_description",
    "units",
    "conversion",
    "proposed_action",
    "public_name",
    "public_description",
    "reviewer_notes",
)

# Conservative triage only. Final decisions are made row by row by the mission.
_OPAQUE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:^|_)cmd(?:_|$)",
        r"command",
        r"uplink",
        r"(?:^|_)arm(?:ed|ing)?(?:_|$)",
    )
)


def _data_type_bits(data_type: str) -> int:
    match = re.fullmatch(r"[A-Za-z](\d+)", data_type.strip())
    if match is None:
        raise ValueError(f"unsupported CTDB DataType {data_type!r}")
    return int(match.group(1))


def _initial_action(row: dict[str, str]) -> str:
    searchable = " ".join(
        (
            row.get("ItemName", ""),
            row.get("ShortDescription", ""),
            row.get("LongDescription", ""),
        )
    )
    if any(pattern.search(searchable) for pattern in _OPAQUE_PATTERNS):
        return "OMIT"
    return "REVIEW"


def build_review_rows(
    source_rows: Iterable[dict[str, str]],
    *,
    apid: int = 1,
) -> list[dict[str, object]]:
    """Return ordered field-review rows for one CTDB APID."""

    review_rows: list[dict[str, object]] = []
    bit_offset = 0
    for source_row, row in enumerate(source_rows, start=2):
        try:
            row_apid = int(row.get("APID", ""))
        except (TypeError, ValueError):
            continue
        if row_apid != apid:
            continue

        data_type = row.get("DataType", "").strip()
        bit_length = _data_type_bits(data_type)
        review_rows.append(
            {
                "source_row": source_row,
                "bit_offset": bit_offset,
                "byte_offset": bit_offset // 8,
                "bit_in_byte": bit_offset % 8,
                "bit_length": bit_length,
                "data_type": data_type,
                "source_name": row.get("ItemName", "").strip(),
                "short_description": row.get("ShortDescription", "").strip(),
                "units": row.get("Units", "").strip(),
                "conversion": row.get("Conversion", "").strip(),
                "proposed_action": _initial_action(row),
                "public_name": "",
                "public_description": "",
                "reviewer_notes": "",
            }
        )
        bit_offset += bit_length
    return review_rows


def export_review_csv(source: Path, output: Path, *, apid: int = 1) -> tuple[int, int]:
    """Write a private review CSV and return ``(field_count, total_bits)``."""

    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = build_review_rows(csv.DictReader(stream), apid=apid)
    if not rows:
        raise ValueError(f"no APID {apid} rows found in {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total_bits = int(rows[-1]["bit_offset"]) + int(rows[-1]["bit_length"])
    return len(rows), total_bits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a private APID field-publication review worksheet."
    )
    parser.add_argument("--source", type=Path, required=True, help="private ct_tlm.csv")
    parser.add_argument("--output", type=Path, required=True, help="private review CSV")
    parser.add_argument("--apid", type=int, default=1)
    args = parser.parse_args(argv)

    field_count, total_bits = export_review_csv(args.source, args.output, apid=args.apid)
    print(
        f"Wrote {field_count} APID {args.apid} fields ({total_bits} bits) to "
        f"{args.output}. This is a private review artifact, not a public schema."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
