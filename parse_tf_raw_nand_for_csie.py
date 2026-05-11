#!/usr/bin/env python3
"""
Strip SunCET **transfer-frame** wrappers and concatenate per-frame **payload** bytes for CSIE EM
tools (same role as ``parse_tf_assume_raw_nand.py``).

From this folder, **``python3 parse_tf_raw_nand_for_csie.py``** alone uses the FM2 raw file and
writes **``parsed_output_raw_csie_12p5mbps_4sps``** (same as explicit ``-o``).

**FM2 / fixed-Bluefin firmware layout** (default): each **2056**-byte TF is still::

    [4 ASM sync][6 TF primary][2 DFS][2040 payload][4 Fletcher]

The payload is **4 bytes shorter** than the legacy **2044**-byte data field; the trailing **4**
bytes are reserved for Fletcher-32; when wired, the placeholder is expected to be
``0x55 0x55 0x55 0x55``. This script **always** writes only the **2040**-byte payload (the tail
is never concatenated). Use **``--check-fletcher``** once tails match that placeholder.

The old **2044**-byte slice included the **4**-byte Fletcher placeholder as if it were part of the
CCSDS payload, which **shifts** every following byte in the concatenated output. FM2 mode takes
**2040** bytes and **does not** write the trailing ``0x55``×4.

Use **``--legacy-2044``** for older captures with no trailing Fletcher slot (matches
``parse_tf_assume_raw_nand.py``).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

SYNC_MARKER = b"\x1A\xCF\xFC\x1D"
SYNC_SIZE = 4
TF_HEADER_SIZE = 6
DFS_SIZE = 2
FRAME_SIZE = 2056

# Legacy: entire region after TF+DFS was payload (2044 bytes).
LEGACY_PAYLOAD_LEN = 2044

# FM2: 4-byte Fletcher placeholder after 2040-byte payload (still 2056-byte TF).
FM2_PAYLOAD_LEN = 2044 - 4
FLETCHER_PLACEHOLDER_LEN = 4
FLETCHER_PLACEHOLDER = b"\x55\x55\x55\x55"


def payload_offset(*, tf_header: int, dfs: int) -> int:
    return SYNC_SIZE + tf_header + dfs


def inspect_sync_layout(
    path: Path, *, sync: bytes = SYNC_MARKER, limit_gaps: int = 500_000
) -> None:
    data = path.read_bytes()
    positions = [m.start() for m in re.finditer(re.escape(sync), data)]
    n = len(positions)
    print(f"File: {path}  ({len(data)} bytes)")
    print(f"Sync occurrences ({sync.hex().upper()}): {n}")
    if n == 0:
        print("No sync markers at expected ASM.")
        return
    print(f"First sync offsets (up to 8): {positions[:8]}")
    if n < 2:
        print("Need at least two syncs to infer spacing.")
        return
    gaps = []
    for a, b in zip(positions, positions[1:]):
        gaps.append(b - a)
        if len(gaps) >= limit_gaps:
            break
    ctr = Counter(gaps)
    print(f"Inter-sync distance histogram (top 12 of {len(gaps)} gaps):")
    for dist, cnt in ctr.most_common(12):
        print(f"  {cnt:8d}  gap={dist} bytes")
    print(
        f"Total file length mod {FRAME_SIZE} = {len(data) % FRAME_SIZE} "
        f"(0 => strict {FRAME_SIZE}-byte TF tiling from offset 0)"
    )
    if positions[0] != 0:
        print(f"Note: first sync at offset {positions[0]} (leading prefix before first TF).")


def extract_raw_nand(
    input_file: Path,
    output_file: Path,
    *,
    skip_prefix: int,
    legacy_2044: bool,
    check_fletcher_placeholder: bool,
    skip_all_ff_payloads: bool,
) -> None:
    p0 = payload_offset(tf_header=TF_HEADER_SIZE, dfs=DFS_SIZE)
    if legacy_2044:
        payload_len = LEGACY_PAYLOAD_LEN
        tail_len = 0
    else:
        payload_len = FM2_PAYLOAD_LEN
        tail_len = FLETCHER_PLACEHOLDER_LEN

    if p0 + payload_len + tail_len != FRAME_SIZE:
        raise SystemExit(
            f"Internal layout error: {p0}+{payload_len}+{tail_len} != {FRAME_SIZE}"
        )

    data = input_file.read_bytes()
    if skip_prefix:
        if skip_prefix > len(data):
            raise ValueError("--skip-prefix larger than file")
        data = data[skip_prefix:]

    if len(data) % FRAME_SIZE != 0:
        raise ValueError(
            f"After skip-prefix, size {len(data)} is not a multiple of {FRAME_SIZE}. "
            "Try --inspect or adjust --skip-prefix."
        )

    frame_count = len(data) // FRAME_SIZE
    kept_frames = 0
    skipped_ff_frames = 0
    bad_fletcher = 0

    with output_file.open("wb") as fout:
        for i in range(frame_count):
            start = i * FRAME_SIZE
            frame = data[start : start + FRAME_SIZE]

            if frame[:SYNC_SIZE] != SYNC_MARKER:
                raise ValueError(
                    f"Bad sync at frame {i}, offset {skip_prefix + start}: "
                    f"got {frame[:SYNC_SIZE].hex().upper()}"
                )

            chunk = frame[p0 : p0 + payload_len]
            if tail_len and check_fletcher_placeholder:
                tail = frame[p0 + payload_len : p0 + payload_len + tail_len]
                if tail != FLETCHER_PLACEHOLDER:
                    bad_fletcher += 1

            if skip_all_ff_payloads and all(b == 0xFF for b in chunk):
                skipped_ff_frames += 1
                continue

            fout.write(chunk)
            kept_frames += 1

    print(f"Input file:         {input_file}")
    print(f"Output file:        {output_file}")
    print(f"Mode:               {'legacy 2044-byte payload' if legacy_2044 else 'FM2 2040-byte + 4 tail (not written)'}")
    print(f"Skip prefix:        {skip_prefix} bytes")
    print(f"Total frames:       {frame_count}")
    print(f"Kept frames:        {kept_frames}")
    print(f"Skipped all-0xFF:   {skipped_ff_frames}")
    if not legacy_2044 and check_fletcher_placeholder and bad_fletcher:
        print(
            f"Non-0x55555555 Fletcher tail: {bad_fletcher} frame(s) "
            f"(payload still written; omit --check-fletcher to silence)"
        )
    print(f"Bytes written:      {kept_frames * payload_len}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Concatenate TF NAND payloads (FM2: 2040 B + ignored Fletcher tail; legacy: 2044 B)"
    )
    p.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("CSIEtestpattern96_12p5Mbps_4sps_FM2"),
        help="Raw binary (default: CSIEtestpattern96_12p5Mbps_4sps_FM2 when run from this folder)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("parsed_output_raw_csie_12p5mbps_4sps"),
        help="Output binary (default: parsed_output_raw_csie_12p5mbps_4sps)",
    )
    p.add_argument(
        "--legacy-2044",
        action="store_true",
        help="Old capture layout: 2044-byte payload per TF, no 4-byte Fletcher tail",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print sync spacing and file length vs TF size (no output written)",
    )
    p.add_argument(
        "--skip-prefix",
        type=int,
        default=0,
        metavar="N",
        help="Ignore first N bytes before strict 2056-byte TF tiling",
    )
    p.add_argument(
        "--check-fletcher",
        action="store_true",
        help="Count frames whose last 4 bytes are not placeholder 0x55555555 (FM2 mode only)",
    )
    args = p.parse_args()

    if args.inspect:
        inspect_sync_layout(args.input_file)
        return

    extract_raw_nand(
        args.input_file,
        args.output,
        skip_prefix=args.skip_prefix,
        legacy_2044=args.legacy_2044,
        check_fletcher_placeholder=args.check_fletcher,
        skip_all_ff_payloads=True,
    )


if __name__ == "__main__":
    main()
