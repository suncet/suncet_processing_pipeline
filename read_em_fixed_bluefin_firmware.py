#!/usr/bin/env python3
"""
**Bluefin fixed-firmware EM / CSIE** ingest helper: like ``read_em_bla_correct_extra_word.py``,
but **defaults to CCSDS length-delimited** packet walking (no ASM between packets), matching
**parsed** streams where flight software no longer leaves ``1A CF FC 1D`` inside TF payloads.

Older **EM** parses that still contain embedded ASMs (e.g. ``2026-05-08_em_xband_downlink_test``)
need **``--sync-split``** so segments match the legacy reader.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

# Test pattern (suncet_csie_configuration_translator) for ground-truth diff vs parsed CSIE
_DEFAULT_REFERENCE_FITS = os.path.join(
    os.path.expanduser("~/Dropbox/suncet_dropbox/9000 Processing/code/"),
    "suncet_csie_configuration_translator",
    "test_pattern_96.fits",
)

# Default: FM2 parse next to ``parse_tf_raw_nand_for_csie.py`` (``-o parsed_output_raw_csie_12p5mbps_4sps``).
_BLUEFIN_TEST_DIR = os.path.join(
    os.path.expanduser("~/Dropbox/suncet_dropbox/9000 Processing/data/test_data"),
    "2026-05-08_em_xband_fixed_bluefin_firmware_downlink_test",
)
_DEFAULT_INPUT = os.path.join(
    _BLUEFIN_TEST_DIR, "parsed_2x_bitwise"
)


SYNC_PATTERN = b"\x1a\xcf\xfc\x1d"
SYNC_LEN = 4
PRIMARY_LEN = 6
SECONDARY_LEN = 6
CHECKSUM_LEN = 4
CSIE_DATA_APID = 536


def strip_leading_0xff(data: bytes) -> bytes:
    i = 0
    n = len(data)
    while i < n and data[i] == 0xFF:
        i += 1
    return data[i:]


def remove_consecutive_repeated_4byte_words(data: bytes) -> bytes:
    """
    Sequentially scan ``data``; whenever ``data[i:i+4] == data[i+4:i+8]**, emit the first word
    only and advance by **8** bytes (drop the second copy). Otherwise emit one byte and advance
    by 1. This removes *consecutive* duplicate 32-bit words at any offset.
    """
    b = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if i + 8 <= n and data[i : i + 4] == data[i + 4 : i + 8]:
            b.extend(data[i : i + 4])
            i += 8
        else:
            b.append(data[i])
            i += 1
    return bytes(b)


def split_by_sync(data: bytes) -> list[bytes]:
    """
    Return raw bytes for each inter-sync **frame** (region after one sync, before the next).

    Legacy captures embed ASM inside the TF payload stream; **new** FM2-style parses often
    have **no** ASMs in the concatenated bytes—use ``iter_ccsds_space_packets`` instead.
    """
    if not data:
        return []
    matches = [m.start() for m in re.finditer(re.escape(SYNC_PATTERN), data)]
    if not matches:
        return [data]
    out: list[bytes] = []
    for a, b in zip(matches, matches[1:] + [len(data)]):
        out.append(data[a + SYNC_LEN : b])
    return out


# Upper bound for CCSDS space packet size (header + data field); avoids runaway on mis-sync.
_MAX_CCSDS_PACKET_BYTES = 6 + 65536


def ccsds_packet_byte_length_from_prefix(prefix6: bytes) -> int | None:
    """Total packet length in bytes from first 6 bytes, or None if not a plausible v0 header."""
    if len(prefix6) < PRIMARY_LEN:
        return None
    h = parse_space_packet_header(prefix6)
    if h is None:
        return None
    _apid, _seq, dlen = h
    if dlen < 1 or dlen > 65536:
        return None
    total = PRIMARY_LEN + dlen
    if total > _MAX_CCSDS_PACKET_BYTES:
        return None
    return total


def iter_ccsds_space_packets(data: bytes) -> list[bytes]:
    """
    Split ``data`` into space packets using only the CCSDS primary header length (no ASM).

    If the stream does not start on a packet boundary, advances one byte at a time until a
    valid v0 header and length are seen (same idea as a UART bit-slip recovery).
    """
    if not data:
        return []
    out: list[bytes] = []
    i = 0
    n = len(data)
    while i + PRIMARY_LEN <= n:
        tot = ccsds_packet_byte_length_from_prefix(data[i : i + PRIMARY_LEN])
        if tot is None or i + tot > n:
            i += 1
            continue
        out.append(data[i : i + tot])
        i += tot
    return out


def parse_space_packet_header(packet: bytes) -> tuple[int, int, int] | None:
    """(apid, ccsds_sequence, data_field_length) or None if too short / invalid version."""
    if len(packet) < PRIMARY_LEN:
        return None
    if ((packet[0] >> 5) & 0x7) != 0:
        return None
    apid = int.from_bytes(packet[0:2], "big") & 0x7FF
    seq = int.from_bytes(packet[2:4], "big") & 0x3FFF
    dlen = int.from_bytes(packet[4:6], "big") + 1
    return apid, seq, dlen


def extract_csie_data_row_bytes(
    packet: bytes,
) -> tuple[int, int, bytes, int, bytes | None] | None:
    """
    From one CCSDS space packet, return
    (``image_id``, 1-based row index, **row** big-endian uint16 **wire** ``bytes``,
    ``n_cols``, **wire_checksum** or ``None``) for APID-536 (``csie_data``) rows, or None.

    The data field is: 6-byte secondary + ``n_cols``×2 row pixels + 4-byte **additive**
    checksum (not Fletcher-32). Pixels for assembly are taken **without** the checksum bytes;
    we still **do not** reject rows or drop packets based on checksum (no validation here).
    ``wire_checksum`` is returned for future **additive** checks if needed.

    The CSIE **secondary** is six bytes: U32 ``image_id`` (big endian), U16 (ICD).
    Row index is the CCSDS **sequence** field (1-based), matching ``make_level0_5``.
    """
    h = parse_space_packet_header(packet)
    if h is None or h[0] != CSIE_DATA_APID:
        return None
    _apid, seq, dlen = h
    if len(packet) < PRIMARY_LEN + SECONDARY_LEN:
        return None
    image_id = int.from_bytes(packet[PRIMARY_LEN : PRIMARY_LEN + 4], "big")
    p0 = PRIMARY_LEN + SECONDARY_LEN
    expected_data_len = dlen - SECONDARY_LEN - CHECKSUM_LEN
    if expected_data_len < 2 or (expected_data_len & 1):
        return None
    need_total = PRIMARY_LEN + dlen
    wire_checksum: bytes | None
    if len(packet) >= need_total:
        if len(packet) < p0 + expected_data_len + CHECKSUM_LEN:
            return None
        rowb = packet[p0 : p0 + expected_data_len]
        wire_checksum = bytes(
            packet[p0 + expected_data_len : p0 + expected_data_len + CHECKSUM_LEN]
        )
    else:
        raw = packet[p0:]
        n = len(raw)
        if n >= expected_data_len + CHECKSUM_LEN:
            rowb = raw[:expected_data_len]
            wire_checksum = bytes(
                raw[expected_data_len : expected_data_len + CHECKSUM_LEN]
            )
        elif n >= expected_data_len:
            rowb = raw[:expected_data_len]
            wire_checksum = None
        else:
            rowb = raw
            wire_checksum = None
        if not rowb or (len(rowb) & 1):
            return None
    if not rowb or (len(rowb) & 1):
        return None
    ncols = len(rowb) // 2
    return image_id, seq, rowb, ncols, wire_checksum


def diagnose_extract_failure(packet: bytes) -> str:
    """Rough reason ``extract_csie_data_row_bytes`` is None (call only when extract already failed)."""
    h = parse_space_packet_header(packet)
    if h is None:
        if len(packet) < PRIMARY_LEN:
            return f"short_primary len={len(packet)}"
        if ((packet[0] >> 5) & 0x7) != 0:
            return f"not_ccsds_v0 b0=0x{packet[0]:02x}"
        return "header_parse_failed"
    apid, seq, dlen = h
    if apid != CSIE_DATA_APID:
        return f"apid={apid} seq={seq} len={len(packet)}"
    if len(packet) < PRIMARY_LEN + SECONDARY_LEN:
        return f"short_secondary len={len(packet)}"
    expected_data_len = dlen - SECONDARY_LEN - CHECKSUM_LEN
    if expected_data_len < 2 or (expected_data_len & 1):
        return f"bad_row_wire_len edl={expected_data_len} dlen={dlen}"
    need_total = PRIMARY_LEN + dlen
    p0 = PRIMARY_LEN + SECONDARY_LEN
    if len(packet) >= need_total:
        if len(packet) < p0 + expected_data_len + CHECKSUM_LEN:
            return f"truncated_full_packet need={need_total} len={len(packet)}"
    else:
        raw = packet[p0:]
        n = len(raw)
        if n < expected_data_len:
            return f"truncated_row raw_tail={n} need_edl={expected_data_len}"
        if not raw[:expected_data_len] or (len(raw[:expected_data_len]) & 1):
            return "row_bytes_empty_or_odd"
    head = packet[: min(24, len(packet))].hex()
    return f"extract_failed_other len={len(packet)} head24={head}"


def probe_preprocessing(raw: bytes, *, use_sync_split: bool) -> None:
    """Print packet counts and top skip reasons for no-dedup vs dedup (quick A/B)."""
    from collections import Counter

    def split(d: bytes) -> list[bytes]:
        if use_sync_split:
            return split_by_sync(d)
        return iter_ccsds_space_packets(d)

    for label, use_dedup in (("no_dedup", False), ("dedup", True)):
        d = remove_consecutive_repeated_4byte_words(raw) if use_dedup else raw
        d = strip_leading_0xff(d)
        frames = split(d)
        reasons: Counter[str] = Counter()
        n_ok = 0
        for fr in frames:
            if extract_csie_data_row_bytes(fr) is not None:
                n_ok += 1
            else:
                reasons[diagnose_extract_failure(fr)] += 1
        split_name = "ASM split" if use_sync_split else "CCSDS length"
        print(
            f"--- probe {label} ({split_name}) --- bytes={len(d)} "
            f"segments={len(frames)} APID-536_ok={n_ok}"
        )
        for reason, cnt in reasons.most_common(12):
            print(f"  {cnt:7d}  {reason}")


def row_bytes_to_u16be(row: bytes) -> np.ndarray:
    """Device send order is big-endian uint16; match ``make_level0_5`` (byteswap to native u16)."""
    return np.frombuffer(row, dtype=np.uint16).byteswap().copy()


def assemble_images(
    row_map: dict[int, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """``row_map[image_id][1-based row seq] = 1D uint16 row`` → full images."""
    out: dict[int, np.ndarray] = {}
    for image_id, rows in row_map.items():
        if not rows:
            continue
        max_row = max(rows)
        n_rows = int(max_row)
        col_candidates = {int(r.size) for r in rows.values() if r.size}
        n_cols = max(col_candidates) if col_candidates else 0
        if n_rows < 1 or n_cols < 1:
            continue
        im = np.zeros((n_rows, n_cols), dtype=np.uint16)
        for r1, rvec in rows.items():
            r0 = int(r1) - 1
            if 0 <= r0 < n_rows:
                c = min(n_cols, int(rvec.size))
                im[r0, :c] = rvec[:c].reshape(1, c)
        out[image_id] = im
    return out


def preview_for_png(image: np.ndarray) -> np.ndarray:
    """Array in assembly order for ``imsave`` / ``imshow``; ``origin=upper`` in callers."""
    return np.asanyarray(image, dtype=np.uint16)


def _configure_matplotlib_for_show(show: bool) -> None:
    """``show``: interactive window backend; else headless ``Agg`` (must run before any pyplot)."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    elif sys.platform == "darwin":
        matplotlib.use("macosx", force=True)
    else:
        matplotlib.use("TkAgg", force=True)


# Inferno clim for per-pixel difference (absolute) in float space
INFERNO_DIFF_VMAX: float = 2.0


def write_inferno_png(path: str, array_u16: np.ndarray, *, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    preview = preview_for_png(array_u16)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.imsave(path, preview, cmap="inferno", origin="upper")
    if show:
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
                f"CSIE: {os.path.basename(path)}"
            )
        except (AttributeError, TypeError):
            pass
        ax.imshow(preview, cmap="inferno", origin="upper")
        ax.set_axis_off()
    print(f"Wrote {path}  (shape {array_u16.shape[0]}×{array_u16.shape[1]})")


def show_interactive_top_row_zoom(
    array_u16: np.ndarray, *, label: str = "CSIE top row (y=0) · inferno vmin=0 vmax=2"
) -> None:
    """
    Only when ``--show``: wide, short figure with **one** row of the image, inferno, ``vmax=2``.
    """
    if array_u16.shape[0] < 1 or array_u16.shape[1] < 1:
        return
    import matplotlib.pyplot as plt

    row0 = array_u16[0:1, :].astype(np.float64)
    fig, ax = plt.subplots(figsize=(16, 2.2), dpi=100)
    fig.patch.set_facecolor("black")
    im = ax.imshow(
        row0, cmap="inferno", origin="upper", aspect="auto", vmin=0.0, vmax=INFERNO_DIFF_VMAX
    )
    ax.set_title(label, color="white", fontsize=10)
    ax.set_axis_off()
    try:
        fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
            "CSIE: top row zoom (inferno 0..2)"
        )
    except (AttributeError, TypeError):
        pass
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors="0.85")


def load_reference_fits_array(path: str) -> np.ndarray:
    from astropy.io import fits

    p = os.path.expanduser(path)
    data = fits.getdata(p)
    if data is None:
        raise SystemExit(f"No data array in FITS: {p}")
    a = np.asanyarray(data)
    if a.itemsize != 2 or a.dtype.kind not in "iu":
        raise SystemExit(
            f"Reference FITS must be 16-bit integer for CSIE diff; got {a.dtype!r} in {p}"
        )
    # Match parsed CSIE (uint16): re-interpret bit patterns as unsigned (FITS is often int16)
    return np.asanyarray(a, dtype=np.uint16)


def overlap_parsed_ref_diff(
    parsed: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """
    Top-left ``min`` overlap in each axis: ``(parsed_crop, ref_crop, diff, note|None)``.

    ``diff = parsed_crop − ref_crop`` in float64 (handles e.g. a missing row in the capture).
    """
    h = min(int(parsed.shape[0]), int(ref.shape[0]))
    w = min(int(parsed.shape[1]), int(ref.shape[1]))
    if h < 1 or w < 1:
        raise SystemExit(
            f"No overlap: parsed {parsed.shape} vs reference FITS {ref.shape}"
        )
    p_sub = parsed[:h, :w]
    r_sub = ref[:h, :w]
    diff = p_sub.astype(np.float64) - r_sub.astype(np.float64)
    if tuple(parsed.shape) == tuple(ref.shape):
        return p_sub, r_sub, diff, None
    return p_sub, r_sub, diff, (
        f"note: top-left {h}×{w} overlap (parsed {parsed.shape[0]}×{parsed.shape[1]}"
        f" vs FITS {ref.shape[0]}×{ref.shape[1]})"
    )


def print_and_format_diff_stats(
    diff: np.ndarray, overlap_note: str | None = None
) -> str:
    """Log statistics to stdout; return the same text for the figure annotation."""
    d = diff.ravel()
    n = d.size
    lines = [
        "Difference: parsed - reference (FITS)",
    ]
    if overlap_note:
        lines.append(overlap_note)
    lines.extend(
        [
            f"shape: {diff.shape[0]} x {diff.shape[1]}  (n={n})",
            f"min:   {float(np.min(d)):.6g}",
            f"max:   {float(np.max(d)):.6g}",
            f"mean:  {float(np.mean(d)):.6g}",
            f"std:   {float(np.std(d)):.6g}",
            f"rmse:  {float(np.sqrt(np.mean(d**2))):.6g}",
            f"max|d|: {float(np.max(np.abs(d))):.6g}",
            f"count (|d|>0): {int(np.count_nonzero(d))} / {n}",
        ]
    )
    print("\n=== Reference FITS difference ===")
    for line in lines:
        print(line)
    return "\n".join(lines)


def write_diff_inferno_png(
    out_path: str, diff: np.ndarray, annotation: str, *, show: bool = False
) -> None:
    import matplotlib.pyplot as plt

    vis = np.asanyarray(diff, dtype=np.float64)
    ad = np.abs(vis)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=120)
    dmax = float(np.max(ad))
    if dmax == 0.0:
        ax.imshow(vis, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(
            ad,
            cmap="inferno",
            origin="upper",
            vmin=0.0,
            vmax=INFERNO_DIFF_VMAX,
        )
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.65),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.patch.set_facecolor("black")
    fig.savefig(out_path, bbox_inches="tight", facecolor="black", pad_inches=0.1)
    if not show:
        plt.close(fig)
    print(f"Wrote difference PNG {out_path}  (shape {diff.shape[0]}×{diff.shape[1]})")


def write_parsed_ref_diff_panel_png(
    out_path: str,
    p_sub: np.ndarray,
    r_sub: np.ndarray,
    diff: np.ndarray,
    annotation: str,
    *,
    reference_title: str = "Reference (FITS)",
    diff_title: str = "Difference (parsed − ref)",
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    p_vis = np.asanyarray(p_sub, dtype=np.float64)
    r_vis = np.asanyarray(r_sub, dtype=np.float64)
    d_vis = np.asanyarray(diff, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(bottom=0.2, wspace=0.08, top=0.9)
    for ax, im, title, is_diff in (
        (axes[0], p_vis, "Parsed (CSIE)", False),
        (axes[1], r_vis, reference_title, False),
        (axes[2], d_vis, diff_title, True),
    ):
        if is_diff:
            ad = np.abs(np.asanyarray(im, dtype=np.float64))
            dmax = float(np.max(ad))
            if dmax == 0.0:
                ax.imshow(ad, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(
                    ad,
                    cmap="inferno",
                    origin="upper",
                    vmin=0.0,
                    vmax=INFERNO_DIFF_VMAX,
                )
        else:
            ax.imshow(
                im,
                cmap="inferno",
                origin="upper",
                vmin=0.0,
                vmax=65535.0,
            )
        ax.set_title(title, color="white", fontsize=11)
        ax.set_axis_off()
    fig.text(
        0.5,
        0.01,
        annotation,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=6,
        color="0.85",
        family="monospace",
        linespacing=1.15,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="black", pad_inches=0.2)
    if not show:
        plt.close(fig)
    print(f"Wrote parsed/ref/diff panel PNG {out_path}  (shape {diff.shape[0]}×{diff.shape[1]})")


def emit_reference_diff(
    main_png_path: str, arr: np.ndarray, ref: np.ndarray, *, show: bool = False
) -> None:
    p_sub, r_sub, diff, overlap_note = overlap_parsed_ref_diff(arr, ref)
    ann = print_and_format_diff_stats(diff, overlap_note)
    base, ext = os.path.splitext(main_png_path)
    diff_path = f"{base}_diff{ext}"
    write_diff_inferno_png(diff_path, diff, ann, show=show)
    triple_path = f"{base}_parsed_ref_diff{ext}"
    write_parsed_ref_diff_panel_png(triple_path, p_sub, r_sub, diff, ann, show=show)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bluefin EM CSIE reader: CCSDS length-delimited packets by default; optional dedup / ASM split"
    )
    p.add_argument(
        "input",
        nargs="?",
        default=_DEFAULT_INPUT,
        help="Parsed CSIE binary (default: Bluefin parsed_output_raw_csie_12p5mbps_4sps)",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output PNG path (default: <input_basename>_csie.png next to the input file)",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="Do not load reference FITS, difference stats, _diff.png, or parsed/ref panel PNGs",
    )
    p.add_argument(
        "--reference-fits",
        default=_DEFAULT_REFERENCE_FITS,
        metavar="PATH",
        help="FITS test pattern to subtract (default: test_pattern_96.fits next to config translator; loaded as uint16)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open Matplotlib window(s) for the CSIE, diff, and triptych figures (keeps run alive until you close all)",
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Apply consecutive duplicate 4-byte word removal (same as read_em_bla_correct_extra_word default)",
    )
    p.add_argument(
        "--sync-split",
        action="store_true",
        help="Use ASM 1A CF FC 1D between segments (legacy EM); default is CCSDS length only",
    )
    args = p.parse_args()
    _configure_matplotlib_for_show(args.show)

    in_path = os.path.expanduser(args.input)
    with open(in_path, "rb") as f:
        data = f.read()

    print(f"Input: {in_path}  ({len(data)} bytes)\n")
    probe_preprocessing(data, use_sync_split=args.sync_split)
    print()

    # Set a breakpoint on the next line (or uncomment) to inspect raw bytes / first frames.
    # breakpoint()

    n0 = len(data)
    if args.dedup:
        data = remove_consecutive_repeated_4byte_words(data)
        n1 = len(data)
        if n1 != n0:
            print(
                f"Consecutive 4-byte dedup: {n0} -> {n1} bytes ({n0 - n1} byte(s) removed)\n"
            )
    else:
        print("No consecutive 4-byte dedup (default for this script). Use --dedup to enable.\n")
    data = strip_leading_0xff(data)
    if args.sync_split:
        frames = split_by_sync(data)
    else:
        frames = iter_ccsds_space_packets(data)

    row_by_image: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    n_seen = 0
    n_skip = 0
    for fr in frames:
        ext = extract_csie_data_row_bytes(fr)
        if ext is None:
            n_skip += 1
            continue
        image_id, seq, rowb, _n_cols, _row_checksum = ext
        n_seen += 1
        rvec = row_bytes_to_u16be(rowb)
        row_by_image[image_id][seq] = rvec.ravel()

    if not any(row_by_image.values()):
        raise SystemExit(
            "No APID-536 (csie_data) rows were recovered. "
            "Check CCSDS alignment / try --sync-split for legacy ASM-delimited captures."
        )

    images = assemble_images(row_by_image)
    if not images:
        raise SystemExit("Could not assemble any images from the parsed rows.")

    in_base, _ = os.path.splitext(in_path)
    n_img = len(images)
    ref: np.ndarray | None = None
    if not args.no_reference:
        ref = load_reference_fits_array(args.reference_fits)
    if n_img == 1:
        image_id, arr = next(iter(images.items()))
        if args.output:
            op = os.path.expanduser(args.output)
            if os.path.isdir(op):
                op = os.path.join(op, f"csie_id{image_id}.png")
        else:
            op = f"{in_base}_csie.png"
        write_inferno_png(op, arr, show=args.show)
        if args.show:
            show_interactive_top_row_zoom(arr, label="CSIE top row (y=0) · inferno [0, 2]")
        if ref is not None:
            emit_reference_diff(op, arr, ref, show=args.show)
    else:
        for image_id, arr in images.items():
            if args.output:
                op = os.path.expanduser(args.output)
                if os.path.isdir(op):
                    pth = os.path.join(op, f"csie_id{image_id}.png")
                else:
                    stem, ext = os.path.splitext(op)
                    if ext.lower() == ".png":
                        pth = f"{stem}_id{image_id}{ext}"
                    else:
                        pth = f"{in_base}_csie_id{image_id}.png"
            else:
                pth = f"{in_base}_csie_id{image_id}.png"
            write_inferno_png(pth, arr, show=args.show)
            if args.show:
                show_interactive_top_row_zoom(
                    arr, label=f"CSIE id={image_id} top row (y=0) · inferno [0, 2]"
                )
            if ref is not None:
                emit_reference_diff(pth, arr, ref, show=args.show)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    if n_skip:
        print(
            f"Note: {n_skip} packet segment(s) did not parse as complete APID-536 "
            f"(filler, metadata APID-538, or other)."
        )
    print(f"Decoded {n_seen} csie_data row packet(s) into {len(images)} image(s).")


if __name__ == "__main__":
    main()
